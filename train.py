#!/usr/bin/env python3
import torch
import os
from Dataloaders import CustomDataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from torch.optim import SGD
from Augmentations import get_augmentations
from monai.losses import DiceFocalLoss, DiceLoss, DiceCELoss, AsymmetricUnifiedFocalLoss
from swin_unetr import SwinUNETR
import sys
import numpy as np
from lrScheduler import PolyLRScheduler
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel
import re
import random
from monai.utils import DiceCEReduction, LossReduction, Weight, deprecated_arg, look_up_option, pytorch_after
from collections.abc import Callable, Sequence
from skimage.transform import resize
import time
# 5 fold cross validation library
from sklearn.model_selection import KFold
from threading import Thread
import wandb
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*")


class SWINUNETRTrainer(object):
    def __init__(self, model=None, optimizer=None, weights=None, device='0', continue_tr = False, fold = '', dataset_dir = '', batch_size = 2, roi = True, LR = False):

        self.initial_lr = 0.01
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 500
        self.num_val_iterations_per_epoch = 250
        self.num_epochs = 500
        self.current_epoch = 0
        self.enable_deep_supervision = True
        self.device = device
        self.best_loss = 1000
        self.loss = 1000
        self.dataset_dir = dataset_dir
        self.fold = fold
        self.continue_tr = continue_tr
        self.batch_size = batch_size
        self.divis_x = [2]
        self.divis_y = [5, 6, 7, 8]
        self.divis_z = [5, 6, 7, 8, 9, 10]
        self.max_dims = [4, 10, 13]
        self.num_accumulation = 4
        self.roi = roi
        self.LR = LR

        torch.device("cuda:0")

        if continue_tr:
            cwd = os.getcwd()
            dataset_name = self.dataset_dir.split('/')[-3]
            self.fold_dir = os.path.join(cwd, dataset_name, f'fold_{self.fold}')
            self.load_most_recent_checkpoint(self.fold_dir)
        else:
            if model is not None:
                self.model = model.cuda(device)
            else:
                self.model = SwinUNETR(img_size=(32, 160, 256), in_channels=1, out_channels=1, feature_size=24, deep_supervision = self.enable_deep_supervision, use_v2=True).cuda(self.device);
            
            if optimizer is not None:
                self.optimizer = optimizer
            else:
                self.optimizer = SGD(self.model.parameters(), lr=self.initial_lr, momentum=0.99, weight_decay=self.weight_decay, nesterov=True)
                #self.optimizer = AdamW(self.model.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)

            if weights is not None:
                self.model.load_state_dict(weights)
            

        if self.enable_deep_supervision:
            self.weights = np.array([1/(2.5**i) for i in range(6)])
            #reverse
            self.weights[-1] = 0
            self.weights = self.weights/self.weights.sum()

        self.criterion = DiceLoss(include_background=False, sigmoid=True)

        self.lr = self.optimizer.param_groups[0]['lr']

        self.lr_scheduler = PolyLRScheduler(self.optimizer, initial_lr=self.lr, max_steps=self.num_epochs*self.num_iterations_per_epoch//256, exponent=0.9, current_step=self.current_epoch*self.num_iterations_per_epoch//256) 
            
    def load_most_recent_checkpoint(self, fold_dir):
        #check for pt file 

        #if pt file exists, load the model and optimizer state

        if os.path.exists(os.path.join(fold_dir, 'checkpoint.pt')):
            checkpoint = torch.load(os.path.join(fold_dir, 'checkpoint.pt'))
            self.model = checkpoint['model_arch']
            self.optimizer = checkpoint['optimizer']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.loss = checkpoint['loss']
            self.best_loss = checkpoint['best_loss']
            print(f"Checkpoint found. Continuing training from epoch {self.current_epoch} with loss {self.loss}")
        else:
            print("WARNING: There is no checkpoint to load from. Training from scratch.")
            self.model = SwinUNETR(img_size=(32, 160, 256), in_channels=1, out_channels=1, feature_size=24, use_v2 = True, enable_deep_supervision = self.enable_deep_supervision).cuda(self.device) # create an augmentation that will randomly decrease the image size so we get some lower res images
            self.optimizer = SGD(self.model.parameters(), lr=self.initial_lr, momentum=0.99, weight_decay=self.weight_decay, nesterov=True)
            self.current_epoch = 0
            self.loss = 1000
            self.best_loss = 1000
    

    def run_training(self):
        # check if a dataset folder has already been created
        
        wdir = os.getcwd()
        # break dataset folder by /
        dataset_name = self.dataset_dir.split('/')[-3] # fix how dataset folders are created
        #check if dataset folder exists
        if not os.path.exists(self.dataset_dir):
            raise ValueError(f"Dataset folder {self.dataset_dir} does not exist.")
        
        if not os.path.exists(os.path.join(wdir, dataset_name)):
            os.makedirs(os.path.join(wdir, dataset_name))

        #check if fold folder exists, if not create one, if it does and c is not true warn the user that you will be overwriting the original fold

        if not os.path.exists(os.path.join(wdir, dataset_name, f'fold_{self.fold}')):
            os.makedirs(os.path.join(wdir, dataset_name, f'fold_{self.fold}'))
            self.fold_dir = os.path.join(wdir, dataset_name, f'fold_{self.fold}')
        else:
            # check if checkpoint file exists
            temp_files = os.listdir(os.path.join(wdir, dataset_name, f'fold_{self.fold}'))
            if 'checkpoint.pt' in temp_files and not self.continue_tr:
                raise ValueError(f"Fold {self.fold} already exists. If you do not want to overwrite the fold, include --c") # fix this so that the log file is overwritten if there is no checkpoint file
            self.fold_dir = os.path.join(wdir, dataset_name, f'fold_{self.fold}')
            
        fold_dir = os.path.join(wdir, dataset_name, f'fold_{self.fold}')
        print(f"Training fold {self.fold} at {fold_dir}")
        
        files = os.listdir(fold_dir)

        log_file_name = "log0.txt"

        i = 0
        while log_file_name in files:
            i += 1
            log_file_name = f"log{i}.txt"

        log_file_path = os.path.join(fold_dir, log_file_name)

        log_file = open(log_file_path, 'w')

        sys.stdout = log_file

        files = os.listdir(self.dataset_dir)

        files = np.array(files, dtype=str) # operation results in nonetype

        assert self.fold[0].isdigit(), "Fold number must be an integer between 0 and 4"

        # check fold number
        folds = KFold(n_splits=5, shuffle=True, random_state=42)
        # get fold based on fold number
        for i, (train_index, test_index) in enumerate(folds.split(files)):
            if i == int(self.fold[0]):
                train_files = files[train_index]
                val_files = files[test_index]
                np.save(os.path.join(fold_dir, f'val_files_{self.fold}.npy'), val_files)
                print(f"Training on fold {self.fold}")
                break

        train_files = [self.dataset_dir + file for file in train_files]
        val_files = [self.dataset_dir + file for file in val_files]


        test_files = os.listdir(self.dataset_dir.replace('Tr', 'Ts'))
        test_files = [self.dataset_dir.replace('Tr', 'Ts') + file for file in test_files]

        train_loader = CustomDataLoader(train_files, batch_size=self.batch_size, LR=self.LR)
        #val_loader = CustomDataLoader(val_files, batch_size=1, val=True, LR=self.LR) # not using val loader because we more care about the test set
        test_loader = CustomDataLoader(test_files, batch_size=1, val=True, LR=self.LR)

        transforms = get_augmentations()

        self.train_loader = MultiThreadedAugmenter(train_loader, transforms, num_processes=16, num_cached_per_queue=4, pin_memory=True, useroi=self.roi)
        #self.val_loader = MultiThreadedAugmenter(val_loader, None, num_processes=8, num_cached_per_queue=3, pin_memory=False)
        self.test_loader = MultiThreadedAugmenter(test_loader, None, num_processes=4, num_cached_per_queue=3, pin_memory=True, useroi=self.roi, val=True)
        self.train_files = train_files
        self.transforms = transforms

        self.train_model()

    def calculate_roi(self, data, mask, roi, val=False):
        
        coords = np.where(roi == 1)
        try:
            # randomly we feed in the whole scan or do a random crop
            min_coords = [np.min(coords[i]) for i in range(2, 5)]
            max_coords = [np.max(coords[i]) for i in range(2, 5)]
            # we add 15 to the x coordinate because I am a nice guy :)
            if max_coords[0] - min_coords[0] < 64 and not val:
                const = 64 - (max_coords[0] - min_coords[0])
                neg_const = 0
                if np.random.rand() > 0.5:
                    neg_const = min_coords[0]
                    const = np.max(np.array([64 - (max_coords[0] - min_coords[0] - neg_const), 0]))
            else:
                const = 64 - (max_coords[0] - min_coords[0])
                neg_const = 0

            if val:
                const_y = 32 - ((max_coords[1] - min_coords[1]) % 32)
                const_z = 32 - ((max_coords[2] - min_coords[2]) % 32)
                data = data[:, :, min_coords[0]:max_coords[0]+const, min_coords[1]:max_coords[1]+const_y, min_coords[2]:max_coords[2]+const_z]
                mask = mask[:, :, min_coords[0]:max_coords[0]+const, min_coords[1]:max_coords[1]+const_y, min_coords[2]:max_coords[2]+const_z]

            else:
                    
                data = data[:, :, min_coords[0]-neg_const:max_coords[0]+const, min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
                mask = mask[:, :, min_coords[0]-neg_const:max_coords[0]+const, min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
                
        except Exception as e:
            print(f"Error in calculating ROI: {e}")

        return data, mask

    
    def generate_patch_size(self, inputs, labels, val=False):
        '''
        The patch size for swin unetr can be any size that is a multiple of 32
        '''
        min_x = 1000
        min_y = 1000
        min_z = 1000
        for input in inputs:
            min_x = min(min_x, input.shape[1])
            min_y = min(min_y, input.shape[2])
            min_z = min(min_z, input.shape[3])

        factors_x = min_x // 32
        factors_y = min_y // 32
        factors_z = min_z // 32

        #if val we select largest possible patch
        if val:
            max_x = inputs[0].shape[1] // 32
            max_y = inputs[0].shape[2] // 32
            max_z = inputs[0].shape[3] // 32


            patch = (max_x*32, max_y*32, max_z*32)

            new_inputs = np.zeros((1, 1, patch[0], patch[1], patch[2]))
            new_labels = np.zeros((1, 1, patch[0], patch[1], patch[2])) # this was changed

            for i in range(inputs.shape[0]):
                new_inputs[i], new_labels[i] = crop(inputs[i:i+1], labels[i:i+1], patch, crop_type='random')

            return torch.from_numpy(new_inputs).float().cuda(0), torch.from_numpy(new_labels).float().cuda(0)
        
        # select randomly factors that are less than or equal to the min
        possible_x = [factor for factor in self.divis_x if factor <= factors_x]
        possible_y = [factor for factor in self.divis_y if factor <= factors_y]
        possible_z = [factor for factor in self.divis_z if factor <= factors_z]

        if len(possible_x) == 0:
            possible_x = [2]

        if len(possible_y) == 0:
            possible_y = [factors_y]
        
        if len(possible_z) == 0:
            possible_z = [factors_z]

        patch = (np.random.choice(possible_x)*32, np.random.choice(possible_y)*32, np.random.choice(possible_z)*32)

        new_inputs = np.zeros((self.batch_size, 1, patch[0], patch[1], patch[2]))
        new_labels = np.zeros((self.batch_size, 1, patch[0], patch[1], patch[2]))

        for i in range(inputs.shape[0]):
            c0, c1, c2 = int(np.random.uniform(0, inputs.shape[2]-patch[0])), int(np.random.uniform(0, inputs.shape[3]-patch[1])), int(np.random.uniform(0, inputs.shape[4]-patch[2]))
            new_inputs[i], new_labels[i] = inputs[i:i+1, :, c0:patch[0]+c0, c1:patch[1]+c1, c2:patch[2]+c2], labels[i:i+1, :, c0:patch[0]+c0, c1:patch[1]+c1, c2:patch[2]+c2] # because the crop function is slow as shit

        return new_inputs, new_labels

    def create_new_dataloader(self):
        train_loader = CustomDataLoader(self.train_files, batch_size=self.batch_size, LR=self.LR)
        train_loader = MultiThreadedAugmenter(train_loader, self.transforms, num_processes=12, num_cached_per_queue=5, pin_memory=True, useroi=self.roi)
        train_loader.restart()
        train_loader.generator.shuffle_indices()
        return train_loader

    def log_metrics(self, loss, calc_loss, split):
        loss_value = loss.detach().cpu().mean().item()
        calc_loss_value = calc_loss.detach().cpu().mean().item()
        if split == 'val':
            metrics = {'Test loss': loss_value, 'Test dice loss': calc_loss_value}
        metrics = {'Train loss': loss_value, 'Train dice loss': calc_loss_value}
        wandb.log(metrics)
    
    def async_log(self, loss, calc_loss, split='train'):
        log_thread = Thread(target=self.log_metrics, args=(loss, calc_loss, split,))
        log_thread.start()
        log_thread.join()

    def train_model(self):
        self.train_loader.restart()
        self.test_loader.restart()

        #criterion = DiceFocalLoss(include_background=True, sigmoid=False, squared_pred=True, weight=[.7, .3]) # onehot was changed to true. changed squared_pred to true
        criterion = TverskyLoss(include_background=True, sigmoid=False, alpha=0.75)
        criterion = FocalLoss(include_background=True, weight=[0.4, 1], gamma=4)
        criterion = AsymmetricUnifiedFocalLoss(gamma=0.30, delta=0.7)
        val_criterion = AsymmetricUnifiedFocalLoss(gamma=0.30, delta=0.7)
        '''criterion.lambda_focal = 1
        criterion.lambda_dice = 1'''
        #clean_criterion = torch.nn.MSELoss()
        calc_criterion = DiceLoss(include_background=False, sigmoid=False, to_onehot_y=False, squared_pred=True).requires_grad_(False)

        # login
        wandb.login()
        if self.continue_tr:
            # continue training with previous run id
            print("Continuing training from previous run")
            wandb.init(project="segmentation-pipeline", id="110n8a2l")
        else:
            wandb.init(project="segmentation-pipeline")

        wandb.watch(self.model)

        #train the model
        #save the model and optimizer state every epoch
        print(f"Starting training at epoch {self.current_epoch}")
        best_val = self.best_loss
        import time
        for epoch in range(self.current_epoch, self.num_epochs):
            loss_total = 0
            dice_total = 0
            num_iter = 0
            self.model.train()
            t1 = time.time()
            for i in range(self.num_iterations_per_epoch):
                self.optimizer.zero_grad()

                data = next(self.train_loader)

                if 'S1' in self.dataset_dir and (i + 1) % 250 == 0 and i < 300:
                    # creat copy of train loader
                    self.train_loader.restart()
                    self.train_loader.generator.shuffle_indices()
                

                inputs = data['data']
                #roi = data['seg'][:, 1:]
                labels = data['seg'] # this was changed to support onehot

                inputs, labels = inputs.cuda(), labels.cuda()

                if (i+1) % 256 == 0:
                    self.lr_scheduler.step()
                    print(f'Learning rate: {self.optimizer.param_groups[0]["lr"]}', flush=True)

                if labels[0].max() == 0:
                    print("skipping...")
                    continue

                outputs = self.model(inputs)

                temp_loss = 0
                temp_calc_loss = 0

                if self.enable_deep_supervision:
                    logits = outputs
                    for j, logit in enumerate(logits):
                        if j != 0:
                            down_labels = F.interpolate(labels, size=logit.shape[2:], mode='nearest')
                        else:
                            down_labels = labels

                        complement_labels = 1 - down_labels
                        down_labels = torch.cat((complement_labels, down_labels), 1)

                        temp_logit = F.sigmoid(logit)
                        complement_logit = 1 - temp_logit
                        logit = torch.cat((complement_logit, temp_logit), 1)

                        temp_loss += self.weights[j] * criterion(logit, down_labels)

                    out = F.sigmoid(outputs[0])
                    complement = 1 - out
                    out = torch.cat((complement, out), 1)
                    complement_labels = 1 - labels
                    labels = torch.cat((complement_labels, labels), 1)
                    temp_calc_loss += calc_criterion(out, labels)
                    outputs = outputs[0]
                else:
                    temp_output = F.sigmoid(outputs)
                    complement = 1 - temp_output
                    outputs = torch.cat((complement, temp_output), 1)
                    complement = 1 - labels
                    labels = torch.cat((complement, labels), 1)
                    temp_loss += criterion(outputs[:, :, ...], labels[:, :, ...])
                    temp_calc_loss += calc_criterion(outputs[:, :, ...], labels[:, :, ...])

                temp_loss.backward()
                self.optimizer.step()
                
                loss_total += temp_loss.detach().mean()
                dice_total += temp_calc_loss.detach().mean()
                num_iter += 1
                if self.enable_deep_supervision:
                    self.async_log(temp_loss, temp_calc_loss)
                else:
                    wandb.log({'train loss': temp_loss.mean().item(), 'train dice loss': temp_loss.mean().item()})

            print(f"Train {epoch} took {time.time() - t1} seconds", flush=True)

            loss_total = loss_total.item()
            dice_total = dice_total.item()
                    
            print(f'Epoch: {epoch}, Loss: {loss_total/(num_iter)}, Dice Loss: {dice_total/(num_iter)}', flush=True)
            self.train_loader.restart()
            self.train_loader.generator.shuffle_indices()
            self.model.eval()
            with torch.no_grad():
                loss_total = 0
                loss_dice = 0
                num_iter = 0
                t2 = time.time()
                for i in range(self.num_val_iterations_per_epoch):

                    data = next(self.test_loader)
                    inputs = data['data']
                    #roi = data['seg'][:, 1:]
                    labels = data['seg']

                    if labels[0].max() == 0:
                        continue

                    inputs, labels = inputs.cuda(), labels.cuda()

                    outputs = self.model(inputs)

                    if self.enable_deep_supervision:
                        outputs = outputs[0]
                    
                    labels = torch.cat((1-labels, labels), 1)
                    temp_output = F.sigmoid(outputs)
                    complement = 1 - temp_output
                    outputs = torch.cat((complement, temp_output), 1)

                    temp_loss_calc = calc_criterion(outputs, labels)

                    learned_loss = val_criterion(outputs[:, :, ...], labels[:, :, ...])

                    loss_total += learned_loss.mean()
                    loss_dice += temp_loss_calc.mean()
                    num_iter += 1


                    self.async_log(learned_loss, temp_loss_calc, split='val')

                loss_total = loss_total.item()
                loss_dice = loss_dice.item()

                self.test_loader.restart()
                self.test_loader.generator.shuffle_indices()
                print(f'Epoch: {epoch}, Test Loss: {loss_total/(num_iter)}, Test Dice Loss: {loss_dice/(num_iter)}', flush=True)
                loss_total = loss_total/num_iter
                if loss_total < best_val:
                    best_val = loss_total
                    # save model_dict, optimizer_dict, epoch, loss, best_loss(best_val), max_epoch (100)
                    torch.save({'model_arch': self.model, 'model_state_dict': self.model.state_dict(),
                                'optimizer': self.optimizer, 'optimizer_state_dict': self.optimizer.state_dict(),
                                'epoch': epoch, 'loss': loss_total, 'best_loss': best_val, 'max_epoch': self.num_epochs}, os.path.join(self.fold_dir, 'checkpoint_best.pt'))
                torch.save({'model_arch': self.model, 'model_state_dict': self.model.state_dict(),
                                'optimizer': self.optimizer, 'optimizer_state_dict': self.optimizer.state_dict(),
                                'epoch': epoch, 'loss': loss_total, 'best_loss': best_val, 'max_epoch': self.num_epochs}, os.path.join(self.fold_dir, 'checkpoint.pt'))

            print(f"Validation took:", t2 - time.time(), flush=True)
            print(f"epoch {epoch} took:", t1 - time.time(), flush=True)

from torch.nn.modules.loss import _Loss
from monai.losses.tversky import TverskyLoss
from monai.losses.focal_loss import FocalLoss

'''class TverskyFocalLoss(_Loss):
    def __init__(self, include_background=True, to_onehot_y=False, sigmoid=False, softmax=False, weight=[1, 1], alpha=0.5, beta=0.5, lambda_tversky=1, lambda_focal=1):
        self.tverskyloss = TverskyLoss(include_background=include_background, sigmoid=sigmoid, alpah=alpha, beta=beta)
        self.focalloss = FocalLoss(include_background=include_background, weight=weight)

        def forward(input, target):'''

class TverskyFocalLoss(_Loss):
    """
    Compute both Dice loss and Focal Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Focal Loss is shown in ``monai.losses.FocalLoss``.

    ``gamma`` and ``lambda_focal`` are only used for the focal loss.
    ``include_background``, ``weight`` and ``reduction`` are used for both losses
    and other parameters are only used for dice loss.

    """

    @deprecated_arg(
        "focal_weight", since="1.2", removed="1.4", new_name="weight", msg_suffix="please use `weight` instead."
    )
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        gamma: float = 2.0,
        focal_weight = None,
        weight = None,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
    ) -> None:
        """
        Args:
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            other_act: callable function to execute other activation layers, Defaults to ``None``.
                for example: `other_act = torch.tanh`. only used by the `DiceLoss`, not for `FocalLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            gamma: value of the exponent gamma in the definition of the Focal loss.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes).
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_focal: the trade-off weight value for focal loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        weight = focal_weight if focal_weight is not None else weight
        self.dice = TverskyLoss(
            include_background=include_background,
            reduction=reduction
        )
        self.focal = FocalLoss(
            include_background=include_background, to_onehot_y=False, gamma=gamma, weight=weight, reduction=reduction
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.to_onehot_y = to_onehot_y

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD]. The input should be the original logits
                due to the restriction of ``monai.losses.FocalLoss``.
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 (without one-hot encoding) nor the same as input.

        Returns:
            torch.Tensor: value of the loss.
        """
        if input.dim() != target.dim():
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} (nb dims: {len(input.shape)}) and {target.shape} (nb dims: {len(target.shape)}). "
                "if target is not one-hot encoded, please provide a tensor with shape B1H[WD]."
            )

        if target.shape[1] != 1 and target.shape[1] != input.shape[1]:
            raise ValueError(
                "number of channels for target is neither 1 (without one-hot encoding) nor the same as input, "
                f"got shape {input.shape} and {target.shape}."
            )

        if self.to_onehot_y:
            n_pred_ch = input.shape[1]
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)
        dice_loss = self.dice(input, target)
        focal_loss = self.focal(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss
        return total_loss

    
def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=str, 
                        help="The directory containing the dataset")
    parser.add_argument('fold', type=str, 
                        help="Fold of the 5-fold cross-validation. Should be an int between 0 and 4.")
    parser.add_argument('-pretrained_weights', type=str, default=None, required=False, 
                        help="Path to the pretrained weights.")
    parser.add_argument('--c', action='store_true', required=False, 
                        help="continue training from latest checkpoint")
    parser.add_argument('-device', type=int, required=False, default=0, 
                        help="Device to train the model on.")    
    parser.add_argument('-LR',type=bool, required=False, default=False)
    parser.add_argument('-roi', type=bool, required=False, default=True)
    args = parser.parse_args()


    trainer = SWINUNETRTrainer(weights=args.pretrained_weights, fold=args.fold, dataset_dir=args.dataset_dir, device=args.device, continue_tr=args.c, LR=args.LR, roi=args.roi)
    trainer.run_training()

if __name__ == '__main__':
    run_training_entry()