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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.modules.loss import _Loss
from monai.losses.tversky import TverskyLoss
from monai.losses.focal_loss import FocalLoss

# Add memory tracking utilities
def get_gpu_memory_usage(device=None):
    """Return GPU memory usage in MB for a specific device"""
    if torch.cuda.is_available():
        if device is None:
            # Get memory for all devices
            memory_stats = {}
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
                memory_stats[f"GPU {i}"] = torch.cuda.memory_allocated(i) / 1024 / 1024
            return memory_stats
        else:
            # Get memory for specific device
            device_idx = device if isinstance(device, int) else int(device.split(':')[-1])
            torch.cuda.synchronize(device_idx)
            return torch.cuda.memory_allocated(device_idx) / 1024 / 1024
    return 0

def get_tensor_size_mb(tensor):
    """Return tensor size in MB"""
    if tensor is None:
        return 0
    
    # Handle tuple or list of tensors
    if isinstance(tensor, (tuple, list)):
        return sum(get_tensor_size_mb(t) for t in tensor)
    
    # Handle regular tensor
    return tensor.element_size() * tensor.nelement() / 1024 / 1024

def log_memory_usage(tag, tensors_dict=None, device=None, verbose=True):
    """Log memory usage and tensor sizes"""
    memory_usage = get_gpu_memory_usage(device)
    
    if verbose:
        if isinstance(memory_usage, dict):
            # Multiple GPUs
            log_str = f"[{tag}] GPU Memory: " + ", ".join([f"{k}: {v:.2f} MB" for k, v in memory_usage.items()])
        else:
            # Single GPU
            log_str = f"[{tag}] GPU Memory: {memory_usage:.2f} MB"
        
        if tensors_dict:
            log_str += " | Tensors: "
            for name, tensor in tensors_dict.items():
                if tensor is not None:
                    size_mb = get_tensor_size_mb(tensor)
                    
                    # Handle shape display for different types
                    if isinstance(tensor, (tuple, list)):
                        shape_str = "[" + ", ".join(f"{t.shape}" for t in tensor) + "]"
                    else:
                        shape_str = 'x'.join(str(dim) for dim in tensor.shape)
                    
                    log_str += f"{name}({shape_str}): {size_mb:.2f}MB, "
        
        print(log_str, flush=True)
    
    return memory_usage

class SWINUNETRTrainer(object):
    def __init__(self, model=None, optimizer=None, weights=None, device='0', continue_tr = False, fold = '', dataset_dir = '', batch_size = 2, roi = True, LR = False, verbose=True):
        # Add verbose parameter to control logging verbosity
        self.verbose = verbose
        self.initial_lr = 0.01
        self.min_lr = 1e-6
        self.warmup_epochs = 5
        self.T_0 = 20
        self.T_mult = 2
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
        self.divis_y = [5, 6, 7]
        self.divis_z = [5, 6, 7, 8, 9]
        self.max_dims = [3, 7, 10]
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
            

        # Initialize different loss functions for different supervision levels
        self.deep_supervision_losses = [
            DiceFocalLoss(include_background=False, sigmoid=False, gamma=5, squared_pred=True, batch=True),    # Main output (doesn't have include_background)
            DiceFocalLoss(include_background=False, sigmoid=False, gamma=5, squared_pred=True, batch=True), # High resolution
            TverskyLoss(include_background=False, sigmoid=False, alpha=0.02, beta=.98, batch=True), # Medium resolution 
            TverskyLoss(include_background=False, sigmoid=False, alpha=0.02, beta=.98), # Lower resolution
            TverskyLoss(include_background=False, sigmoid=False, alpha=0.02, beta=.98), # Low resolution
            FocalLoss(include_background=False, weight=[0.4, 1], gamma=2)    # Lowest resolution
        ]

        # Initialize weights that will be dynamically adjusted
        if self.enable_deep_supervision:
            self.weights = np.array([1/(2.5**i) for i in range(6)])
            self.weights[-1] = 0  # Zero out the last weight as before
            self.weights = self.weights/self.weights.sum()

        self.criterion = DiceLoss(include_background=False, sigmoid=True)

        self.lr = self.optimizer.param_groups[0]['lr']

        self.lr_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=self.min_lr
        )
            
    def load_most_recent_checkpoint(self, fold_dir):
        #check for pt file 

        #if pt file exists, load the model and optimizer state

        if os.path.exists(os.path.join(fold_dir, 'checkpoint.pt')):
            checkpoint = torch.load(os.path.join(fold_dir, 'checkpoint.pt'), weights_only=False)
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
        val_files = os.listdir(self.dataset_dir.replace('train', 'validate'))
        # add val files to files since we are doing kfold cross validation on the train set
        files.extend(val_files)

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
        # add val files to train files
        train_files.extend(val_files)


        test_files = os.listdir(self.dataset_dir.replace('train', 'test'))
        test_files = [self.dataset_dir.replace('train', 'test') + file for file in test_files]

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

            return torch.from_numpy(new_inputs).float().cuda(self.device), torch.from_numpy(new_labels).float().cuda(self.device)
        
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

    def async_log(self, loss, calc_loss, split='train'):
        # Extract scalar values to avoid tensor references
        loss_value = loss.detach().cpu().mean().item()
        calc_loss_value = calc_loss.detach().cpu().mean().item()
        
        # Create a dictionary of metrics to log
        if split == 'val':
            metrics = {'Test loss': loss_value, 'Test dice loss': calc_loss_value}
        else:
            metrics = {'Train loss': loss_value, 'Train dice loss': calc_loss_value}
        
        # Use a thread to log the metrics
        def log_thread_fn(metrics_dict):
            try:
                wandb.log(metrics_dict)
            except Exception as e:
                print(f"Error in wandb logging: {e}")
            finally:
                # Ensure the dictionary is deleted
                del metrics_dict
        
        log_thread = Thread(target=log_thread_fn, args=(metrics.copy(),))
        log_thread.daemon = True  # Make thread daemon so it doesn't prevent program exit
        log_thread.start()
        
        # Delete our copies to ensure we don't keep references
        del loss_value, calc_loss_value, metrics

    def get_lr(self, epoch):
        # Implement warmup
        if epoch < self.warmup_epochs:
            return self.initial_lr * ((epoch + 1) / self.warmup_epochs)
        return self.lr_scheduler.get_last_lr()[0]

    def adjust_learning_rate(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.get_lr(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.lr_scheduler.step()

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
        calc_criterion = DiceLoss(include_background=False, sigmoid=False, to_onehot_y=False)

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
                try:
                    self.optimizer.zero_grad()

                    data = next(self.train_loader)

                    if 'S1' in self.dataset_dir and (i + 1) % 250 == 0 and i < 300:
                        # creat copy of train loader
                        self.train_loader.restart()
                        self.train_loader.generator.shuffle_indices()
                    
                    # Record file information for debugging
                    file_info = {}
                    if 'file_path' in data:
                        file_info['file_path'] = data['file_path']
                    if 'file_name' in data:
                        file_info['file_name'] = data['file_name']
                    
                    inputs = data['data']
                    #roi = data['seg'][:, 1:]
                    labels = data['seg'] # this was changed to support onehot
                    
                    # Log memory usage and tensor sizes before processing
                    log_memory_usage(f"Iteration {i} - Before processing", {
                        "inputs": inputs,
                        "labels": labels
                    }, device=self.device, verbose=self.verbose)
                    
                    # Log file and shape information
                    if self.verbose:
                        shape_info = f"Input shape: {inputs.shape}, Labels shape: {labels.shape}"
                        file_info_str = ""
                        if 'file_info' in locals() and file_info:
                            file_info_str = ", ".join([f"{k}: {v}" for k, v in file_info.items()])
                        
                        print(f"Iteration {i} - {shape_info} - {file_info_str}", flush=True)
                    
                    # Track largest tensors seen so far
                    if not hasattr(self, 'largest_input_shape'):
                        self.largest_input_shape = inputs.shape
                        self.largest_input_file = file_info if 'file_info' in locals() else None
                    elif np.prod(inputs.shape) > np.prod(self.largest_input_shape):
                        self.largest_input_shape = inputs.shape
                        self.largest_input_file = file_info if 'file_info' in locals() else None
                        if self.verbose:
                            print(f"New largest input: {self.largest_input_shape} from {self.largest_input_file}", flush=True)
                    
                    for data in inputs:
                        if data.shape[1] < 32:
                            continue
                        if data.shape[2] < 32:
                            continue
                        if data.shape[3] < 32:
                            continue


                    inputs, labels = inputs.cuda(self.device), labels.cuda(self.device)
                    
                    # Log memory after moving to GPU
                    log_memory_usage(f"Iteration {i} - After moving to GPU", device=self.device, verbose=self.verbose)

                    if (i+1) % 256 == 0:
                        self.adjust_learning_rate(epoch)
                        if self.verbose:
                            print(f'Learning rate: {self.optimizer.param_groups[0]["lr"]}', flush=True)

                    if labels[0].max() == 0:
                        if self.verbose:
                            print("skipping...")
                        continue

                    outputs = self.model(inputs)
                    
                    # Log memory after forward pass
                    log_memory_usage(f"Iteration {i} - After forward pass", {
                        "outputs": outputs
                    }, device=self.device, verbose=self.verbose)

                    temp_loss = 0
                    temp_calc_loss = 0

                    if self.enable_deep_supervision:
                        logits = outputs
                        total_valid_layers = 0
                        layer_weights = []
                        
                        # Log memory at start of deep supervision
                        log_memory_usage(f"Iteration {i} - Start of deep supervision", {
                            "logits": logits
                        }, device=self.device, verbose=self.verbose)
                        
                        # First pass: check which layers have valid labels
                        for j, logit in enumerate(logits):
                            if j != 0:
                                down_labels = F.interpolate(labels, size=logit.shape[2:], mode='nearest')
                            else:
                                down_labels = labels
                            
                            # Layer is valid if it has any valid labels
                            if down_labels.sum() > 0:
                                total_valid_layers += 1
                                layer_weights.append(self.weights[j])
                                # Keep track of the most recent valid layer
                                last_valid_layer = j
                            else:
                                layer_weights.append(0.0)
                            
                            # Clean up intermediate tensors
                            if j != 0:
                                del down_labels
                        
                        # Normalize weights for valid layers
                        if total_valid_layers > 0:
                            layer_weights = np.array(layer_weights)
                            # Simply normalize the weights - keep all valid layers
                            layer_weights = layer_weights / (layer_weights.sum() + 1e-8)
                        
                            # Second pass: calculate losses only for valid layers
                            for j, logit in enumerate(logits):
                                if layer_weights[j] == 0:
                                    continue
                                    
                                if j != 0:
                                    down_labels = F.interpolate(labels, size=logit.shape[2:], mode='nearest')
                                else:
                                    down_labels = labels
                                
                                # Since we're not including background in loss calculation,
                                # we don't need to create complement labels
                                
                                # Use the appropriate loss function for this level
                                if j != 0 and j != 1:
                                    temp_logit = F.sigmoid(logit)
                                    # how do i make this equal to 1 instead of boolean
                                    labels_1 = down_labels == 1 
                                    labels_2 = down_labels == 2
                                    down_labels_tensor = labels_1 + labels_2
                                    down_labels_tensor = down_labels_tensor.long()
                                    temp_complement = 1 - temp_logit
                                    temp_logit_cat = torch.cat((temp_complement, temp_logit), 1)
                                    down_labels_complement = 1 - down_labels_tensor
                                    down_labels_cat = torch.cat((down_labels_complement, down_labels_tensor), 1)
                                    
                                    current_loss = self.deep_supervision_losses[j](temp_logit_cat, down_labels_cat)
                                    temp_loss += layer_weights[j] * current_loss

                                    # Store loss value as a Python scalar, not a tensor
                                    loss_value = current_loss.item()
                                    # Log individual layer losses outside the loop to prevent memory leaks
                                    wandb.log({
                                        f'layer_{j}_loss': loss_value,
                                        f'layer_{j}_weight': layer_weights[j]
                                    })
                                    
                                    # Clean up intermediate tensors
                                    del temp_logit, temp_complement, temp_logit_cat
                                    del labels_1, labels_2, down_labels_tensor
                                    del down_labels_complement, down_labels_cat
                                    del current_loss, loss_value  # Also delete the loss tensor and value
                                else:
                                    # three classes (including background)
                                    temp_logit = F.softmax(logit, dim=1)
                                    # when down labels are 1
                                    labels_1 = down_labels == 1
                                    labels_2 = down_labels == 2
                                    labels_1_tensor = labels_1.long()
                                    labels_2_tensor = labels_2.long()
                                    
                                    bg_channel = 1 - temp_logit[:, 1:2] - temp_logit[:, 2:3]
                                    temp_logit_cat = torch.cat((bg_channel, temp_logit[:, 1:2], temp_logit[:, 2:3]), 1)
                                    
                                    bg_mask = 1 - labels_1_tensor - labels_2_tensor
                                    down_labels_cat = torch.cat((bg_mask, labels_1_tensor, labels_2_tensor), 1)
                                    
                                    current_loss = self.deep_supervision_losses[j](temp_logit_cat, down_labels_cat)
                                    temp_loss += layer_weights[j] * current_loss
                                    
                                    # Store loss value as a Python scalar, not a tensor
                                    loss_value = current_loss.item()
                                    # Log individual layer losses outside the loop to prevent memory leaks
                                    wandb.log({
                                        f'layer_{j}_loss': loss_value,
                                        f'layer_{j}_weight': layer_weights[j]
                                    })
                                    
                                    # Clean up intermediate tensors
                                    del temp_logit, temp_logit_cat
                                    del labels_1, labels_2, labels_1_tensor, labels_2_tensor
                                    del bg_channel, bg_mask, down_labels_cat
                                    del current_loss, loss_value  # Also delete the loss tensor and value
                                
                                # Clean up per-iteration tensors
                                if j != 0:
                                    del down_labels
                                
                        else:
                            # If no valid layers, use only the main output
                            temp_logit = F.sigmoid(logits[0])
                            temp_complement = 1 - temp_logit
                            temp_logit_cat = torch.cat((temp_complement, temp_logit), 1)
                            labels_complement = 1 - labels
                            labels_cat = torch.cat((labels_complement, labels), 1)
                            temp_loss = self.deep_supervision_losses[0](temp_logit_cat, labels_cat)
                            
                            # Clean up intermediate tensors
                            del temp_logit, temp_complement, temp_logit_cat
                            del labels_complement, labels_cat

                        # Calculate final output loss for monitoring
                        out = F.softmax(outputs[0], dim=1)
                        labels_1 = labels == 1
                        labels_2 = labels == 2
                        labels_1_tensor = labels_1.long()
                        labels_2_tensor = labels_2.long()
                        bg_mask = 1 - labels_1_tensor - labels_2_tensor
                        labels_cat = torch.cat((bg_mask, labels_1_tensor, labels_2_tensor), 1)

                        temp_calc_loss = calc_criterion(out, labels_cat)
                        
                        # Clean up intermediate tensors
                        del out, labels_1, labels_2, labels_1_tensor, labels_2_tensor, bg_mask, labels_cat
                        del logits
                        
                        # Keep the first output for further processing
                        outputs = outputs[0]
                    else:
                        temp_output = F.sigmoid(outputs)
                        temp_loss = criterion(temp_output, labels)
                        temp_calc_loss = calc_criterion(temp_output, labels)
                        
                        # Clean up intermediate tensors
                        del temp_output
                    
                    # Log memory before backward pass
                    log_memory_usage(f"Iteration {i} - Before backward pass", {
                        "temp_loss": temp_loss
                    }, device=self.device, verbose=self.verbose)
                    
                    temp_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    # Log memory after backward pass
                    log_memory_usage(f"Iteration {i} - After backward pass", device=self.device, verbose=self.verbose)
                    
                    # Extract scalar values to avoid tensor references
                    loss_value = temp_loss.detach().mean().item()
                    dice_value = temp_calc_loss.mean().item()
                    
                    loss_total += loss_value
                    dice_total += dice_value
                    num_iter += 1
                    
                    if self.enable_deep_supervision:
                        self.async_log(temp_loss, temp_calc_loss)
                    else:
                        # Log scalar values directly
                        wandb.log({'train loss': loss_value, 'train dice loss': loss_value})
                    
                    # Clean up remaining tensors
                    del temp_loss, temp_calc_loss
                    del inputs, labels, outputs
                    del loss_value, dice_value  # Also delete scalar values
                    
                    # Clear optimizer state buffers that might be accumulating
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad = None
                    
                    # Force clearing the autograd graph
                    torch.cuda.empty_cache()
                    
                    # Force a garbage collection periodically
                    if i % 50 == 0:
                        # Log memory before garbage collection
                        before_gc = log_memory_usage(f"Iteration {i} - Before garbage collection", device=self.device, verbose=self.verbose)
                        
                        import gc
                        # Collect everything, including cyclic references
                        gc.collect(2)
                        torch.cuda.empty_cache()
                        
                        # Log memory after garbage collection
                        after_gc = log_memory_usage(f"Iteration {i} - After garbage collection", device=self.device, verbose=self.verbose)
                        
                        # Calculate memory freed for the specific device
                        if isinstance(before_gc, dict) and isinstance(after_gc, dict):
                            # Multiple GPUs
                            device_idx = int(self.device) if isinstance(self.device, str) else self.device
                            device_key = f"GPU {device_idx}"
                            memory_freed = before_gc.get(device_key, 0) - after_gc.get(device_key, 0)
                        else:
                            # Single GPU
                            memory_freed = before_gc - after_gc
                            
                        if self.verbose:
                            print(f"Memory freed by GC: {memory_freed:.2f} MB", flush=True)
                    
                    # Additional cleanup for wandb
                    if hasattr(wandb, 'run') and wandb.run is not None:
                        # Force wandb to sync and clear any pending logs
                        try:
                            wandb.log({})  # Empty log to force sync
                        except:
                            pass
                    
                    # Clear any remaining CUDA caches
                    torch.cuda.empty_cache()
                    
                    # Force Python to release memory
                    if i % 100 == 0:
                        import gc
                        # Run multiple collection passes to ensure cyclic references are cleared
                        for _ in range(3):
                            gc.collect()

                except torch.cuda.OutOfMemoryError as e:
                    # Log detailed information about the tensors that might be causing the OOM
                    print(f"\n\n*** CUDA OUT OF MEMORY ERROR in iteration {i} ***")
                    print(f"Error details: {str(e)}")
                    
                    # Try to log the sizes of key tensors if they exist
                    try:
                        tensor_info = {}
                        if 'inputs' in locals():
                            tensor_info['inputs'] = inputs
                        if 'labels' in locals():
                            tensor_info['labels'] = labels
                        if 'outputs' in locals():
                            if isinstance(outputs, (list, tuple)):
                                for j, out in enumerate(outputs):
                                    tensor_info[f'outputs[{j}]'] = out
                            else:
                                tensor_info['outputs'] = outputs
                        if 'logits' in locals():
                            if isinstance(logits, (list, tuple)):
                                for j, logit in enumerate(logits):
                                    tensor_info[f'logits[{j}]'] = logit
                            else:
                                tensor_info['logits'] = logits
                                
                        log_memory_usage("OOM ERROR - Tensor sizes", tensor_info, device=self.device, verbose=self.verbose)
                    except Exception as inner_e:
                        print(f"Error while logging tensor info: {inner_e}")
                    
                    # Try to free memory
                    try:
                        for name in ['inputs', 'labels', 'outputs', 'logits', 'temp_logit', 'temp_output']:
                            if name in locals():
                                print(f"Deleting {name}")
                                del locals()[name]
                        
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        print("Cleared cache after OOM")
                    except Exception as inner_e:
                        print(f"Error while clearing memory: {inner_e}")
                    
                    # Save information about the batch that caused the OOM
                    try:
                        with open(os.path.join(self.fold_dir, 'oom_info.txt'), 'w') as f:
                            f.write(f"OOM occurred at epoch {epoch}, iteration {i}\n")
                            f.write(f"Error details: {str(e)}\n")
                            
                            # Get memory for all GPUs
                            memory_stats = {}
                            for gpu_idx in range(torch.cuda.device_count()):
                                try:
                                    memory_stats[f"GPU {gpu_idx}"] = torch.cuda.memory_allocated(gpu_idx) / 1024 / 1024
                                except:
                                    memory_stats[f"GPU {gpu_idx}"] = "Error getting memory"
                            
                            f.write(f"GPU memory at crash: {memory_stats}\n")
                            f.write(f"Training device: {self.device}\n\n")
                            
                            # Save file information
                            f.write("File Information:\n")
                            if 'file_info' in locals() and file_info:
                                for key, value in file_info.items():
                                    f.write(f"{key}: {value}\n")
                            else:
                                f.write("No file information available\n")
                            
                            # Save detailed tensor shapes
                            f.write("\nTensor Shapes:\n")
                            if 'inputs' in locals():
                                f.write(f"inputs: {inputs.shape}, dtype={inputs.dtype}, device={inputs.device}\n")
                                if hasattr(inputs, 'min') and hasattr(inputs, 'max'):
                                    f.write(f"inputs value range: min={inputs.min().item()}, max={inputs.max().item()}\n")
                            
                            if 'labels' in locals():
                                f.write(f"labels: {labels.shape}, dtype={labels.dtype}, device={labels.device}\n")
                                if hasattr(labels, 'min') and hasattr(labels, 'max'):
                                    f.write(f"labels value range: min={labels.min().item()}, max={labels.max().item()}\n")
                                    f.write(f"labels unique values: {torch.unique(labels).tolist()}\n")
                            
                            if 'outputs' in locals():
                                if isinstance(outputs, (list, tuple)):
                                    for j, out in enumerate(outputs):
                                        f.write(f"outputs[{j}]: {out.shape}, dtype={out.dtype}, device={out.device}\n")
                                else:
                                    f.write(f"outputs: {outputs.shape}, dtype={outputs.dtype}, device={outputs.device}\n")
                            
                            if 'logits' in locals():
                                if isinstance(logits, (list, tuple)):
                                    for j, logit in enumerate(logits):
                                        f.write(f"logits[{j}]: {logit.shape}, dtype={logit.dtype}, device={logit.device}\n")
                                else:
                                    f.write(f"logits: {logits.shape}, dtype={logits.dtype}, device={logits.device}\n")
                    except Exception as inner_e:
                        print(f"Error while saving OOM info: {inner_e}")
                    
                    # Re-raise the error
                    raise
                
                except Exception as e:
                    print(f"Error in iteration {i}: {e}")
                    raise

            if self.verbose:
                print(f"Train {epoch} took {time.time() - t1} seconds", flush=True)

            loss_total = loss_total
            dice_total = dice_total
                    
            print(f'Epoch: {epoch}, Loss: {loss_total/(num_iter)}, Dice Loss: {dice_total/(num_iter)}', flush=True)
            
            # Save information about the largest tensors seen in this epoch
            if hasattr(self, 'largest_input_shape'):
                with open(os.path.join(self.fold_dir, f'largest_tensors_epoch_{epoch}.txt'), 'w') as f:
                    f.write(f"Epoch {epoch} largest tensor information:\n")
                    f.write(f"Largest input shape: {self.largest_input_shape}\n")
                    f.write(f"Total elements: {np.prod(self.largest_input_shape)}\n")
                    f.write(f"Estimated memory (float32): {np.prod(self.largest_input_shape) * 4 / (1024*1024):.2f} MB\n")
                    if self.largest_input_file:
                        f.write(f"From file: {self.largest_input_file}\n")
                    if self.verbose:
                        print(f"Saved largest tensor information for epoch {epoch}", flush=True)
                
                # Reset for next epoch
                del self.largest_input_shape
                del self.largest_input_file
            
            # Perform aggressive memory cleanup at the end of each epoch
            if self.verbose:
                print(f"Performing aggressive memory cleanup at end of epoch {epoch}", flush=True)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = None
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Full garbage collection
            import gc
            gc.collect(2)
            
            # Log memory after cleanup
            log_memory_usage(f"End of epoch {epoch} - After cleanup", device=self.device, verbose=self.verbose)
            
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

                    inputs, labels = inputs.cuda(self.device), labels.cuda(self.device)

                    outputs = self.model(inputs)

                    if self.enable_deep_supervision:
                        outputs = outputs[0]
                    
                    temp_logit = F.softmax(outputs, dim=1)
                    label_1 = labels == 1
                    label_2 = labels == 2
                    labels_1_tensor = label_1.long()
                    labels_2_tensor = label_2.long()

                    bg_channel = 1 - temp_logit[:, 1:2] - temp_logit[:, 2:3]
                    temp_logit_cat = torch.cat((bg_channel, temp_logit[:, 1:2], temp_logit[:, 2:3]), 1)
                    
                    bg_mask = 1 - labels_1_tensor - labels_2_tensor
                    down_labels_cat = torch.cat((bg_mask, labels_1_tensor, labels_2_tensor), 1)

                    outputs = temp_logit_cat
                    labels = down_labels_cat

                    temp_loss_calc = calc_criterion(outputs, labels)

                    learned_loss = val_criterion(outputs[:, :, ...], labels[:, :, ...])

                    loss_total += learned_loss.item()
                    loss_dice += temp_loss_calc.item()
                    num_iter += 1


                    self.async_log(learned_loss, temp_loss_calc, split='val')

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
    parser.add_argument('--quiet', action='store_true', required=False,
                        help="Reduce verbosity of logging output")
    args = parser.parse_args()


    trainer = SWINUNETRTrainer(weights=args.pretrained_weights, fold=args.fold, dataset_dir=args.dataset_dir, 
                              device=args.device, continue_tr=args.c, LR=args.LR, roi=args.roi, 
                              verbose=not args.quiet)
    trainer.run_training()

if __name__ == '__main__':
    run_training_entry()