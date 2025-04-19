import argparse
import os
import time
import shutil
from pathlib import Path
import warnings
import pickle
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as PyTorchDataLoader # Alias PyTorch DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm

import monai
from monai.data import Dataset, list_data_collate
from monai.transforms.transform import MapTransform
from monai.transforms import (
    Compose, EnsureChannelFirstd, Orientationd, Spacingd,
    Resized, RandRotate90d, RandGaussianNoised, EnsureTyped,
    ScaleIntensityRanged,
)
from monai.networks.nets import DenseNet121
from monai.utils import set_determinism
# import monai focal loss
from monai.losses import FocalLoss

from Augmentations import get_augmentations

from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

# --- WandB Import ---
try:
    import wandb
except ImportError:
    print("wandb not installed, pip install wandb to enable logging.")
    wandb = None
# --- End WandB Import ---

warnings.filterwarnings("ignore", category=UserWarning, module="monai")
warnings.filterwarnings("ignore", message=".*weights_only=False.*") # Suppress torch.load warning

# --- Updated get_image_label with threshold ---
def get_image_label(segmentation_mask_data: np.ndarray, min_pixels_threshold: int = 100) -> int:
    """
    Derives a single image-level classification label from a segmentation mask,
    requiring a minimum number of artifact pixels.

    Args:
        segmentation_mask_data: NumPy array of the *transformed* segmentation mask.
                                 Assumes labels 1 and 2 represent artifact types.
        min_pixels_threshold: Minimum number of voxels required for class 1 or 2.

    Returns:
        0 = No Artifact, 1 = Artifact Type 1, 2 = Artifact Type 2
    """
    unique_labels = np.unique(segmentation_mask_data)

    # Check for class 1 first (priority)
    if 1 in unique_labels:
        count1 = np.count_nonzero(segmentation_mask_data == 1)
        if count1 >= min_pixels_threshold:
            return 1
    # Check for class 2 if class 1 wasn't dominant enough
    elif 2 in unique_labels:
        count2 = np.count_nonzero(segmentation_mask_data == 2)
        if count2 >= min_pixels_threshold:
            return 2

    return 0


# --- Custom Transform for NPZ Loading ---
class LoadPairedArr0d(MapTransform):
    """
    Custom dictionary transform to load image and segmentation data from
    separate NPZ files specified by 'image_path' and 'seg_path' keys.
    It assumes the relevant data in both files is stored under the key 'arr_0'.
    Outputs the loaded arrays under the keys 'image' and 'seg'.
    """
    def __init__(self, keys=("image_path", "seg_path"), allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        img_path = d.get("image_path")
        seg_path = d.get("seg_path")

        if img_path is None or seg_path is None:
            raise KeyError("Input dictionary must contain 'image_path' and 'seg_path' keys.")

        try:
            img_npz = np.load(img_path)
            if 'arr_0' not in img_npz:
                raise KeyError(f"Key 'arr_0' not found in image npz file: {img_path}. Available keys: {list(img_npz.keys())}")
            d["image"] = img_npz['arr_0']
            img_npz.close()

            seg_npz = np.load(seg_path)
            if 'arr_0' not in seg_npz:
                raise KeyError(f"Key 'arr_0' not found in seg npz file: {seg_path}. Available keys: {list(seg_npz.keys())}")
            d["seg"] = seg_npz['arr_0']
            seg_npz.close()

            # Remove original path keys after successful loading
            del d["image_path"]
            del d["seg_path"]

        except Exception as e:
            print(f"Error loading paired NPZ files ({img_path}, {seg_path}): {e}")
            raise e

        return d
# --- End Custom Transform ---


# --- New DataLoader based on Dataloaders.py structure ---
class ArtifactClassificationDataLoader(DataLoader):
    """
    batchgenerators compatible DataLoader for artifact classification.
    Loads paired image/segmentation NPZ files using MONAI transforms for initial preprocessing,
    derives classification labels, and prepares batches for MultiThreadedAugmenter.
    Modeled after CustomDataLoader in Dataloaders.py.
    """
    def __init__(self, data_dicts, batch_size, monai_transforms, min_pixels_threshold=100, num_threads_in_multithreaded=1):
        """
        Args:
            data_dicts: List of dictionaries, each containing 'image_path' and 'seg_path'.
            batch_size: The batch size for generating batches.
            monai_transforms: MONAI Compose object for initial loading and preprocessing.
                              Should output 'image' and 'seg' keys (Tensors or NumPy).
            min_pixels_threshold: Minimum artifact pixels to assign label 1 or 2.
            num_threads_in_multithreaded: Passed to DataLoader for internal use.
        """
        super().__init__(data_dicts, batch_size, num_threads_in_multithreaded)
        self.monai_transforms = monai_transforms
        self.min_pixels_threshold = min_pixels_threshold
        self.indices = list(range(len(data_dicts)))

    def __len__(self):
        return len(self._data)

    # get_indices is inherited from DataLoader

    def generate_train_batch(self):
        """
        Generates a batch of data suitable for training.
        Loads data, applies MONAI transforms, derives labels, returns NumPy arrays.
        """
        indices = self.get_indices()

        batch_images = []
        batch_segs = [] # Keep segs for potential spatial transforms in batchgenerators
        batch_labels = []
        batch_filenames = []

        for idx in indices:
            data_dict_i = self._data[idx]

            try:
                filename = Path(data_dict_i['image_path']).name
                transformed_data = self.monai_transforms(data_dict_i)

                if "image" not in transformed_data or "seg" not in transformed_data:
                    raise KeyError(f"MONAI transforms did not produce 'image' and 'seg' keys for {filename}. Keys found: {transformed_data.keys()}")

                image_tensor = transformed_data['image']
                seg_tensor = transformed_data['seg']

                image_np = image_tensor.cpu().numpy()
                seg_np = seg_tensor.cpu().numpy()

                if seg_np.shape[0] == 1:
                    seg_for_label = seg_np[0]
                else:
                    tqdm.write(f"Warning: Seg mask for {filename} has shape {seg_np.shape}. Using first channel for label.")
                    seg_for_label = seg_np[0]

                image_label = get_image_label(seg_for_label, self.min_pixels_threshold)

                batch_images.append(image_np)
                batch_segs.append(seg_np)
                batch_labels.append(image_label)
                batch_filenames.append(filename)

            except FileNotFoundError as e:
                 tqdm.write(f"File not found error processing sample index {idx} ({data_dict_i.get('image_path', 'N/A')}): {e}. Skipping.")
                 continue
            except Exception as e:
                tqdm.write(f"Error processing sample index {idx} ({data_dict_i.get('image_path', 'N/A')}): {e}")
                import traceback
                traceback.print_exc() # Keep traceback for now, might still interfere
                continue # Skip this sample

        if not batch_images:
            tqdm.write("Warning: generate_train_batch generated an empty batch.")
            # Determine expected shape (e.g., from args.input_size) - need to know channel count (1?)
            c, d, h, w = 1, args.input_size[0], args.input_size[1], args.input_size[2]
            return {
                'data': np.empty((0, c, d, h, w), dtype=np.float32),
                'seg': np.empty((0, c, d, h, w), dtype=np.float32),
                'label': np.empty((0,), dtype=np.int64),
                'filenames': []
             }

        try:
            image_batch_np = np.stack(batch_images, axis=0)
            seg_batch_np = np.stack(batch_segs, axis=0)
            label_batch_np = np.array(batch_labels, dtype=np.int64)
        except Exception as stack_e:
            tqdm.write(f"Error stacking batch data: {stack_e}")
            tqdm.write(f"Individual image shapes: {[img.shape for img in batch_images]}")
            tqdm.write(f"Individual seg shapes: {[seg.shape for seg in batch_segs]}")
            # Return empty batch on stacking error
            c, d, h, w = 1, args.input_size[0], args.input_size[1], args.input_size[2]
            return {
                'data': np.empty((0, c, d, h, w), dtype=np.float32),
                'seg': np.empty((0, c, d, h, w), dtype=np.float32),
                'label': np.empty((0,), dtype=np.int64),
                'filenames': []
             }

        return {
            'data': image_batch_np,
            'seg': seg_batch_np, # Include seg for spatial transforms
            'label': label_batch_np,
            # Use ones_like for ROI placeholder, indicating entire image is ROI
            'roi': np.ones_like(seg_batch_np, dtype=seg_batch_np.dtype),
            'filenames': batch_filenames
        }


# --- Original MONAI Dataset (can still be used for validation) ---
class ArtifactDataset(monai.data.Dataset): # Inherit from monai.data.Dataset
    """
    Custom Dataset to load .npy/.npz files (containing image and seg)
    and derive classification labels from segmentations.
    """
    def __init__(self, data_dicts, transforms):
        super().__init__(data=data_dicts, transform=transforms)

    def _transform(self, index: int):
        """
        Loads paired image/seg, applies transforms, derives label.
        """
        data_i = self.data[index]
        filename = Path(data_i['image_path']).name
        transformed_data = self.transform(data_i)

        seg_data_tensor = transformed_data['seg']
        seg_data_np = seg_data_tensor.cpu().numpy()

        global cli_args
        min_pixels = cli_args.min_artifact_pixels if 'cli_args' in globals() else 100 # Default fallback
        image_label = get_image_label(seg_data_np, min_pixels_threshold=min_pixels)

        image_tensor = transformed_data["image"]
        return {"image": image_tensor, "label": image_label, "filename": filename}


def main(args):
    set_determinism(seed=args.seed)
    output_dir = Path(args.output_dir)
    run_name = f"run_ep{args.epochs}_bs{args.batch_size}_lr{args.initial_lr}_optim{args.optimizer}"
    run_output_dir = output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_output_dir / f"classify_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"

    # --- Initialize WandB ---
    if wandb and not args.no_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                dir=str(output_dir),
                resume="allow",
                # For simplicity now, generate new ID unless specific resume path given
                id=wandb.util.generate_id() if args.resume is None else None
            )
            print(f"WandB logging enabled. Project: {args.wandb_project}, Run: {run_name}")
            wandb_enabled = True
        except Exception as e:
            print(f"Error initializing WandB: {e}. Disabling WandB logging.")
            wandb_enabled = False
    else:
        print("WandB logging disabled.")
        wandb_enabled = False
    # --- End WandB Init ---

    # --- Data Setup ---
    dataset_path = Path(args.dataset_dir)
    train_path = dataset_path / 'train'
    val_path = dataset_path / 'validate'
    test_path = dataset_path / 'test'

    if not train_path.is_dir() or not val_path.is_dir() or not test_path.is_dir():
        raise NotADirectoryError(
            f"Expected 'train', 'validate', and 'test' subdirectories in {args.dataset_dir}"
        )

    # --- Find and Pair Files ---
    def find_and_pair_files(data_dir, file_pattern):
        paired_files = []
        missing_labels = 0
        print(f"Searching for image files ('{file_pattern}' containing '_image_') in {data_dir}...")
        potential_image_files = sorted([str(f) for f in data_dir.glob(file_pattern) if '_image_' in f.name])
        print(f"Found {len(potential_image_files)} potential image files. Pairing with labels...")

        for img_path_str in potential_image_files:
            img_path = Path(img_path_str)
            label_filename = img_path.name.replace('_image_', '_maskArtifact_', 1)
            label_path = img_path.with_name(label_filename)

            if label_path.is_file():
                paired_files.append({"image_path": img_path_str, "seg_path": str(label_path)})
            else:
                tqdm.write(f"Warning: Derived label file {label_path.name} not found for image {img_path.name} in {data_dir}. Skipping.")
                missing_labels += 1
        print(f"Successfully paired {len(paired_files)} files in {data_dir}. {missing_labels} missing labels.")
        return paired_files, missing_labels

    train_files_list, missing_train = find_and_pair_files(train_path, args.file_pattern)
    validate_files_list, missing_val = find_and_pair_files(val_path, args.file_pattern)
    test_files_list, missing_test = find_and_pair_files(test_path, args.file_pattern)

    train_files = train_files_list + validate_files_list
    test_files = test_files_list

    total_missing = missing_train + missing_val + missing_test
    if total_missing > 0:
        print(f"Warning: Total {total_missing} label files were missing across all directories.")

    if not train_files:
        raise ValueError("No training image/label pairs were successfully found in 'train' or 'validate' directories.")
    if not test_files:
        raise ValueError("No validation/testing image/label pairs were successfully found in 'test' directory.")

    print(f"Using {len(train_files)} samples from 'train'+'validate' for Training.")
    print(f"Using {len(test_files)} samples from 'test' for Testing during training loop.")

    img_key = "image"
    seg_key = "seg"

    base_transforms = [
        LoadPairedArr0d(keys=("image_path", "seg_path")),
        EnsureChannelFirstd(keys=[img_key, seg_key], channel_dim="no_channel"),
        Orientationd(keys=[img_key, seg_key], axcodes="RAS"),
        Spacingd(keys=[img_key, seg_key], pixdim=args.target_spacing, mode=("bilinear", "nearest")),
        Resized(keys=[img_key, seg_key], spatial_size=args.input_size, mode=("bilinear", "nearest")),
    ]

    try:
        augmentations_result = get_augmentations()
        print("Successfully called get_augmentations().")

        if isinstance(augmentations_result, Compose):
            specific_transforms_list = list(augmentations_result.transforms)
            print(f"Extracted {len(specific_transforms_list)} transforms from returned Compose object.")
        elif isinstance(augmentations_result, (list, tuple)):
            specific_transforms_list = list(augmentations_result)
            print(f"Using {len(specific_transforms_list)} transforms from returned list/tuple.")
        elif isinstance(augmentations_result, monai.transforms.Transform):
            specific_transforms_list = [augmentations_result]
            print("Using the single transform returned by get_augmentations.")
        elif augmentations_result is None:
            print("get_augmentations() returned None. Applying only base transforms.")
            specific_transforms_list = []
        else:
            print(f"Warning: get_augmentations() returned unusable type: {type(augmentations_result)}. Applying only base transforms.")
            specific_transforms_list = []

    except ImportError:
        print("Warning: Could not import get_augmentations from Augmentations.py. Proceeding without custom augmentations.")
        specific_transforms_list = []
    except Exception as e:
        print(f"Warning: Error calling get_augmentations(): {e}. Proceeding without custom augmentations.")
        specific_transforms_list = []


    pre_aug_transforms = Compose([
        *base_transforms,
    ])
    test_transforms = Compose([
        *base_transforms,
        EnsureTyped(keys=[img_key, seg_key], dtype=torch.float32),
    ])


    print("Setting up BatchGenerators training data loader...")
    train_dl = ArtifactClassificationDataLoader(
        data_dicts=train_files,
        batch_size=args.batch_size,
        monai_transforms=pre_aug_transforms,
        min_pixels_threshold=args.min_artifact_pixels,
        num_threads_in_multithreaded=args.num_workers
    )

    print("Setting up MONAI test data loader...")
    test_ds = ArtifactDataset(data_dicts=test_files, transforms=test_transforms) if test_files else None

    print("Getting batchgenerators transforms from Augmentations.py...")
    bg_transforms = get_augmentations()
    if bg_transforms is None:
         print("Warning: get_augmentations returned None, training loader will have no batchgenerators transforms.")

    print("Initializing MultiThreadedAugmenter for training...")
    train_loader = MultiThreadedAugmenter(
        train_dl,
        transform=bg_transforms,
        num_processes=args.num_workers,
        num_cached_per_queue=3,
        pin_memory=True,
        seeds=None,
    )
    print("MultiThreadedAugmenter initialized.")


    test_loader = PyTorchDataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=list_data_collate
    ) if test_ds else None
    print("Test DataLoader initialized.")

    batches_per_epoch = math.ceil(len(train_dl) / args.batch_size)
    print(f"Calculated batches per epoch: {batches_per_epoch}")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=args.num_classes
    ).to(device)

    if args.use_weighted_loss:
        # IMPORTANT: These weights are ESTIMATES. Ideal weights depend on the distribution *after* transforms.
        weights = torch.tensor(args.class_weights).float().to(device)
        print(f"Using weighted CrossEntropyLoss. Weights (C0, C1, C2): {weights.cpu().numpy()}")
        criterion = FocalLoss(alpha=weights, gamma=2)
    else:
        print("Using standard CrossEntropyLoss.")
        criterion = FocalLoss(alpha=weights, gamma=2)

    if args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
        print(f"Using AdamW optimizer: LR={args.initial_lr}, WeightDecay={args.weight_decay}")
    elif args.optimizer.lower() == 'sgd':
         optimizer = optim.SGD(model.parameters(), lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
         print(f"Using SGD optimizer: LR={args.initial_lr}, Momentum={args.momentum}, Nesterov={args.nesterov}, WeightDecay={args.weight_decay}")
    else:
         raise ValueError(f"Unsupported optimizer: {args.optimizer}")


    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    resume_path = Path(args.resume) if args.resume else run_output_dir / "checkpoint_latest.pt"

    if resume_path.exists() and resume_path.is_file():
        print(f"Attempting to load checkpoint: {resume_path}")
        ckpt_path = Path(resume_path)
        if ckpt_path.is_file():
            print(f"Resuming training from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            try:
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_metric = checkpoint.get('best_metric', -1)
                best_metric_epoch = checkpoint.get('best_metric_epoch', -1)
                print(f"  Loaded model, optimizer, scheduler. Resuming from Epoch {start_epoch}")
                print(f"  Previous best metric (Accuracy): {best_metric:.4f} at epoch {best_metric_epoch}")
            except Exception as e:
                print(f"Error loading checkpoint state: {e}. Starting from scratch.")
                start_epoch = 0
                best_metric = -1
                best_metric_epoch = -1
        else:
            if args.resume:
                 print(f"Resume path specified ({args.resume}), but file not found. Starting from scratch.")
            else:
                 print(f"No latest checkpoint found in {run_output_dir}. Starting from scratch.")
    else:
        start_epoch = 0
        best_metric = -1 # Best validation accuracy
        best_metric_epoch = -1


    print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        print("-" * 10)
        print(f"Epoch {epoch}/{args.epochs - 1}")

        # --- Training Phase ---
        model.train()
        train_loss = 0
        train_steps = 0
        train_correct = 0
        train_total = 0
        recent_train_losses = [] # For moving average
        recent_train_accuracies = [] # For moving average

        ema_loss = 0.0 # Initialize differently for first update
        ema_acc = None

        alpha = 2.0 / (args.log_freq + 1) if args.log_freq > 0 else 0.5 # Smoothing factor

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Train", unit="batch", leave=False)
        for batch_data in progress_bar:
            try:
                 if not batch_data or 'data' not in batch_data or 'label' not in batch_data:
                     tqdm.write(f"Warning: Skipping empty or invalid batch from train_loader in epoch {epoch}.")
                     continue

                 image_np = batch_data['data']
                 label_np = batch_data['label']

                 inputs = torch.from_numpy(image_np).float().to(device)
                 labels = torch.from_numpy(label_np).long().to(device)

            except Exception as e:
                 tqdm.write(f"Error converting batch from MultiThreadedAugmenter output to tensors: {e}")
                 tqdm.write(f"Batch keys: {batch_data.keys() if isinstance(batch_data, dict) else type(batch_data)}")
                 if isinstance(batch_data, dict):
                      for k in ['data', 'label', 'seg']:
                          if k in batch_data:
                               v = batch_data[k]
                               tqdm.write(f"  {k}: {type(v)}, shape: {getattr(v, 'shape', 'N/A')}")
                 continue # Skip this batch

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            train_loss += loss.item()
            train_steps += 1

            recent_train_losses.append(current_loss)
            if len(recent_train_losses) > args.log_freq:
                recent_train_losses.pop(0)

            _, predicted = torch.max(outputs.data, 1)
            batch_total = labels.size(0)
            batch_correct = (predicted == labels).sum().item()
            batch_accuracy = (batch_correct / batch_total) * 100 if batch_total > 0 else 0

            recent_train_accuracies.append(batch_accuracy)
            if len(recent_train_accuracies) > args.log_freq:
                 recent_train_accuracies.pop(0)

            if train_steps == 1:
                 ema_loss = current_loss
                 ema_acc = batch_accuracy
            else:
                 ema_loss = alpha * current_loss + (1 - alpha) * ema_loss
                 ema_acc = alpha * batch_accuracy + (1 - alpha) * ema_acc

            train_total += batch_total
            train_correct += batch_correct

            if wandb_enabled and train_steps % args.log_freq == 0:
                moving_avg_loss = np.mean(recent_train_losses) if recent_train_losses else current_loss
                wandb.log({
                    "train/step_loss": current_loss,
                    "train/step_loss_moving_avg": moving_avg_loss,
                    "train/step_accuracy": batch_accuracy,
                    "train/step_accuracy_moving_avg": np.mean(recent_train_accuracies) if recent_train_accuracies else batch_accuracy,
                    "epoch": epoch + (train_steps / batches_per_epoch)
                }, step=epoch * batches_per_epoch + train_steps)

            if train_steps > 0:
                 moving_avg_loss = np.mean(recent_train_losses) if recent_train_losses else current_loss
                 moving_avg_acc = np.mean(recent_train_accuracies) if recent_train_accuracies else batch_accuracy
                 progress_bar.set_postfix(
                     loss_ema=f"{ema_loss:.4f}",
                     acc_ema=f"{ema_acc:.2f}%"
                 )

        avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
        train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
        print(f"Epoch {epoch} Average Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch} Learning Rate: {current_lr:.6f}")


        # --- Validation Phase ---
        model.eval()
        avg_test_loss = np.nan # Default if no testing
        test_accuracy = np.nan
        test_metrics = {"epoch": epoch}

        if test_loader:
            test_loss = 0
            test_steps = 0
            all_preds = []
            all_labels = []
            print(f"Running Testing for Epoch {epoch}...")

            with torch.no_grad():
                for batch_data in tqdm(test_loader, desc=f"Epoch {epoch} Test", unit="batch", leave=False):
                    inputs = batch_data["image"].to(device)
                    labels = batch_data["label"].to(device)
                    test_outputs = model(inputs)
                    loss = criterion(test_outputs, labels)

                    test_loss += loss.item()
                    test_steps += 1

                    _, predicted = torch.max(test_outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            avg_test_loss = test_loss / test_steps if test_steps > 0 else 0
            test_accuracy = accuracy_score(all_labels, all_preds) * 100 if all_labels else 0

            print(f"Epoch {epoch} Average Test Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

            test_metrics["test/epoch_loss"] = avg_test_loss
            test_metrics["test/epoch_accuracy"] = test_accuracy

            if all_labels:
                target_names = [f"Class_{i}" for i in range(args.num_classes)]
                try:
                    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
                    print("Test Classification Report:")
                    print(report)

                    cm = confusion_matrix(all_labels, all_preds)
                    print("Test Confusion Matrix:")
                    print(cm)

                    if wandb_enabled:
                        try:
                            report_dict = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0, output_dict=True)
                            for class_name, metrics_dict in report_dict.items():
                                if isinstance(metrics_dict, dict):
                                    for metric_name, value in metrics_dict.items():
                                        test_metrics[f"test_report/{class_name}_{metric_name}"] = value
                                else:
                                    test_metrics[f"test_report/{class_name}"] = metrics_dict

                            test_metrics["test/confusion_matrix"] = wandb.Table(
                                columns=target_names,
                                data=cm.tolist(),
                                rows=target_names
                            )

                        except Exception as report_e:
                            print(f"Warning: Could not format detailed report/cm for WandB: {report_e}")

                except ValueError as e:
                    print(f"Could not generate classification report (likely due to missing classes in batch): {e}")
        else:
             print(f"Epoch {epoch} - No testing performed (test directory empty or issue loading).")

        # --- Checkpointing ---
        epoch_metric = test_accuracy

        current_best_metric = best_metric if not np.isnan(best_metric) else -1
        is_best = False
        if not np.isnan(epoch_metric):
            is_best = epoch_metric > current_best_metric

        latest_checkpoint_path = run_output_dir / "checkpoint_latest.pt"
        checkpoint_data = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metric': best_metric,
            'best_metric_epoch': best_metric_epoch,
            'test_accuracy': epoch_metric
        }
        torch.save(checkpoint_data, latest_checkpoint_path)
        print(f"Saved latest checkpoint to {latest_checkpoint_path.name}")

        if is_best:
            best_metric = epoch_metric
            if wandb_enabled:
                wandb.summary['best_test_accuracy'] = best_metric
                wandb.summary['best_test_epoch'] = epoch

            best_metric_epoch = epoch
            best_checkpoint_path = run_output_dir / "checkpoint_best.pt"
            shutil.copyfile(latest_checkpoint_path, best_checkpoint_path)
            print(f"Saved new best checkpoint (Test Accuracy: {best_metric:.2f}%) to {best_checkpoint_path.name}")
        else:
             print(f"Metric ({epoch_metric:.2f}%) did not improve from best ({best_metric:.2f}% at epoch {best_metric_epoch}).")

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds.")

        with open(log_file, 'a') as f:
             f.write(f"Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}, "
                     f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}, LR: {current_lr:.6f}")
             if all_labels and 'report' in locals() and 'cm' in locals():
                 f.write("Test Report:")
                 f.write(str(report) + "")
                 f.write("Test Confusion Matrix:")
                 f.write(np.array2string(cm) + "")
             f.write("-" * 20 + "")

        if wandb_enabled:
             wandb.log(test_metrics, step=epoch * batches_per_epoch + batches_per_epoch)
             moving_avg_loss_epoch_end = np.mean(recent_train_losses) if recent_train_losses else avg_train_loss
             wandb.log({
                 "train/epoch_loss": avg_train_loss,
                 "train/epoch_accuracy": train_accuracy,
                 "train/epoch_loss_moving_avg": moving_avg_loss_epoch_end,
                 "train/epoch_accuracy_moving_avg": np.mean(recent_train_accuracies) if recent_train_accuracies else train_accuracy,
                 "learning_rate": current_lr,
                 "epoch": epoch
             }, step=epoch * batches_per_epoch + batches_per_epoch)


    print(f"Training finished. Best Test Accuracy: {best_metric:.2f}% at epoch {best_metric_epoch}")
    print(f"Logs saved to: {log_file}")
    print(f"Best checkpoint saved to: {run_output_dir / 'checkpoint_best.pt'}")

    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 3D CNN for Artifact Classification")

    # --- Data Arguments ---
    parser.add_argument('dataset_dir', type=str, help="Directory containing 'train', 'validate', and 'test' subfolders with NPZ files.")
    parser.add_argument('--output_dir', type=str, default='./results_classify', help="Directory to save checkpoints and logs.")
    parser.add_argument('--file_pattern', type=str, default='*.np[yz]', help="Glob pattern for data files (e.g., '*.npy', '*.npz'). Note: Script assumes _image_ in filename identifies image files.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of worker processes for data loading.")

    # --- Model & Training Arguments ---
    parser.add_argument('--num_classes', type=int, required=True, help="Number of output classes (e.g., 3 for Bkg, Art1, Art2).")
    parser.add_argument('--input_size', type=int, nargs=3, default=[64, 160, 256], help='Input size for the network (D, H, W).')
    parser.add_argument('--target_spacing', type=float, nargs=3, default=[1.5, 1.5, 1.5], help='Target voxel spacing (x, y, z).')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size.')

    # --- NPZ Loading Arguments ---
    parser.add_argument('--initial_lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for scheduler.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'], help='Optimizer type.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for AdamW/SGD.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
    parser.add_argument('--nesterov', action=argparse.BooleanOptionalAction, default=True, help='Use Nesterov momentum for SGD.')

    # --- Data Labeling Arguments ---
    parser.add_argument('--min_artifact_pixels', type=int, default=20,
                        help='Minimum number of artifact voxels required in transformed mask to assign label 1 or 2.')

    # --- Loss Function Arguments ---
    parser.add_argument('--use_weighted_loss', action=argparse.BooleanOptionalAction, default=True,
                        help='Use weighted CrossEntropyLoss based on class distribution.')
    # Default weights based on transformed distribution (Fold 0) with capped Class 0 weight
    parser.add_argument('--class_weights', type=float, nargs=3, default=[2, 1.27, 1],
                        help='Weights for Class 0, 1, 2 for weighted loss. Ignored if --no_use_weighted_loss.')

    # --- System Arguments ---
    parser.add_argument('--device', type=int, default=0, help="GPU device ID to use.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training from.')

    # --- Logging Arguments ---
    parser.add_argument('--log_freq', type=int, default=100, help='Log training metrics every N steps.')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging.')
    parser.add_argument('--wandb_project', type=str, default='ArtifactClassification', help='WandB project name.')

    args = parser.parse_args()

    global cli_args
    cli_args = args

    if args.num_classes <= 1:
         raise ValueError("num_classes must be at least 2 for classification.")

    main(args) 