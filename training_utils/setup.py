import sys
import time
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from Dataloaders import CustomDataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from Augmentations import get_augmentations
from batchgenerators.transforms.abstract_transforms import Compose

# --- Helper Functions ---

def setup_output_directory(output_dir: str, fold: str, dataset_name: str, continue_tr: bool):
    """Creates fold-specific output directory and sets up logging.

    Args:
        output_dir: Base directory for results.
        fold: Current fold number (string).
        dataset_name: Name of the dataset.
        continue_tr: Flag indicating if training is being continued.

    Returns:
        Path: Path to the fold-specific output directory.
        TextIOWrapper: Open log file handle.
        bool: Flag indicating if a checkpoint might exist.
    """
    base_path = Path(output_dir)
    dataset_path = base_path / dataset_name
    fold_path = dataset_path / f"fold_{fold}"

    fold_path.mkdir(parents=True, exist_ok=True)

    checkpoint_exists = (fold_path / 'checkpoint.pt').exists()

    if checkpoint_exists and not continue_tr:
        print(f"WARNING: Checkpoint found in {fold_path} but --c flag not used. Potential to overwrite.")
        # Decide if overwriting is allowed or if an error should be raised.
        # For now, we'll allow overwriting but warn.
        # raise ValueError(f"Fold {fold} checkpoint exists at {fold_path / 'checkpoint.pt'}. Use --c to continue.")

    # Setup logging to file
    log_file_name = "log0.txt"
    i = 0
    while (fold_path / log_file_name).exists():
        # If continuing training and a log file exists, append to the latest one
        if continue_tr:
            break 
        i += 1
        log_file_name = f"log{i}.txt"
    
    log_file_path = fold_path / log_file_name
    log_mode = 'a' if continue_tr and (fold_path / log_file_name).exists() else 'w'
    # Ensure log file handle is managed correctly
    try:
        log_file = open(log_file_path, mode=log_mode)
    except IOError as e:
        print(f"Error opening log file {log_file_path}: {e}")
        # Fallback to standard output if log file fails
        log_file = sys.stdout 

    # Don't redirect stdout globally here, manage it in the main script
    # if log_file != sys.stdout:
    #     sys.stdout = log_file # Redirect stdout only if file opened successfully
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Output directory: {fold_path}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Logging to {log_file_path} (mode: {log_mode})")

    return fold_path, log_file, checkpoint_exists

def setup_dataloaders(dataset_dir: str, fold: str, fold_dir: Path, batch_size: int, use_roi: bool, num_workers_train: int, num_workers_val: int):
    """Sets up KFold splits, DataLoaders, and Augmenters.

    Args:
        dataset_dir: Path to the root dataset directory.
        fold: Current fold number (string).
        fold_dir: Path to the fold-specific output directory.
        batch_size: Training batch size.
        use_roi: Whether to use ROI cropping/padding.
        num_workers_train: Number of workers for training augmenter.
        num_workers_val: Number of workers for validation augmenter.

    Returns:
        MultiThreadedAugmenter: Training data loader.
        MultiThreadedAugmenter: Test data loader.
        list: List of file paths used for validation in this fold.
        Compose: Training augmentations.
    """
    dataset_path = Path(dataset_dir)
    # Assume dataset_dir points to the directory containing 'train', 'validate', 'test'
    train_path = dataset_path / 'train'
    val_path = dataset_path / 'validate'
    test_path = dataset_path / 'test'

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Dataset directory structure incomplete in {dataset_dir}. Expected 'train', 'validate', 'test' subdirectories.")

    train_files = [str(f) for f in train_path.glob("*.np[yz]")] # Find .npy or .npz
    val_files_orig = [str(f) for f in val_path.glob("*.np[yz]")]
    test_files = [str(f) for f in test_path.glob("*.np[yz]")]

    if not train_files or not val_files_orig:
        raise FileNotFoundError(f"No training/validation data found in {train_path} / {val_path}")
    if not test_files:
        print(f"Warning: No test data found in {test_path}")

    # Combine train and original validation files for KFold splitting
    all_train_val_files = np.array(train_files + val_files_orig, dtype=str)

    if not (fold.isdigit() and 0 <= int(fold) < 5):
         raise ValueError(f"Fold must be an integer between 0 and 4, got {fold}")
    fold_idx = int(fold)

    # Perform KFold split
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, val_indices = list(folds.split(all_train_val_files))[fold_idx]

    train_files_fold = all_train_val_files[train_indices]
    val_files_fold = all_train_val_files[val_indices]

    # Save the validation file list for this fold
    np.save(fold_dir / f'val_files_{fold}.npy', val_files_fold)
    print(f"Training on fold {fold} with {len(train_files_fold)} train and {len(val_files_fold)} validation files (split from combined train+validate dirs).")

    # --- Create DataLoaders ---
    # Note: The original code adds val_files to train_files. Replicating this behavior.
    # Consider if validation set should be separate or used for training.
    train_files_for_loader = list(train_files_fold) + list(val_files_fold)
    print(f"Creating training loader with {len(train_files_for_loader)} files (train_fold + val_fold)...")

    # Pass target_class to CustomDataLoader only for training
    train_loader_base = CustomDataLoader(train_files_for_loader, batch_size=batch_size, 
                                         LR=False)
    # No filtering needed for validation/test loader
    test_loader_base = CustomDataLoader(test_files, batch_size=1, val=True, LR=False)

    # --- Setup Augmentations ---
    transforms = get_augmentations()

    # --- Create MultiThreadedAugmenters ---
    train_loader = MultiThreadedAugmenter(
        train_loader_base, 
        transforms, 
        num_processes=num_workers_train, 
        num_cached_per_queue=8, # Keep original queue size? Or make configurable?
        pin_memory=True, 
        useroi=use_roi
    )
    test_loader = MultiThreadedAugmenter(
        test_loader_base, 
        None, # No augmentation for test set
        num_processes=num_workers_val, 
        num_cached_per_queue=3, 
        pin_memory=True, 
        useroi=use_roi, 
        val=True
    )

    return train_loader, test_loader, val_files_fold, transforms 