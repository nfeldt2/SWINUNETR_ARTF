# training_utils/setup.py

import sys
import time
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
import torch
import os
import concurrent.futures
from tqdm import tqdm

# --- BatchGenerators Imports ---
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose as BGCompose
# Important: Use the correct base class alias as used in classify_artifacts.py
from batchgenerators.dataloading.data_loader import DataLoader as BGDataLoader

# --- MONAI Imports ---
from monai.data import Dataset as MonaiDataset, DataLoader as MonaiDataLoader
# Use MONAI Compose for the transform pipeline passed to the custom loader
from monai.transforms import Compose as MonaiCompose
from monai.data import pad_list_data_collate, list_data_collate

# --- Custom Local Imports ---
try:
    # get_augmentations should return a BGCompose object or None
    from Augmentations import get_augmentations
except ImportError:
     print("Warning: Could not import get_augmentations from Augmentations.py. No BG augmentations will be applied.")
     # Define a dummy function returning an empty BGCompose if import fails
     def get_augmentations(): return BGCompose([])

# --- Define get_image_label (copied from train.py for use here) ---
def get_image_label(segmentation_mask_data: np.ndarray, min_pixels_threshold: int = 100) -> int:
    """ Derives binary label (1 if Class 1 present >= threshold, else 0) """
    # Check only for class 1 based on original goal description in train.py's Trainer class
    if 1 in np.unique(segmentation_mask_data):
        count1 = np.count_nonzero(segmentation_mask_data == 1)
        if count1 >= min_pixels_threshold:
             return 1 # Class 1 is present and significant
    return 0 # Class 1 is not present or not significant

# --- REWRITTEN MultiTaskDataLoader mimicking ArtifactClassificationDataLoader ---
class MultiTaskDataLoader(BGDataLoader):
    """
    batchgenerators compatible DataLoader REWRITTEN to mimic classify_artifacts.py.
    Applies a pre-composed MONAI transform pipeline sequentially inside generate_train_batch.
    Derives classification labels after MONAI transforms.
    Yields batches of NumPy arrays with 'data', 'seg', and 'label' keys
    ready for batchgenerators.MultiThreadedAugmenter.
    """
    # Inside MultiTaskDataLoader class in training_utils/setup.py

    def __init__(self, data_dicts, batch_size, monai_transforms: MonaiCompose,
                 min_pixels_threshold: int, # Added for label derivation
                 num_threads_in_multithreaded=1):
        # ... (super init, store transforms, threshold, num_samples) ...
        super().__init__(data_dicts, batch_size, num_threads_in_multithreaded)
        if not isinstance(monai_transforms, MonaiCompose): raise TypeError("monai_transforms must be MONAI Compose.")
        self.monai_transforms = monai_transforms
        self.min_pixels_threshold = min_pixels_threshold
        self.num_samples = len(data_dicts)
        # Define keys expected FROM monai_transforms
        self.monai_output_keys = ['image', 'seg'] # <--- Use 'image', 'seg'
        # Define keys TO BE OUTPUT by this loader
        self.final_output_keys = ['data', 'seg', 'label'] # <--- Final keys

    def __len__(self):
        return self.num_samples

    def generate_train_batch(self):
        # ... (get indices, setup lists) ...
        indices = self.get_indices()
        batch_data_np = []; batch_seg_np = []; batch_labels_np = []
        skipped_count = 0; processed_indices_debug = []

        for idx in indices:
            # ... (get data_dict_i, filename) ...
            data_dict_i = self._data[idx]; img_path_str = data_dict_i.get('image_path', 'unknown_image'); filename = Path(img_path_str).name

            try:
                # Apply MONAI pipeline (outputs 'image', 'seg' Tensors)
                transformed_data_monai = self.monai_transforms(data_dict_i)

                # Validate MONAI output uses the NEW expected keys
                if not isinstance(transformed_data_monai, dict):
                     tqdm.write(f"Error: MONAI transforms not dict for {filename}. Skip."); skipped_count += 1; continue
                if not all(k in transformed_data_monai for k in self.monai_output_keys): # Check for 'image', 'seg'
                     tqdm.write(f"Error: MONAI output missing keys for {filename}. Exp: {self.monai_output_keys}, Got: {list(transformed_data_monai.keys())}. Skip."); skipped_count += 1; continue

                # Get Tensors using 'image', 'seg' keys
                image_tensor = transformed_data_monai['image'] # <--- Use 'image'
                seg_tensor = transformed_data_monai['seg']     # <--- Use 'seg'

                # Convert Tensors to NumPy
                data_np_sample = image_tensor.cpu().numpy()
                seg_np_sample = seg_tensor.cpu().numpy()

                # Derive Label (same as before)
                label_sample = 0
                if seg_np_sample.ndim >= 3:
                     seg_for_label = seg_np_sample[0] if seg_np_sample.ndim == 4 else seg_np_sample
                     label_sample = get_image_label(seg_for_label, self.min_pixels_threshold)
                # else: tqdm.write(f"Warning: Unexpected seg shape {seg_np_sample.shape}. Label=0.") # Reduce noise

                # Append NumPy arrays
                batch_data_np.append(data_np_sample)
                batch_seg_np.append(seg_np_sample)
                batch_labels_np.append(label_sample)
                processed_indices_debug.append(idx)

            except Exception as e: # ... (error handling) ...
                import traceback; tqdm.write(f"Error processing {idx} ({filename}): {e}\n{traceback.format_exc()}"); skipped_count += 1; continue

        if not batch_data_np: # ... (handle empty batch) ...
             return {'data': np.array([]), 'seg': np.array([]), 'label': np.array([])}

        try: # ... (stack arrays) ...
            data_batch_np_stacked = np.stack(batch_data_np, axis=0)
            seg_batch_np_stacked = np.stack(batch_seg_np, axis=0)
            label_batch_np_stacked = np.array(batch_labels_np, dtype=np.int64)

            # Return dict using FINAL output keys ('data', 'seg', 'label')
            output_dict = {
                'data': data_batch_np_stacked,  # <--- Key is 'data'
                'seg': seg_batch_np_stacked,   # <--- Key is 'seg'
                'label': label_batch_np_stacked
            }
            return output_dict
        except Exception as stack_e: # ... (handle stack error) ...
            import traceback; tqdm.write(f"Error stacking batch: {stack_e}\n{traceback.format_exc()}"); return {'data': np.array([]), 'seg': np.array([]), 'label': np.array([])}


# --- setup_dataloaders REWRITTEN to use new MultiTaskDataLoader ---
def setup_dataloaders(
    dataset_dir: str, fold: str, fold_dir: Path, batch_size: int,
    num_workers_train: int, num_workers_val: int,
    # Pass MONAI transforms and threshold for label derivation
    monai_train_transforms: MonaiCompose,
    monai_val_transforms: MonaiCompose,
    min_artifact_pixels: int, # Needed by new loader
    # Keep other args used by file pairing etc.
    file_pattern: str = "*.np[yz]", img_suffix: str = "_image_",
    seg_suffix: str = "_maskArtifact_",
    is_multi_task: bool = True, # Keep this flag if used elsewhere
    **kwargs # Capture unused args
):
    print(f"--- Setting up DataLoaders (REWRITTEN to mimic classify_artifacts.py) ---")
    dataset_path = Path(dataset_dir); train_path = dataset_path / 'train'
    val_path = dataset_path / 'validate'; test_path = dataset_path / 'test'
    if not all([p.exists() for p in [train_path, val_path, test_path]]): raise FileNotFoundError(f"Dataset structure incomplete in {dataset_dir}.")

    # --- File Pairing Logic (remains the same) ---
    def find_and_pair_files_parallel(data_dir, pattern, img_suff, seg_suff, num_threads=None):
        # ... (same parallel file pairing logic as before) ...
        if num_threads is None: num_threads = max(1, os.cpu_count() // 2)
        paired = []; missing_count = 0
        print(f" Searching {data_dir} using up to {num_threads} threads...")
        potential_images = sorted([str(f) for f in data_dir.glob(pattern) if img_suff in f.name])
        print(f" Found {len(potential_images)} potential images. Pairing...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_path = {executor.submit(check_and_pair_file_worker, img_path, img_suff, seg_suff): img_path for img_path in potential_images}
            for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(potential_images), desc=f"Pairing {data_dir.name}", leave=False):
                try:
                    result = future.result()
                    if result["status"] == "paired": paired.append({"image_path": result["image_path"], "seg_path": result["seg_path"]})
                    else: missing_count += 1
                except Exception as exc: missing_count += 1
        print(f" Paired {len(paired)} files in {data_dir.name}. {missing_count} missing/error.")
        return paired

    train_files_pairs = find_and_pair_files_parallel(train_path, file_pattern, img_suffix, seg_suffix, num_workers_train)
    val_files_pairs = find_and_pair_files_parallel(val_path, file_pattern, img_suffix, seg_suffix, num_workers_train)
    test_files_pairs = find_and_pair_files_parallel(test_path, file_pattern, img_suffix, seg_suffix, num_workers_val)

    if not train_files_pairs and not val_files_pairs: raise FileNotFoundError("No paired train/val data.")
    if not test_files_pairs: print("Warning: No test data found for validation.")

    # --- KFold Split (remains the same) ---
    all_train_val_pairs = np.array(train_files_pairs + val_files_pairs, dtype=object)
    if not (fold.isdigit() and 0 <= int(fold) < 5): raise ValueError(f"Invalid fold: {fold}")
    fold_idx = int(fold); folds = KFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, _ = list(folds.split(all_train_val_pairs))[fold_idx]
    train_files_fold_pairs = all_train_val_pairs[train_indices].tolist()
    val_files_fold_pairs = test_files_pairs # Use test set for validation

    val_image_paths_fold = [d['image_path'] for d in val_files_fold_pairs]
    np.save(fold_dir / f'val_files_{fold}.npy', np.array(val_image_paths_fold))
    print(f"Using Fold {fold}: {len(train_files_fold_pairs)} train samples, {len(val_files_fold_pairs)} validation samples (from test dir).")

    # --- Get BatchGenerators Augmentations ---
    bg_transforms = get_augmentations() # From Augmentations.py
    if isinstance(bg_transforms, BGCompose):
         print(f"Loaded {len(bg_transforms.transforms)} BatchGenerators transforms.")
    else: print("Warning: get_augmentations() did not return BGCompose. No BG transforms applied.")

    # --- Instantiate DataLoaders (REWRITTEN) ---
    if not monai_train_transforms or not monai_val_transforms:
        raise ValueError("monai_train_transforms and monai_val_transforms must be provided.")

    print(f" Initializing REWRITTEN MultiTaskDataLoader for {len(train_files_fold_pairs)} training files...")
    # Instantiate the NEW base loader, passing MONAI transforms and pixel threshold
    train_loader_base = MultiTaskDataLoader(
        data_dicts=train_files_fold_pairs,
        batch_size=batch_size,
        monai_transforms=monai_train_transforms, # The simplified MONAI pipeline
        min_pixels_threshold=min_artifact_pixels # Threshold for label derivation
        # num_threads_in_multithreaded can be default (1) or adjusted
    )
    print(f" Base training loader (MultiTaskDataLoader) initialized. Length: {len(train_loader_base)}")

    print(" Initializing MultiThreadedAugmenter for training...")
    # Wrap the base loader with MTA. Apply BG transforms. DO NOT pass indices.
    train_loader = MultiThreadedAugmenter(
        data_loader=train_loader_base,  # Instance of the new MultiTaskDataLoader
        transform=bg_transforms,        # BG transforms from Augmentations.py
        num_processes=num_workers_train,
        num_cached_per_queue=2, # Adjust as needed
        pin_memory=True,
        # Do NOT pass 'indices' here - rely on base loader's internal handling
    )
    print(" Training Loader (MultiThreadedAugmenter) initialized.")

    # --- Validation Loader (Standard MONAI - remains mostly the same) ---
    print(f" Initializing MONAI Dataset/DataLoader for {len(val_files_fold_pairs)} validation files...")
    # Validation uses standard MONAI Dataset/DataLoader with its specific transforms
    val_ds = MonaiDataset(data=val_files_fold_pairs, transform=monai_val_transforms) if val_files_fold_pairs else None
    val_loader = MonaiDataLoader(
        val_ds, batch_size=1, # Usually batch size 1 for validation/inference
        shuffle=False, num_workers=num_workers_val,
        pin_memory=torch.cuda.is_available(),
        collate_fn=list_data_collate # Use standard MONAI collate
    ) if val_ds else None
    print(" Validation Loader (MONAI) initialized." if val_loader else " Validation Loader is None.")

    return train_loader, val_loader, val_image_paths_fold, bg_transforms


# --- Helper Functions (setup_output_directory, check_and_pair_file_worker remain the same) ---
def setup_output_directory(output_dir: str, fold: str, dataset_name: str, continue_tr: bool):
    # ... (same as before) ...
    base_path = Path(output_dir); dataset_path = base_path / dataset_name
    fold_path = dataset_path / f"fold_{fold}"; fold_path.mkdir(parents=True, exist_ok=True)
    checkpoint_exists = (fold_path / 'checkpoint_latest.pt').exists()
    if checkpoint_exists and not continue_tr: print(f"WARN: Checkpoint exists but --c not used.")
    elif not checkpoint_exists and continue_tr: print(f"WARN: --c used but no checkpoint found.")
    log_file_name = "log0.txt"; i = 0
    while (fold_path / log_file_name).exists():
        if continue_tr: break # Append to latest log if continuing
        i += 1; log_file_name = f"log{i}.txt"
    log_file_path = fold_path / log_file_name
    log_mode = 'a' if continue_tr and log_file_path.exists() else 'w'
    try: log_file = open(log_file_path, mode=log_mode)
    except IOError as e: print(f"Error opening log file {log_file_path}: {e}"); log_file = sys.stdout
    print(f"Output directory: {fold_path}"); print(f"Logging to {log_file_path} (mode: {log_mode})")
    return fold_path, log_file, checkpoint_exists

def check_and_pair_file_worker(img_path_str, img_suffix, seg_suffix):
    # ... (same as before) ...
    img_path = Path(img_path_str)
    expected_seg_name = img_path.name.replace(img_suffix, seg_suffix, 1)
    seg_p = img_path.with_name(expected_seg_name)
    if not seg_p.is_file(): seg_p = seg_p.with_suffix('.npy')
    if not seg_p.is_file(): seg_p = seg_p.with_suffix('.npz')
    if seg_p.is_file():
        return {"image_path": img_path_str, "seg_path": str(seg_p), "status": "paired"}
    else:
        return {"status": "missing_label", "image_name": img_path.name}