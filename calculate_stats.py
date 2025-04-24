import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import warnings
import concurrent.futures # Import concurrent.futures

warnings.filterwarnings("ignore", message=".*Reading from failed file.*") # Suppress specific warnings if needed

def find_image_files(data_dir, file_pattern):
    """Finds image files based on pattern and '_image_' substring."""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        print(f"Warning: Directory not found - {data_dir}")
        return []
    potential_image_files = sorted([str(f) for f in data_path.glob(file_pattern) if '_image_' in f.name])
    return potential_image_files

def process_single_file(img_path):
    """Loads a single image file and returns its sum, sum_squares, and voxel count."""
    try:
        with np.load(img_path, allow_pickle=False) as img_npz:
            if 'arr_0' not in img_npz:
                # Return None for errors to be handled later
                # tqdm.write(f"Warning: 'arr_0' key not found in {img_path}. Skipping.") 
                return None 
            
            img = img_npz['arr_0'].astype(np.float64) 
            
            current_voxels = img.size
            if current_voxels == 0:
                # tqdm.write(f"Warning: Image array is empty in {img_path}. Skipping.")
                return None

            current_sum = np.sum(img)
            current_sum_squares = np.sum(np.square(img))
            
            return current_sum, current_sum_squares, current_voxels

    except FileNotFoundError:
         # tqdm.write(f"Error: File not found {img_path}. Skipping.")
         return None
    except ValueError as ve:
         # tqdm.write(f"Error loading or processing {img_path}: {ve}. Skipping.")
         return None
    except Exception as e:
         # tqdm.write(f"An unexpected error occurred processing {img_path}: {e}. Skipping.")
         return None

def calculate_stats_parallel(image_paths, num_threads=16):
    """Calculates mean and std dev in parallel using ThreadPoolExecutor."""
    total_voxels = 0
    total_sum = 0.0
    total_sum_squares = 0.0
    processed_count = 0
    error_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(process_single_file, path): path for path in image_paths}
        
        # Process results as they complete, using tqdm for progress
        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(image_paths), desc="Calculating Stats"):
            result = future.result()
            if result is not None:
                current_sum, current_sum_squares, current_voxels = result
                total_sum += current_sum
                total_sum_squares += current_sum_squares
                total_voxels += current_voxels
                processed_count += 1
            else:
                # Optionally log the error path: path = future_to_path[future]
                error_count += 1

    if processed_count == 0:
        raise ValueError("No valid image data found or processed successfully to calculate statistics.")
    
    if error_count > 0:
        print(f"\nWarning: Encountered errors processing {error_count} files.")

    # Calculate mean
    mean = total_sum / total_voxels
    
    # Calculate variance using E[X^2] - (E[X])^2 formula
    variance = (total_sum_squares / total_voxels) - np.square(mean)
    
    # Handle potential floating point inaccuracies leading to slightly negative variance
    if variance < 0:
         print(f"Warning: Calculated variance is slightly negative ({variance:.4e}). Clamping to zero.")
         variance = 0.0
         
    std_dev = np.sqrt(variance)

    return mean, std_dev

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate global mean and std dev for training/validation images.")
    parser.add_argument('dataset_dir', type=str, help="Directory containing 'train' and 'validate' subfolders.")
    parser.add_argument('--file_pattern', type=str, default='*.np[yz]', help="Glob pattern for data files (e.g., '*.npy', '*.npz').")
    parser.add_argument('--num_threads', type=int, default=16, help="Number of threads for parallel processing.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    train_path = dataset_path / 'train'
    val_path = dataset_path / 'validate'

    # Find image files in both train and validate directories
    print(f"Finding image files in {train_path}...")
    train_images = find_image_files(train_path, args.file_pattern)
    print(f"Finding image files in {val_path}...")
    val_images = find_image_files(val_path, args.file_pattern)
    all_train_val_images = train_images + val_images

    if not all_train_val_images:
         raise ValueError("No image files matching the pattern found in 'train' or 'validate' directories.")

    print(f"Found {len(all_train_val_images)} total training/validation image files.")
    print(f"Calculating mean and standard deviation using {args.num_threads} threads...")
    
    # Use the parallel calculation function
    global_mean, global_std_dev = calculate_stats_parallel(all_train_val_images, num_threads=args.num_threads)

    print("\n" + "-"*20)
    print(f"Calculation Complete:")
    print(f"  Global Mean: {global_mean}")
    print(f"  Global Std Dev: {global_std_dev}")
    print("-"*20 + "\n")
    print("Please use these values in your MONAI NormalizeIntensityd transform:")
    print(f"NormalizeIntensityd(keys=[img_key], subtrahend={global_mean:.6f}, divisor={global_std_dev:.6f})") 