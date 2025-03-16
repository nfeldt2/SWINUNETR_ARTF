import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path
from tqdm import tqdm

def process_file(npz_path, output_base_dir):
    try:
        # Load NPZ file
        data = np.load(npz_path)
        
        # Create corresponding output path
        rel_path = os.path.relpath(npz_path, '/raid/swin_npz/NPZ_S2_roi')
        output_path = os.path.join(output_base_dir, rel_path)
        output_path = output_path.replace('.npz', '.pt')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to torch tensor and save
        # If NPZ file has multiple arrays, save them all in a dictionary
        torch_data = {key: torch.from_numpy(data[key]) for key in data.files}
        if len(torch_data) == 1:
            # If there's only one array, save it directly
            torch.save(torch_data[data.files[0]], output_path)
        else:
            # If there are multiple arrays, save as dictionary
            torch.save(torch_data, output_path)
            
    except Exception as e:
        print(f"Error processing {npz_path}: {str(e)}")

def main():
    input_dir = '/raid/swin_npz/NPZ_S2_roi'
    output_base_dir = '/raid/swin_npz/TORCH_S2_roi'
    
    # Create output base directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Collect all NPZ files
    npz_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(root, file))
    
    print(f"Found {len(npz_files)} NPZ files to process")
    
    # Process files using thread pool
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(
            executor.map(lambda x: process_file(x, output_base_dir), npz_files),
            total=len(npz_files),
            desc="Converting files"
        ))

if __name__ == "__main__":
    main() 