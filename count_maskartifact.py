#!/usr/bin/env python3
import os
import glob
import numpy as np

def main():
    # Directory containing the mask files
    directory = '/raid/addedArtifacts_S1/validate'
    # Pattern to match .npz and .npy files with 'maskArtifact_' in name
    pattern = os.path.join(directory, '*maskArtifact_*np[yz]')
    file_paths = glob.glob(pattern)
    total_files = len(file_paths)
    count_with_one = 0

    for path in file_paths:
        try:
            data = np.load(path)
            # support both .npz and .npy
            if isinstance(data, np.lib.npyio.NpzFile):
                arr = data.get('arr_0')
            else:
                arr = data
            if arr is None:
                print(f"Warning: no 'arr_0' in {path}")
            elif np.any(arr == 1):
                count_with_one += 1
        except Exception as e:
            print(f"Error loading {path}: {e}")

    print(f"Total 'maskArtifact_' files found: {total_files}")
    print(f"Files containing class 1: {count_with_one}")

if __name__ == '__main__':
    main() 