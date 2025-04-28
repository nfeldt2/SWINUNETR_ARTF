import os
from SwinUNeTRGen import SwinUNETRMaskGen
import numpy as np
import ntpath # Use ntpath for OS-agnostic path manipulation

def load_npz(file_path):
    arr = np.load(file_path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        # If .npz, get the first array
        arr = arr[list(arr.keys())[0]]
    return arr

def predict(model_path, device, LR=False, use_roi=False, window_size=(64, 192, 224), overlap=0.25, output_dir_base=None, input_dir=None, limit=None, num_workers=1):
    """
    Predict artifact masks using the SwinUNETR model.
    
    Parameters:
    model_path (str): Path to the model checkpoint file
    device (str): CUDA device to use (e.g., 'cuda:0')
    LR (bool): Use Left/Right specific ROI naming convention
    use_roi (bool): Use ROI masks for prediction
    window_size (tuple): Size of sliding window for inference
    overlap (float): Overlap ratio between adjacent windows (0-1)
    output_dir_base (str): Base directory for output files
    input_dir (str): Directory for input inference datasets (default: /raid/trueArtifacts)
    limit (int): Max number of images to predict (default: all)
    num_workers (int): Number of worker threads for parallel inference (default: 1)
    """
    # adjust window_size: floor to nearest multiple of 32 if >=32, else set to 32
    adjusted_ws = []
    for d in window_size:
        if d >= 32:
            adj = (d // 32) * 32
        else:
            adj = 32
        adjusted_ws.append(adj)
    adjusted_ws = tuple(adjusted_ws)
    if adjusted_ws != window_size:
        print(f"Adjusting window_size from {window_size} to {adjusted_ws} (must be divisible by 32)")
    window_size = adjusted_ws
    # Initialize the mask generator with sliding window support
    maskGen = SwinUNETRMaskGen(
        model_path,
        device=device,
        full_size=True,
        window_size=window_size,
        overlap=overlap
    )
    
    # List all input images from provided inference dataset directory
    src_dir = input_dir if input_dir else "/raid/trueArtifacts"
    true_artf = os.listdir(src_dir)

    true_arts = [img for img in true_artf if "image" in img]
    if use_roi:
        if LR:
            true_rois = [roi for roi in true_artf if "maskArtifactROI_R" in roi]
        else:
            true_rois = [roi for roi in true_artf if "maskArtifactROI." in roi]

        true_arts = np.sort(true_arts)
        true_rois = np.sort(true_rois)
        if LR:
            temp = []
            for i in range(len(true_arts)):
                temp.append((true_rois[i], true_rois[i].replace("_R", "_L")))

            true_rois = temp

    # Apply optional limit on number of images to process
    if limit is not None:
        true_arts = true_arts[:limit]
        if use_roi:
            true_rois = true_rois[:limit]

    # Determine a meaningful suffix for the output directory from the model path
    model_dir_name = ntpath.basename(ntpath.dirname(model_path)) # Get the parent directory name (e.g., fold_0)
    output_suffix = model_dir_name if model_dir_name else "output" # Use parent dir name or default

    # Determine output directory: use provided or default '/raid/trueArtifacts_{suffix}'
    if output_dir_base:
        output_dir = output_dir_base
    else:
        output_dir = f"/raid/trueArtifacts_{output_suffix}"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare list of jobs: tuples (image_filename, roi_info)
    jobs = []
    for i, img_name in enumerate(true_arts):
        # get ROI for this image
        if use_roi:
            roi_item = true_rois[i]
        else:
            roi_item = None
        jobs.append((img_name, roi_item))
    # Apply limit if provided
    if limit is not None:
        jobs = jobs[:limit]
    # Function to process a single image
    def process_job(job):
        img_file, roi_item = job
        # Determine paths
        name_no_ext, _ = os.path.splitext(img_file)
        pred_name = name_no_ext.replace("image_", "predMask_")
        save_base = os.path.join(output_dir, pred_name)
        save_path = f"{save_base}.npz"
        if os.path.exists(save_path):
            print(f"Skipping {img_file} (exists)")
            return
        img = load_npz(os.path.join(src_dir, img_file))
        # Load ROI(s)
        if use_roi and roi_item:
            if isinstance(roi_item, tuple):
                roi1 = load_npz(os.path.join(src_dir, roi_item[0]))
                roi2 = load_npz(os.path.join(src_dir, roi_item[1]))
                roi = (roi1, roi2)
            else:
                roi = load_npz(os.path.join(src_dir, roi_item))
        else:
            roi = np.ones_like(img)
        # Generate mask
        mask1 = maskGen(img, roi, roi, img_file)
        # Save
        np.savez_compressed(save_base, mask1)
        print(f"Saved {save_path}")
    # Run jobs in ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_job, jobs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SwinUNETR prediction on artifact images.")
    parser.add_argument("--model_path", type=str, required=True, help="Full path to the model checkpoint file (.pt)")
    parser.add_argument("--device", type=str, required=True, help="CUDA device to use (e.g., cuda:0, cuda:1)")
    parser.add_argument("--LR", action="store_true", help="Use Left/Right specific ROI naming convention.")
    parser.add_argument("--roi", action="store_true", default=False, help="Use ROI masks for prediction (required).")
    # Add parameters for sliding window
    parser.add_argument("--window_size", type=int, nargs=3, default=[64, 192, 224], 
                        help="Size of sliding window for inference (default: 64 192 224)")
    parser.add_argument("--overlap", type=float, default=0.2, 
                        help="Overlap ratio between adjacent windows, 0-1 (default: 0.25)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save prediction outputs (overrides default)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory for input inference datasets (default: /raid/trueArtifacts)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of images to predict (default: all)")
    parser.add_argument("--num_workers", type=int, default=6,
                        help="Number of worker threads for parallel inference (default: 1)")
    
    args = parser.parse_args()
    predict(
        args.model_path,
        args.device,
        args.LR,
        args.roi,
        tuple(args.window_size),
        args.overlap,
        output_dir_base=args.output_dir,
        input_dir=args.input_dir,
        limit=args.limit,
        num_workers=args.num_workers
    )

    

