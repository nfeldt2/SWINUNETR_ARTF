import os
from SwinUNeTRGen import SwinUNETRMaskGen
import numpy as np
import ntpath # Use ntpath for OS-agnostic path manipulation

def predict(model_path, device, LR=False, use_roi=False): # Changed model->model_path, added device, removed use_roi default
    maskGen = SwinUNETRMaskGen(model_path, device=device, full_size=True) # Use model_path and device args
    true_artf = os.listdir("/raid/trueArtifacts")

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

    # Determine a meaningful suffix for the output directory from the model path
    model_dir_name = ntpath.basename(ntpath.dirname(model_path)) # Get the parent directory name (e.g., fold_0)
    output_suffix = model_dir_name if model_dir_name else "output" # Use parent dir name or default

    output_dir = f"/raid/trueArtifacts_{output_suffix}" # Create output dir name
    os.makedirs(output_dir, exist_ok=True) # Create the directory

    for i in range(len(true_arts)):
        img = np.load("/raid/trueArtifacts/" + true_arts[i])
        if use_roi:
            if type(true_rois[i]) == tuple:
                roi1 = np.load("/raid/trueArtifacts/" + true_rois[i][0])
                roi2 = np.load("/raid/trueArtifacts/" + true_rois[i][1])
                roi = (roi1, roi2)
            else:
                roi = np.load("/raid/trueArtifacts/" + true_rois[i])
        else:
            # If not using ROI, create a full mask, but this branch might be less relevant now ROI is required
            roi = np.ones_like(img)

        mask1 = maskGen(img, roi, roi) # roi is passed twice based on original code
        # Remove the old model name splitting logic for output path
        np.savez_compressed(f"{output_dir}/" + true_arts[i].replace("image", "predMask").replace(".npy", ""), mask1) # Use the new output_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SwinUNETR prediction on artifact images.")
    parser.add_argument("--model_path", type=str, required=True, help="Full path to the model checkpoint file (.pt)") # Changed --model to --model_path
    parser.add_argument("--device", type=str, required=True, help="CUDA device to use (e.g., cuda:0, cuda:1)") # Added --device argument
    parser.add_argument("--LR", action="store_true", help="Use Left/Right specific ROI naming convention.")
    parser.add_argument("--roi", action="store_true", required=True, help="Use ROI masks for prediction (required).") # Made --roi required
    # Removed -fold argument as it's implicitly handled by model_path parent dir
    args = parser.parse_args()
    predict(args.model_path, args.device, args.LR, args.roi) # Updated function call

    

