import os
from SwinUNeTRGen import SwinUNETRMaskGen
import numpy as np

def predict(model, LR=False, use_roi=True, fold=0):
    maskGen = SwinUNETRMaskGen(f"/home/nathan/Desktop/SWINUNETR_ARTF/{model}/fold_{fold}/checkpoint_best.pt", device='cuda', full_size=True)
    true_artf = os.listdir("/raid/trueArtifacts_updated")

    true_arts = [img for img in true_artf if "CTinhaleArtf" in img]
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

    for i in range(len(true_arts)):
        img = np.load("/raid/trueArtifacts_updated/" + true_arts[i])
        if use_roi:
            if type(true_rois[i]) == tuple:
                roi1 = np.load("/raid/trueArtifacts_updated/" + true_rois[i][0])
                roi2 = np.load("/raid/trueArtifacts_updated/" + true_rois[i][1])
                roi = (roi1, roi2)
            else:
                roi = np.load("/raid/trueArtifacts_updated/" + true_rois[i])
        else:
            roi = np.ones_like(img)
        
        mask1 = maskGen(img, roi, roi)

        os.makedirs(f"/raid/trueArtifacts{model}{fold}", exist_ok=True)
        np.save(f"/raid/trueArtifacts{model}{fold}/" + true_arts[i].replace("CTinhaleArtf", "predMask"), mask1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--LR", action="store_true")
    parser.add_argument("--roi", action="store_true")
    parser.add_argument("-fold", type=str, required=False)
    args = parser.parse_args()
    predict(args.model, args.LR, args.roi, args.fold)

    

