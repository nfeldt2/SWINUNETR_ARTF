import numpy as np
import torch
from skimage.transform import resize
# from monai.networks.nets import SwinUNETR # Comment out MONAI import
from swin_unetr import SwinUNETR # Import from local custom file
import matplotlib.pyplot as plt
import os
from monai.inferers import sliding_window_inference


class SwinUNETRMaskGen:
    def __init__(self, weights_path, device='cuda:1', full_size=True, window_size=(96, 96, 96), overlap=0.5):
        """
        Initialize the MaskPreparer with the model weights and device.
        
        Parameters:
        weights_path (str): Path to the model weights (state_dict).
        device (str): Device to run the model on ('cuda' or 'cpu').
        full_size (bool): Whether to process the image at full size or resize.
        window_size (tuple): Size of sliding window for inference.
        overlap (float): Overlap ratio between adjacent windows (0-1).
        """
        print(device)
        self.window_size = window_size
        self.overlap = overlap
        self.mean = 0.412456
        self.std = 0.278396
        self.step_count = 0 # Add step counter
        # Try feature_size=24, if it fails, try feature_size=12
        try:
            self.model = SwinUNETR(
                img_size=window_size,
                in_channels=1,
                out_channels=1,  # single-channel output
                feature_size=24,
                deep_supervision=False,
                use_v2=True
            )
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict)
        except Exception:
            self.model = SwinUNETR(
                img_size=window_size,
                in_channels=1,
                out_channels=1,  # single-channel output
                feature_size=12,
                deep_supervision=False,
                use_v2=True
            )
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.device = device
        self.model.eval()
        self.full_size = full_size
    
    def __call__(self, img, mask, rois, img_name):
        """
        Generate the mask for the given image and region of interest (ROI).
        
        Parameters:
        img (numpy.ndarray): Input image.
        mask (numpy.ndarray): Input mask.
        roi (numpy.ndarray): Region of interest.

        Returns:
        numpy.ndarray: Processed mask.
        """
        orig_img = img.copy()

        if type(rois) == tuple:
            pass
        else:
            rois = [rois]

        out_masks = []

        for roi in rois:
            if type(mask) == tuple:
                mask = roi

            #min_x, min_y, min_z, max_x, max_y, max_z = self._get_roi_bounds(roi)
            
            # Extract ROI
            roi_img = img
            roi_mask = mask
            
            # Apply sliding window if ROI is larger than window size
            if (roi_img.shape[0] > self.window_size[0] or 
                roi_img.shape[1] > self.window_size[1] or 
                roi_img.shape[2] > self.window_size[2]):
                # Use MONAI sliding_window_inference for patch-wise averaging
                img_t = torch.tensor(roi_img, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    # Use a predictor that returns only the segmentation output (ignore classification head)
                    def seg_predictor(patch):
                        out = self.model(patch)
                        seg_logits = out[0] if isinstance(out, tuple) else out
                        return torch.sigmoid(seg_logits)
                    pred = sliding_window_inference(
                        inputs=img_t,
                        roi_size=self.window_size,
                        sw_batch_size=1,
                        predictor=seg_predictor,
                        overlap=self.overlap,
                        mode="gaussian"
                    )
                # pred shape [1,1,D,H,W] after sigmoid
                prob_map = pred[0, 0].cpu().numpy()  # [D, H, W]
            else:
                # For smaller ROIs, resize to window size, process, then resize back
                print(f"Warning: ROI is smaller than window size. Image name: {img_name}")
                resized_img = resize(roi_img, self.window_size, mode='constant', order=1, cval=0)
                normalized_img = self._normalize_image(resized_img)

                with torch.no_grad():
                    out = self.model(normalized_img.unsqueeze(0).unsqueeze(0).to(self.device))
                    if type(out) == tuple:
                        out = out[0]
                    out_prob = torch.sigmoid(out)  # single-channel probability
                prob_map_resized = out_prob[0, 0].detach().cpu().numpy()

                # Resize probability maps back to original ROI size
                prob_map = resize(prob_map_resized, roi_img.shape, mode='constant', order=1)

                torch.cuda.empty_cache()

            # threshold single-channel probability into binary mask
            output_mask = (prob_map >= 0.5).astype(np.uint8)

            # Reconstruct full image mask and probability maps
            full_mask = np.zeros_like(img, dtype=output_mask.dtype)
            full_prob_map = np.zeros_like(img, dtype=prob_map.dtype)

            # Place results back into the full image shape (assuming single ROI for now or non-overlapping)
            # If using min/max bounds again, uncomment and adjust placement
            # full_mask[min_x:max_x, min_y:max_y, min_z:max_z] = output_mask
            # full_prob_map1[min_x:max_x, min_y:max_y, min_z:max_z] = prob_map1
            # full_prob_map2[min_x:max_x, min_y:max_y, min_z:max_z] = prob_map2
            # full_prob_map3[min_x:max_x, min_y:max_y, min_z:max_z] = prob_map3
            # Using direct assignment assuming roi_img == img as per previous changes
            full_mask = output_mask
            full_prob_map = prob_map

            # Resize back if needed (logic seems commented out or simplified)
            if not self.full_size:
                 print("Error: Full size Lost, interpolating to original size - MASK ONLY")
                 # This resizing logic might need adjustment for probability maps too
                 # full_mask = full_mask[:pre_roi[0], :pre_roi[1], :pre_roi[2]] # pre_roi is undefined now
                 full_mask = resize(full_mask, (orig_img.shape[0], orig_img.shape[1], orig_img.shape[2]),
                                 mode='constant', order=0, cval=0) # order=0 for mask
            else:
                 # Ensure mask is clipped to original image bounds
                 full_mask = full_mask[:orig_img.shape[0], :orig_img.shape[1], :orig_img.shape[2]]
                 full_prob_map = full_prob_map[:orig_img.shape[0], :orig_img.shape[1], :orig_img.shape[2]]


            # No thresholding needed for argmax mask
            # final_mask = (full_mask > 0.95).astype(np.int32)
            out_masks.append(full_mask) # Appending argmax result {0, 1, 2}

        # Aggregation logic might need review if multiple ROIs overlap
        # Current logic sums argmax results, which is unusual.
        # For visualization, we use the maps from the *last* ROI processed.
        final_aggregated_mask = np.zeros_like(out_masks[0])
        for mask in out_masks:
            final_aggregated_mask += mask

        # (debug overlay saving logic removed)

        self.step_count += 1 # Increment step counter

        # Return the aggregated final mask (argmax result)
        return final_aggregated_mask

    def _sliding_window_inference(self, img, batch_size=1, save_windows=False, save_prefix=None):
        stride = [int(self.window_size[i] * (1 - self.overlap)) for i in range(3)]
        pad_sizes = []
        for i in range(3):
            remainder = (img.shape[i] - self.window_size[i]) % stride[i]
            if remainder != 0 and img.shape[i] > self.window_size[i]:
                pad_size = stride[i] - remainder
            else:
                pad_size = 0
            pad_sizes.append(pad_size)
        padded_img = np.pad(img, ((0, pad_sizes[0]), (0, pad_sizes[1]), (0, pad_sizes[2])),
                           mode='constant', constant_values=0)
        # Initialize accumulators for all 3 channels
        output_cls1 = np.zeros_like(padded_img, dtype=np.float32)
        output_cls2 = np.zeros_like(padded_img, dtype=np.float32)
        output_cls3 = np.zeros_like(padded_img, dtype=np.float32)
        count_map = np.zeros_like(padded_img, dtype=np.float32)

        # before computing coords:
        pad_lower = []
        pad_upper = []
        for i in (0,1,2):
            # ensure shape >= window_size
            lower = max(0, self.window_size[i] - img.shape[i])
            # then make (img + lower) - window_size divisible by stride
            rem   = ((img.shape[i] + lower) - self.window_size[i]) % stride[i]
            upper = (stride[i] - rem) % stride[i]
            pad_lower.append(lower//2)
            pad_upper.append(lower - lower//2 + upper)

        padded = np.pad(
            img,
            ((pad_lower[0], pad_upper[0]),
            (pad_lower[1], pad_upper[1]),
            (pad_lower[2], pad_upper[2])),
            mode='constant', constant_values=0
        )

        # Prepare window coordinates
        coords = []
        for x in range(0, max(1, padded_img.shape[0] - self.window_size[0] + 1), stride[0]):
            for y in range(0, max(1, padded_img.shape[1] - self.window_size[1] + 1), stride[1]):
                for z in range(0, max(1, padded_img.shape[2] - self.window_size[2] + 1), stride[2]):
                    coords.append((x, y, z))

        # Batch processing
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i+batch_size]
            batch_windows = []
            for (x, y, z) in batch_coords:
                window = padded_img[
                    x:x + self.window_size[0],
                    y:y + self.window_size[1],
                    z:z + self.window_size[2]
                ]
                if window.shape != self.window_size:
                    print(f"Skipping window at ({x},{y},{z}) due to unexpected shape {window.shape}")
                    continue # Skip if window shape is incorrect
                batch_windows.append(self._normalize_image(window))

            if not batch_windows:
                continue

            # Ensure batch_windows is not empty before stacking
            actual_batch_coords = batch_coords[:len(batch_windows)] # Coords corresponding to valid windows

            batch_tensor = torch.stack(batch_windows).unsqueeze(1).to(self.device)
            with torch.no_grad():
                out = self.model(batch_tensor)
                if type(out) == tuple:
                    out = out[0]
                out = torch.softmax(out, dim=1) # Use softmax for probabilities

            for j, (x, y, z) in enumerate(actual_batch_coords): # Iterate over actual coords used
                # Get probabilities for all 3 channels
                pred_cls1 = out[j, 0].detach().cpu().numpy()
                pred_cls2 = out[j, 1].detach().cpu().numpy()
                pred_cls3 = out[j, 2].detach().cpu().numpy()

                # --- Modify window saving logic for overlay ---
                if save_windows and save_prefix:
                    try:
                        save_dir = os.path.join(os.getcwd(), save_prefix)
                        os.makedirs(save_dir, exist_ok=True)

                        # Get original (non-normalized) input window
                        window_orig = padded_img[
                            x:x + self.window_size[0],
                            y:y + self.window_size[1],
                            z:z + self.window_size[2]
                        ]

                        # Calculate the prediction mask for this window (using argmax of all 3 channels now)
                        stacked_window = np.stack([pred_cls1, pred_cls2, pred_cls3], axis=0)
                        window_mask = np.argmax(stacked_window, axis=0) 
                        # window_mask = (window_mask > 0.5).astype(np.int32) # Thresholding after argmax is usually not needed

                        # Select central axial slice index
                        center_z = window_orig.shape[2] // 2

                        # Extract central axial slices for input and mask
                        window_slice = window_orig[:, :, center_z]
                        window_mask_slice = window_mask[:, :, center_z]

                        # Normalize input slice for visualization
                        slice_min = np.min(window_slice)
                        slice_max = np.max(window_slice)
                        if slice_max > slice_min:
                            window_slice_norm = (window_slice - slice_min) / (slice_max - slice_min)
                        else:
                            window_slice_norm = np.zeros_like(window_slice, dtype=float)

                        # Create RGB overlay (grayscale image with color based on predicted class)
                        overlay = plt.cm.gray(window_slice_norm)[:, :, :3] # Get RGB from grayscale
                        overlay[window_mask_slice == 1] = [0, 1, 0]  # Green for class 1 (originally cls2)
                        overlay[window_mask_slice == 2] = [0, 0, 1]  # Blue for class 2 (originally cls3)
                        # Class 0 (background) remains gray

                        # Save the overlay slice
                        filename = os.path.join(save_dir, f"window_overlay_x{x}_y{y}_z{z}_slice.png")
                        plt.imsave(filename, overlay)
                    except Exception as e:
                        print(f"Error saving window overlay slice ({x},{y},{z}) for {save_prefix}: {e}")
                # --- End window saving logic ---

                # Accumulate probabilities for all 3 channels
                output_cls1[
                    x:x + self.window_size[0],
                    y:y + self.window_size[1],
                    z:z + self.window_size[2]
                ] += pred_cls1
                output_cls2[
                    x:x + self.window_size[0],
                    y:y + self.window_size[1],
                    z:z + self.window_size[2]
                ] += pred_cls2
                output_cls3[
                    x:x + self.window_size[0],
                    y:y + self.window_size[1],
                    z:z + self.window_size[2]
                ] += pred_cls3
                count_map[
                    x:x + self.window_size[0],
                    y:y + self.window_size[1],
                    z:z + self.window_size[2]
                ] += 1

        # Average the accumulated probabilities
        output_cls1 = np.divide(output_cls1, count_map, where=count_map > 0)
        output_cls2 = np.divide(output_cls2, count_map, where=count_map > 0)
        output_cls3 = np.divide(output_cls3, count_map, where=count_map > 0)

        # store an image of the output_cls2
        image_name = f"output_cls2.png"
        plt.imsave(image_name, output_cls2[..., 120])
        print(f"Saved output_cls2 image: {os.path.join(os.getcwd(), image_name)}")


        # Crop back to original image size (before padding)
        output_cls1 = output_cls1[:img.shape[0], :img.shape[1], :img.shape[2]]
        output_cls2 = output_cls2[:img.shape[0], :img.shape[1], :img.shape[2]]
        output_cls3 = output_cls3[:img.shape[0], :img.shape[1], :img.shape[2]]

        # Return the 3 probability maps
        return output_cls1, output_cls2, output_cls3
    
    def _image_size(self, img, orig_img, roi):
        """
        Calculate the image size based on available memory.
        Parameters:
        - img: torch.Tensor, the input image tensor.
        - mask: torch.Tensor, the mask tensor.
        - roi: torch.Tensor, the region of interest tensor.
        - model: torch.nn.Module, the model to be used.
        - device: torch.device, the device on which to run the calculations (e.g., torch.device('cuda:0')).
        
        Returns:
        - int, the calculated batch size.
        """
        # Move model to the specified device

        min_coords = [np.min(np.where(roi == 1)[i]) for i in range(3)]
        max_coords = [np.max(np.where(roi == 1)[i]) for i in range(3)]

        roi_shape = [max_coords[i] - min_coords[i] for i in range(3)]
        
        desired_shape = [64, 192, 256]

        min_coef = desired_shape[0] / img.shape[0]

        des_coef = (desired_shape[1] * desired_shape[2]) / (roi_shape[1] * roi_shape[2])

        des_coef = max(min_coef, des_coef)

        new_image = resize(orig_img, (int(img.shape[0] * des_coef), int(img.shape[1] * des_coef),
                                      int(img.shape[2] * des_coef)), mode='constant', order=0, cval=-1)
        new_roi = resize(roi, (int(roi.shape[0] * des_coef), int(roi.shape[1] * des_coef),
                               int(roi.shape[2] * des_coef)), mode='constant', order=0, cval=-1)
        
        return new_image, new_roi, des_coef

    def _get_roi_bounds(self, roi):
        """
        Get the bounding coordinates of the region of interest (ROI).

        Parameters:
        roi (numpy.ndarray): Region of interest.

        Returns:
        tuple: Minimum and maximum coordinates (min_x, min_y, min_z, max_x, max_y, max_z).
        """
        min_coords = [np.min(np.where(roi == 1)[i]) for i in range(3)]
        max_coords = [np.max(np.where(roi == 1)[i]) for i in range(3)]
        
        if max_coords[0] - min_coords[0] < 64:
            incr = 64 - (max_coords[0] - min_coords[0])
            min_coords[0] -= incr // 2
            max_coords[0] += incr // 2 + incr % 2
            if min_coords[0] < 0:
                const = np.abs(min_coords[0])
                min_coords[0] += const
                max_coords[0] += const

        return (*min_coords, *max_coords)
    
    def _normalize_image(self, img):
        """
        Normalize the image.

        Parameters:
        img (numpy.ndarray): Input image.

        Returns:
        torch.Tensor: Normalized image.
        """
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32)

        img = (img - self.mean) / self.std

        return img
