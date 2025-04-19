import numpy as np
import random
import warnings
import torch
import torch.nn.functional as F

def calculate_roi(data, mask, roi, patch_size=None, val=False, verbose=False):
    """Crops data and mask based on ROI bounding box.

    Args:
        data (np.ndarray): Input data (B, C, D, H, W) or (C, D, H, W).
        mask (np.ndarray): Input mask (B, 1, D, H, W) or (1, D, H, W).
        roi (np.ndarray): ROI mask (B, 1, D, H, W) or (1, D, H, W).
        patch_size (tuple, optional): Target patch size for padding adjustments (D, H, W). Defaults to None.
        val (bool): If True, apply validation padding logic (multiple of 32).
        verbose (bool): If True, print verbose messages.

    Returns:
        tuple[np.ndarray, np.ndarray]: Cropped (and potentially padded) data and mask.
    """
    # Handle potential batch dimension
    has_batch_dim = data.ndim == 5
    if not has_batch_dim:
        data = data[np.newaxis, ...]
        mask = mask[np.newaxis, ...]
        roi = roi[np.newaxis, ...]

    cropped_data_list = []
    cropped_mask_list = []

    for b in range(data.shape[0]):
        data_sample = data[b]
        mask_sample = mask[b]
        roi_sample = roi[b]

        # Check if ROI mask is valid and contains foreground
        if roi_sample is None or np.sum(roi_sample) == 0:
            if verbose: print(f"ROI mask for sample {b} is empty or invalid, skipping ROI calculation.")
            cropped_data_list.append(data_sample)
            cropped_mask_list.append(mask_sample)
            continue

        coords = np.where(roi_sample == 1)

        # Ensure coordinates were found
        if len(coords[0]) == 0:
             if verbose: print(f"ROI mask provided for sample {b} but no ROI voxels found, skipping ROI calculation.")
             cropped_data_list.append(data_sample)
             cropped_mask_list.append(mask_sample)
             continue

        try:
            # Calculate bounding box (indices are C, D, H, W)
            min_coords = [np.min(coords[i]) for i in range(len(coords)) if i > 0] # Skip Channel dim
            max_coords = [np.max(coords[i]) for i in range(len(coords)) if i > 0] # Skip Channel dim
            # Ensure max is exclusive
            max_coords = [c + 1 for c in max_coords]

            # Ensure min/max calculation worked (list should have 3 elements)
            if len(min_coords) != 3 or len(max_coords) != 3:
                 print(f"Error: Could not determine 3D min/max coordinates from ROI shape {roi_sample.shape}. Coords found: {coords}")
                 cropped_data_list.append(data_sample)
                 cropped_mask_list.append(mask_sample)
                 continue

            # Add padding/adjustments based on patch size or multiples of 32 (SwinUNETR requirement)
            if val:
                if patch_size is None:
                    print("Warning: `patch_size` must be provided for validation ROI padding. Skipping padding.")
                    current_dims = [max_c - min_c for min_c, max_c in zip(min_coords, max_coords)]
                    padded_dims = current_dims # No padding if patch_size is missing
                else:
                    current_dims = [max_c - min_c for min_c, max_c in zip(min_coords, max_coords)]
                    # Ensure ROI size is at least patch_size, then pad to multiple of 32
                    target_dims = [max(ps, cd) for ps, cd in zip(patch_size, current_dims)]
                    padded_dims = [((d + 31) // 32) * 32 for d in target_dims] # Round up to nearest multiple of 32

                pad_needed = [pd - cd for pd, cd in zip(padded_dims, current_dims)]

                # Distribute padding
                pad_before = [p // 2 for p in pad_needed]
                pad_after = [p - pb for p, pb in zip(pad_needed, pad_before)]

                final_min_coords = [max(0, mc - pb) for mc, pb in zip(min_coords, pad_before)]
                # Adjust max coords based on padding added before, ensuring we don't exceed original image bounds initially
                temp_max_coords = [min(data_sample.shape[i+1], mc + pa) for i, (mc, pa) in enumerate(zip(max_coords, pad_after))]

                # Final max includes initial range + padding after, capped by shape
                final_max_coords = [min(data_sample.shape[i+1], fmin + pdim) 
                                    for i, (fmin, pdim) in enumerate(zip(final_min_coords, padded_dims))]

                # Ensure the final crop size attempts to match padded_dims before boundary clipping
                # Crop first based on potentially expanded box
                data_cropped = data_sample[:, final_min_coords[0]:final_max_coords[0], final_min_coords[1]:final_max_coords[1], final_min_coords[2]:final_max_coords[2]]
                mask_cropped = mask_sample[:, final_min_coords[0]:final_max_coords[0], final_min_coords[1]:final_max_coords[1], final_min_coords[2]:final_max_coords[2]]

                # Calculate actual padding needed AFTER cropping
                final_dims = data_cropped.shape[1:]
                final_padding_needed = [max(0, pd - fd) for pd, fd in zip(padded_dims, final_dims)]
                final_pad_before = [p // 2 for p in final_padding_needed]
                final_pad_after = [p - pb for p, pb in zip(final_padding_needed, final_pad_before)]
                final_padding = list(zip(final_pad_before, final_pad_after))

                # Pad if necessary
                if any(p[0] > 0 or p[1] > 0 for p in final_padding):
                    # Pad data (using edge or reflect might be better than constant 0)
                    data_padded = np.pad(data_cropped, ((0,0), *final_padding), mode='constant', constant_values=np.min(data_cropped))
                    # Pad mask with 0
                    mask_padded = np.pad(mask_cropped, ((0,0), *final_padding), mode='constant', constant_values=0)
                else:
                    data_padded = data_cropped
                    mask_padded = mask_cropped

                # Verify final shape
                if data_padded.shape[1:] != tuple(padded_dims):
                     print(f"Warning: Final validation ROI shape {data_padded.shape[1:]} does not match target padded shape {tuple(padded_dims)}. Check logic.")

                cropped_data_list.append(data_padded)
                cropped_mask_list.append(mask_padded)

            else: # Training ROI calculation (simpler crop, patching handles size)
                 margin = 15
                 final_min_coords = [max(0, c - margin) for c in min_coords]
                 final_max_coords = [min(data_sample.shape[i+1], c + margin) for i, c in enumerate(max_coords)]

                 data_cropped = data_sample[:, final_min_coords[0]:final_max_coords[0], final_min_coords[1]:final_max_coords[1], final_min_coords[2]:final_max_coords[2]]
                 mask_cropped = mask_sample[:, final_min_coords[0]:final_max_coords[0], final_min_coords[1]:final_max_coords[1], final_min_coords[2]:final_max_coords[2]]
                 cropped_data_list.append(data_cropped)
                 cropped_mask_list.append(mask_cropped)

        except Exception as e:
            print(f"Error in calculating ROI for sample {b}, shape {data_sample.shape} with ROI sum {np.sum(roi_sample)}: {e}")
            # Return original data if error occurs
            cropped_data_list.append(data_sample)
            cropped_mask_list.append(mask_sample)

    # Combine results back into a batch
    final_data = np.stack(cropped_data_list, axis=0)
    final_mask = np.stack(cropped_mask_list, axis=0)

    if not has_batch_dim:
        final_data = final_data.squeeze(0)
        final_mask = final_mask.squeeze(0)

    return final_data, final_mask

def generate_patch_size(data_shape, default_patch_size=(96, 96, 96)):
    """
    Generates a patch size for training, ensuring it's compatible with SwinUNETR 
    (multiple of 32) and fits within the data_shape.
    
    Args:
        data_shape (tuple): The shape of the input data (after ROI selection) [D, H, W].
        default_patch_size (tuple): The desired patch size.
    
    Returns:
        tuple: The patch size (pd, ph, pw) to be used for sampling.
    """
    # Ensure the patch size is not larger than the data dimensions
    adjusted_patch_size = [min(ds, ps) for ds, ps in zip(data_shape, default_patch_size)]

    # Ensure patch dimensions are multiples of 32 (SwinUNETR constraint)
    # Adjust *down* to the nearest multiple of 32 if necessary
    final_patch_size = [(ps // 32) * 32 for ps in adjusted_patch_size]
    
    # Handle cases where adjusted size becomes 0
    final_patch_size = [max(32, ps) for ps in final_patch_size] 

    return tuple(final_patch_size)

def sample_foreground_coordinate(seg_mask, foreground_classes):
    """Samples a random coordinate centered on a foreground voxel.
    
    Args:
        seg_mask (np.ndarray): Segmentation mask (D, H, W).
        foreground_classes (list or None): List of foreground class labels.
    
    Returns:
        tuple or None: Coordinates (d, h, w) or None if no foreground found.
    """
    if foreground_classes is None:
        foreground_mask = seg_mask > 0
    else:
        foreground_mask = np.isin(seg_mask, foreground_classes)
        
    foreground_coords = np.argwhere(foreground_mask)
    if len(foreground_coords) == 0:
        return None
        
    center_idx = np.random.randint(len(foreground_coords))
    center_coords = foreground_coords[center_idx] # Shape (3,) [d, h, w]
    return tuple(center_coords)

def sample_random_coordinate(data_shape):
    """Samples a random coordinate within the data shape.
    
    Args:
        data_shape (tuple): Shape of the data (D, H, W).
    
    Returns:
        tuple: Coordinates (d, h, w).
    """
    coords = [np.random.randint(0, ds) for ds in data_shape]
    return tuple(coords)

def sample_patch_foreground_based(data_dict, patch_size, num_samples, foreground_classes=None,
                                  foreground_prob=0.5, allow_empty=False):
    """Samples patches, prioritizing foreground regions based on foreground_prob.
    
    Args:
        data_dict (dict): Dictionary containing 'data' (B, C, D, H, W) 
                          and 'seg' (B, 1, D, H, W) numpy arrays.
                          Optionally contains 'roi' (B, 1, D, H, W).
        patch_size (tuple): The size of the patch to sample (D, H, W).
        num_samples (int): The number of patches to sample (usually batch size).
        foreground_classes (list, optional): List of class indices considered foreground.
                                           Defaults to None (all non-zero are foreground).
        foreground_prob (float): Probability of sampling a foreground patch.
        allow_empty (bool): If True, allows sampling random patches even if 
                           foreground sampling fails.
                           
    Returns:
        dict: Dictionary containing sampled 'data' (num_samples, C, pD, pH, pW),
              'seg' (num_samples, 1, pD, pH, pW), and optional 'roi'.
    """
    data = data_dict['data']
    seg = data_dict['seg']
    roi = data_dict.get('roi') # Optional ROI

    batch_size_actual = data.shape[0]
    if batch_size_actual == 0:
        # Handle empty input batch
        img_channels = 1 # Assume 1 channel if empty
        seg_channels = 1
        roi_channels = roi.shape[1] if roi is not None else 0
        data_patches = np.zeros((num_samples, img_channels, *patch_size), dtype=data_dict.get('data', np.array([])).dtype)
        seg_patches = np.zeros((num_samples, seg_channels, *patch_size), dtype=data_dict.get('seg', np.array([])).dtype)
        roi_patches = np.zeros((num_samples, roi_channels, *patch_size), dtype=data_dict.get('roi', np.array([])).dtype) if roi is not None else None
        return {'data': data_patches, 'seg': seg_patches, 'roi': roi_patches}
        
    img_channels = data.shape[1]
    seg_channels = seg.shape[1] # Should be 1
    roi_channels = roi.shape[1] if roi is not None else 0

    data_patches = np.zeros((num_samples, img_channels, *patch_size), dtype=data.dtype)
    seg_patches = np.zeros((num_samples, seg_channels, *patch_size), dtype=seg.dtype)
    roi_patches = np.zeros((num_samples, roi_channels, *patch_size), dtype=roi.dtype) if roi is not None else None

    for i in range(num_samples):
        # Select sample from the original batch, cycling if num_samples > batch_size_actual
        b = i % batch_size_actual 
        data_orig = data[b] # (C, D, H, W)
        seg_orig = seg[b, 0] # (D, H, W)
        roi_orig = roi[b] if roi is not None else None # (1, D, H, W) or None
        data_shape = data_orig.shape[1:] # (D, H, W)
        
        # Check if data shape is valid
        if any(ds <= 0 for ds in data_shape):
             print(f"Warning: Invalid data shape {data_shape} for sample {b}. Skipping patch {i}.")
             # Fill with zeros or some other placeholder?
             continue
        
        coords = None
        attempts = 0
        max_attempts = 100 # Increased attempts

        # Decide whether to sample foreground for this patch
        sample_foreground = random.random() < foreground_prob

        while coords is None and attempts < max_attempts:
            attempts += 1
            if sample_foreground:
                coords = sample_foreground_coordinate(seg_orig, foreground_classes)
                if coords is None and not allow_empty:
                     # Failed to find foreground, try again if attempts remain
                     sample_foreground = False # Fallback to random sampling for next attempt
                     continue 
                elif coords is None and allow_empty:
                     # Failed to find foreground, but allow_empty is True, force random sampling
                     sample_foreground = False 

            if not sample_foreground: 
                coords = sample_random_coordinate(data_shape)
            
            # Validate coordinates against data shape
            if coords is None or not all(0 <= c < ds for c, ds in zip(coords, data_shape)):
                 print(f"Warning: Generated invalid coordinates {coords} for shape {data_shape}. Retrying attempt {attempts}/{max_attempts}.")
                 coords = None # Force retry if invalid coords generated
                 sample_foreground = random.random() < foreground_prob # Re-decide sampling strategy
                 continue
                 
        if coords is None:
            print(f"ERROR: Could not determine valid sampling coordinates for shape {data_shape} after {max_attempts} attempts. Skipping sample {i}.")
            # Fill with zeros? Or raise error?
            continue

        # --- Cropping and Padding Logic --- 
        # Calculate start/end coordinates based on center coordinate
        starts = [c - ps // 2 for c, ps in zip(coords, patch_size)]
        ends = [st + ps for st, ps in zip(starts, patch_size)]

        # Determine necessary padding BEFORE cropping
        pad_before = [max(0, -st) for st in starts]
        pad_after = [max(0, end - ds) for end, ds in zip(ends, data_shape)]
        padding = list(zip(pad_before, pad_after))

        # Determine slice coordinates within the original data
        crop_starts = [st + pb for st, pb in zip(starts, pad_before)]
        crop_ends = [end - pa for end, pa in zip(ends, pad_after)]
        
        # Ensure crop coordinates are valid
        crop_starts = [max(0, cs) for cs in crop_starts]
        crop_ends = [min(ds, ce) for ds, ce in zip(data_shape, crop_ends)]
        crop_starts = [min(cs, ce) for cs, ce in zip(crop_starts, crop_ends)] # Handle edge case where start > end

        # Crop the data, seg, and roi
        try:
            data_slice = data_orig[:, crop_starts[0]:crop_ends[0], crop_starts[1]:crop_ends[1], crop_starts[2]:crop_ends[2]]
            seg_slice = seg_orig[crop_starts[0]:crop_ends[0], crop_starts[1]:crop_ends[1], crop_starts[2]:crop_ends[2]]
            # Add channel dim back for seg_slice before padding
            seg_slice_with_channel = seg_slice[np.newaxis, ...] 
            
            roi_slice_with_channel = None
            if roi_orig is not None:
                roi_slice = roi_orig[:, crop_starts[0]:crop_ends[0], crop_starts[1]:crop_ends[1], crop_starts[2]:crop_ends[2]]
                roi_slice_with_channel = roi_slice
        except Exception as crop_e:
            print(f"ERROR cropping sample {i} with coords={coords}, starts={starts}, ends={ends}, crop_starts={crop_starts}, crop_ends={crop_ends}, shape={data_shape}: {crop_e}")
            continue # Skip this sample
            
        # Pad if necessary to reach the target patch_size
        if any(p > 0 for p_pair in padding for p in p_pair):
            try:
                # Pad data (consider mode='edge' or 'reflect'?) Use minimum value for now.
                min_val = np.min(data_slice) if data_slice.size > 0 else 0
                data_patches[i] = np.pad(data_slice, ((0, 0), *padding), mode='constant', constant_values=min_val) 
                # Pad segmentation with 0 (background)
                seg_patches[i] = np.pad(seg_slice_with_channel, ((0, 0), *padding), mode='constant', constant_values=0)
                # Pad ROI if it exists
                if roi_patches is not None and roi_slice_with_channel is not None:
                    roi_patches[i] = np.pad(roi_slice_with_channel, ((0, 0), *padding), mode='constant', constant_values=0)
            except Exception as pad_e:
                print(f"ERROR padding sample {i} with padding={padding}, data_slice_shape={data_slice.shape}: {pad_e}")
                continue # Skip this sample
        else:
            # No padding needed, directly assign the cropped slice
            data_patches[i] = data_slice
            seg_patches[i] = seg_slice_with_channel # Already has channel dim
            if roi_patches is not None and roi_slice_with_channel is not None:
                roi_patches[i] = roi_slice_with_channel
                
        # --- Verification --- 
        if data_patches[i].shape[1:] != patch_size:
            print(f"ERROR: Final data patch shape {data_patches[i].shape[1:]} mismatch target {patch_size} for sample {i}")
        if seg_patches[i].shape[1:] != patch_size:
            print(f"ERROR: Final seg patch shape {seg_patches[i].shape[1:]} mismatch target {patch_size} for sample {i}")
        if roi_patches is not None and roi_patches[i].shape[1:] != patch_size:
            print(f"ERROR: Final roi patch shape {roi_patches[i].shape[1:]} mismatch target {patch_size} for sample {i}")
            

    return {'data': data_patches, 'seg': seg_patches, 'roi': roi_patches}

def extract_training_patches(data_dict, patch_size, batch_size, foreground_prob, foreground_classes=[1, 2]):
    """Extracts patches for training, prioritizing foreground sampling.
    Wrapper around sample_patch_foreground_based.
    
    Args:
        data_dict (dict): Batch data dictionary from data loader.
        patch_size (tuple): Desired patch size (D, H, W).
        batch_size (int): Number of patches to extract (batch size).
        foreground_prob (float): Probability to sample foreground patches.
        foreground_classes (list): List of foreground class labels.
        
    Returns:
        dict: Dictionary containing sampled patches.
    """
    # Call the core sampling function
    data_dict_sampled = sample_patch_foreground_based(
        data_dict, 
        patch_size=patch_size,
        num_samples=batch_size, 
        foreground_classes=foreground_classes, 
        foreground_prob=foreground_prob,
        allow_empty=True # Allow random sampling if foreground fails
    )
    return data_dict_sampled

# --- Deprecated Functions (Kept for reference, maybe remove later) ---

def sample_foreground_patch(*args, **kwargs):
    """
    DEPRECATED - Was: Sample a patch centered on a foreground voxel.
    Should not be used. Calls sample_random_patch as fallback.
    """
    warnings.warn("sample_foreground_patch is deprecated and should not be used.", DeprecationWarning)
    # Fallback to random sampling if called accidentally
    return sample_random_patch(*args, **kwargs)

def sample_random_patch(data, seg, roi, patch_size=(96, 96, 96)):
    """Sample random patches from the data batch (Simpler version, less robust padding).
    DEPRECATED in favor of sample_patch_foreground_based with foreground_prob=0.
    """
    warnings.warn("sample_random_patch is deprecated. Use sample_patch_foreground_based with foreground_prob=0 instead.", DeprecationWarning)
    batch_size = data.shape[0]
    img_channels = data.shape[1]
    seg_channels = seg.shape[1] # Should be 1 for integer masks
    roi_channels = roi.shape[1]
    
    # Initialize arrays to store patches
    data_patches = np.zeros((batch_size, img_channels, *patch_size), dtype=data.dtype)
    # Ensure segmentation patches are integer type if target is integer
    seg_patches = np.zeros((batch_size, seg_channels, *patch_size), dtype=seg.dtype) 
    roi_patches = np.zeros((batch_size, roi_channels, *patch_size), dtype=roi.dtype)
    
    for b in range(batch_size):
        data_shape = data.shape[2:] # D, H, W
        
        # Check if the data is smaller than the patch size in any dimension
        if any(ds < ps for ds, ps in zip(data_shape, patch_size)):
            # Calculate necessary padding
            padding = []
            for i in range(3):
                pad_needed = max(0, patch_size[i] - data_shape[i])
                # Distribute padding (mostly) evenly before and after
                pad_before = pad_needed // 2
                pad_after = pad_needed - pad_before
                padding.append((pad_before, pad_after))
            
            # Pad the image, segmentation, and ROI
            # Use mode='constant' with default constant_values=0
            padded_data = np.pad(data[b], ((0,0), *padding), mode='constant', constant_values=np.min(data[b])) # Pad with min value
            padded_seg = np.pad(seg[b], ((0,0), *padding), mode='constant', constant_values=0)
            padded_roi = np.pad(roi[b], ((0,0), *padding), mode='constant', constant_values=0)

            # Now data is large enough, sample a random patch from the padded data
            padded_shape = padded_data.shape[1:] # C, D, H, W -> D, H, W
            if any(ps > pds for ps, pds in zip(patch_size, padded_shape)):
                 print(f"Warning: Padded shape {padded_shape} still smaller than patch size {patch_size}. Cropping entire padded volume.")
                 # This case shouldn't happen with the padding logic, but as a safeguard:
                 data_patches[b] = padded_data[:, :patch_size[0], :patch_size[1], :patch_size[2]]
                 seg_patches[b] = padded_seg[:, :patch_size[0], :patch_size[1], :patch_size[2]]
                 roi_patches[b] = padded_roi[:, :patch_size[0], :patch_size[1], :patch_size[2]]
                 continue
                 
            d_start = np.random.randint(0, padded_shape[0] - patch_size[0] + 1)
            h_start = np.random.randint(0, padded_shape[1] - patch_size[1] + 1)
            w_start = np.random.randint(0, padded_shape[2] - patch_size[2] + 1)

            data_patches[b] = padded_data[:, d_start:d_start+patch_size[0], h_start:h_start+patch_size[1], w_start:w_start+patch_size[2]]
            seg_patches[b] = padded_seg[:, d_start:d_start+patch_size[0], h_start:h_start+patch_size[1], w_start:w_start+patch_size[2]]
            roi_patches[b] = padded_roi[:, d_start:d_start+patch_size[0], h_start:h_start+patch_size[1], w_start:w_start+patch_size[2]]

        else:
            # Data is large enough, sample directly
            d_start = np.random.randint(0, data_shape[0] - patch_size[0] + 1)
            h_start = np.random.randint(0, data_shape[1] - patch_size[1] + 1)
            w_start = np.random.randint(0, data_shape[2] - patch_size[2] + 1)
            
            data_patches[b] = data[b, :, d_start:d_start+patch_size[0], 
                                 h_start:h_start+patch_size[1], 
                                 w_start:w_start+patch_size[2]]
            seg_patches[b] = seg[b, :, d_start:d_start+patch_size[0], 
                               h_start:h_start+patch_size[1], 
                               w_start:w_start+patch_size[2]]
            roi_patches[b] = roi[b, :, d_start:d_start+patch_size[0], 
                               h_start:h_start+patch_size[1], 
                               w_start:w_start+patch_size[2]]
            
    return data_patches, seg_patches, roi_patches 