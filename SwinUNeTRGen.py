import numpy as np
import torch
from skimage.transform import resize


class SwinUNETRMaskGen:
    def __init__(self, weights_path, full_size=True, device='cuda:1'):
        """
        Initialize the MaskPreparer with the model weights and device.
        
        Parameters:
        weights_path (str): Path to the model weights.
        device (str): Device to run the model on ('cuda' or 'cpu').
        """
        print(device)
        self.weights = torch.load(weights_path, map_location=device, weights_only=False)
        self.model = self.weights['model_arch']
        self.model.load_state_dict(self.weights['model_state_dict'])
        self.model.deep_supervision = False
        self.model.out1, self.model.out2, self.model.out3, self.model.out4 = None, None, None, None
        self.model.to('cuda:1')
        self.device = device
        self.model.eval()
        self.full_size = True
    
    def __call__(self, img, mask, rois):
        """
        Generate the mask for the given image and region of interest (ROI).
        
        Parameters:
        img (numpy.ndarray): Input image.
        mask (numpy.ndarray): Input mask.
        roi (numpy.ndarray): Region of interest.

        Returns:
        numpy.ndarray: Processed mask.
        numpy.ndarray: Model output.
        """

        orig_img = img.copy()

        if not self.full_size:
            img, roi, des_coef = self._image_size(img, orig_img, rois)
            pre_roi = img.shape

        if type(rois) == tuple:
            pass
        else:
            rois = [rois]

        out_masks = []

        for roi in rois:
            if type(mask) == tuple:
                mask = roi


            min_x, min_y, min_z, max_x, max_y, max_z = self._get_roi_bounds(roi)
            reconstruct_pad = self._calculate_padding(img, min_x, max_x, min_y, max_y, min_z, max_z)
            inpt, mask = self._extract_roi(img, mask, roi, min_x, max_x, min_y, max_y, min_z, max_z)
            inpt, mask, reconstruct_pad = self._adjust_dimensions(inpt, mask, reconstruct_pad)
            inpt = self._normalize_image(inpt)            
            self.model.eval()
            with torch.no_grad():
                out = self.model(inpt.unsqueeze(0).unsqueeze(0).to(self.device))
                if type(out) == tuple:
                    print("tuple")
                    out = out[0]
                out = torch.sigmoid(out)

            out = out[0, 0].detach().cpu().numpy()
            torch.cuda.empty_cache()

            out_mask = self._reconstruct_padded_output(out, reconstruct_pad)

            if not self.full_size:
                out_mask = out_mask[:pre_roi[0], :pre_roi[1], :pre_roi[2]]
                out_mask = resize(out_mask, (orig_img.shape[0], orig_img.shape[1], orig_img.shape[2]),
                                    mode='constant', order=0, cval=-1)
            else:
                out_mask = out_mask[:orig_img.shape[0], :orig_img.shape[1], :orig_img.shape[2]]

            out_mask = self._threshold_output(out_mask, thresh=0.60)
            out_masks.append(out_mask)

        out_mask = np.zeros_like(out_masks[0])
        for mask in out_masks:
            out_mask += mask
        
        return out_mask
        

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
    
    def _calculate_padding(self, img, min_x, max_x, min_y, max_y, min_z, max_z):
        """
        Calculate the padding needed to reconstruct the output to the original image size.

        Parameters:
        img (numpy.ndarray): Input image.
        min_x (int), max_x (int), min_y (int), max_y (int), min_z (int), max_z (int): ROI bounds.

        Returns:
        list: Padding values for reconstruction.
        """
        return [
            min_x, img.shape[0] - max_x,
            min_y, img.shape[1] - max_y,
            min_z, img.shape[2] - max_z
        ]
    
    def _extract_roi(self, img, mask, roi, min_x, max_x, min_y, max_y, min_z, max_z):
        """
        Extract the region of interest (ROI) from the image and mask.

        Parameters:
        img (numpy.ndarray): Input image.
        mask (numpy.ndarray): Input mask.
        roi (numpy.ndarray): Region of interest.
        min_x (int), max_x (int), min_y (int), max_y (int), min_z (int), max_z (int): ROI bounds.

        Returns:
        tuple: Cropped image and mask.
        """
        return (
            img[min_x:max_x, min_y:max_y+32, min_z:max_z+32],
            mask[min_x:max_x, min_y:max_y+32, min_z:max_z+32]
        )
    
    def _adjust_dimensions(self, img, mask, reconstruct_pad):
        """
        Adjust the dimensions of the image and mask to be multiples of the model's requirements.

        Parameters:
        img (numpy.ndarray): Cropped image.
        mask (numpy.ndarray): Cropped mask.
        reconstruct_pad (list): Padding values for reconstruction.

        Returns:
        tuple: Adjusted image, mask, and updated padding values.
        """
        def calculate_removals(size, divisor):
            remove = size % divisor
            add = 0
            if remove % 2 != 0:
                remove //= 2
                add = 1
            else:
                remove //= 2
            return remove, add

        remove_x, add_x = calculate_removals(img.shape[0], 32)
        remove_y, add_y = calculate_removals(img.shape[1], 32)
        remove_z, add_z = calculate_removals(img.shape[2], 32)

        reconstruct_pad[0] += remove_x
        reconstruct_pad[1] += remove_x + add_x
        reconstruct_pad[2] += remove_y
        reconstruct_pad[3] += remove_y + add_y
        reconstruct_pad[4] += remove_z
        reconstruct_pad[5] += remove_z + add_z

        img = img[remove_x:img.shape[0] - (remove_x + add_x),
                  remove_y:img.shape[1] - (remove_y + add_y),
                  remove_z:img.shape[2] - (remove_z + add_z)]

        mask = mask[remove_x:mask.shape[0] - (remove_x + add_x),
                    remove_y:mask.shape[1] - (remove_y + add_y),
                    remove_z:mask.shape[2] - (remove_z + add_z)]
        
        return img, mask, reconstruct_pad
    
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
        
        return (img - torch.mean(img)) / (torch.std(img) + 1e-6)
    
    def _threshold_output(self, out, thresh):
        """
        Apply a threshold to the model output.

        Parameters:
        out (torch.Tensor): Model output.
        thresh (float): Threshold value.

        Returns:
        numpy.ndarray: Thresholded output.
        """
        return (out > thresh)
    
    def _reconstruct_padded_output(self, out_mask, reconstruct_pad):
        """
        Reconstruct the padded output to match the original image size.

        Parameters:
        out_mask (numpy.ndarray): Thresholded output mask.
        reconstruct_pad (list): Padding values for reconstruction.

        Returns:
        numpy.ndarray: Reconstructed output mask.
        """
        for i in range(len(reconstruct_pad)):
            out_mask = np.pad(out_mask, (
                (reconstruct_pad[i] * (i == 0), (reconstruct_pad[i]) * (i == 1)),
                (reconstruct_pad[i] * (i == 2), (reconstruct_pad[i]) * (i == 3)),
                (reconstruct_pad[i] * (i == 4), (reconstruct_pad[i]) * (i == 5))
            ), mode='constant', constant_values=0)
        return out_mask

# Example usage:
# mask_preparer = MaskPreparer('path_to_weights.pth')
# out_mask, model_output = mask_preparer.prepare_mask(img, mask, roi, device='cuda')