from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from skimage.transform import resize
import numpy as np
import torch


class CustomDataLoader(DataLoader):
    def __init__(self, data, batch_size=1, num_threads_in_multithreaded=1, use_numpy=False, val=False, LR=False):
        super().__init__(data, batch_size, num_threads_in_multithreaded)
        self.indices = list(range(len(data)))
        self.data_names = data
        self.use_numpy = use_numpy
        self.val = val
        self.LR = LR

    def __len__(self):
        return len(self.data_names)

    def load_data(self, data_name, seg_name):
        if self.use_numpy:
            data = np.load(data_name).astype(np.float32)
            seg = np.load(seg_name.replace("CTinhaleArtf", "maskArtifact")).astype(np.float32)
            roi = np.load(data_name.replace("CTinhaleArtf", "maskArtifactROI")).astype(np.float32)
        else:
            data = torch.load(data_name)
            seg = torch.load(seg_name)
            if self.LR:
                if np.random.rand() > 0.5:
                    seg_name = seg_name[:-9] + '_L' + seg_name[-9:]
                    roi = torch.load(seg_name.replace('labels', 'roi'))
                else:
                    seg_name = seg_name[:-9] + '_R' + seg_name[-9:]
                    roi = torch.load(seg_name.replace('labels', 'roi'))
            else:
                roi = torch.load(seg_name.replace('labels', 'roi'))
            if type(data) == np.ndarray:
                data = data.astype(np.float32)
                seg = seg.astype(np.float32)
                roi = roi.astype(np.float32)
            else:
                data = data.numpy().astype(np.float32)
                seg = seg.numpy().astype(np.float32)
                roi = roi.numpy().astype(np.float32)
        
        return data, seg, roi

    @staticmethod
    def normalize(data):
        return (data - np.mean(data)) / (np.std(data) + np.finfo(np.float32).eps)
    
    import numpy as np

    @staticmethod
    def amount_padding(data):
        '''
        Add padding to a list of 3D numpy arrays to ensure all data in the batch is the same size.
        
        Parameters:
        - data (list of np.ndarray): A list of numpy arrays with shape (1, x, y, z).
        
        Returns:
        - padded_data (np.ndarray): A numpy array of shape (batch_size, 1, max_x, max_y, max_z) with padded data.
        '''
        # Determine the maximum size along each axis
        max_x = max(arr.shape[2] for arr in data)
        max_y = max(arr.shape[3] for arr in data)
        max_z = max(arr.shape[4] for arr in data)
        
        # Initialize a list to store the padded arrays
        padded_data = []
        
        for arr in data:
            # Calculate the necessary padding for each axis
            pad_x = (max_x - arr.shape[2]) // 2
            pad_y = (max_y - arr.shape[3]) // 2
            pad_z = (max_z - arr.shape[4]) // 2
            
            pad_x1 = max_x - arr.shape[2] - pad_x
            pad_y1 = max_y - arr.shape[3] - pad_y
            pad_z1 = max_z - arr.shape[4] - pad_z

            # Apply padding
            padded_arr = np.pad(arr, ((0, 0), (0, 0), (pad_x, pad_x1), (pad_y, pad_y1), (pad_z, pad_z1)), mode='constant', constant_values=0)
            
            padded_data.append(padded_arr)
        
        # Stack the padded arrays into a single numpy array
        padded_data = np.concatenate(padded_data, axis=0)
        
        return padded_data
    
    @staticmethod
    def generate_complement_mask(mask):
        '''
        Generate the complement of a binary mask.
        
        Parameters:
        - mask (np.ndarray): A binary mask with shape (B, 1, x, y, z).
        
        Returns:
        - complement_mask (np.ndarray): The complement of the input mask.
        '''
        complement_mask = 1 - mask
        
        return complement_mask
    
    def shuffle_indices(self):
        np.random.shuffle(self.indices)

    def down_size(self, inputs, labels, rois):
        if np.random.rand() > 0.9:
                variance = np.random.uniform(.75, .95)
                inputs = resize(inputs, (inputs.shape[0], inputs.shape[1], int(inputs.shape[2]*variance), int(inputs.shape[3]*variance), int(inputs.shape[4]*variance)), mode='constant', cval=-1, order=0)
                labels = resize(labels, (labels.shape[0], labels.shape[1], int(labels.shape[2]*variance), int(labels.shape[3]*variance), int(labels.shape[4]*variance)), mode='constant', cval=-1, order=0)
                rois = resize(rois, (rois.shape[0], rois.shape[1], int(rois.shape[2]*variance), int(rois.shape[3]*variance), int(rois.shape[4]*variance)), mode='constant', cval=-1, order=0)

        return inputs, labels, rois

    def generate_train_batch(self):
        idx = self.get_indices()
        data_names = [self.data_names[i] for i in idx]
        if self.val:
            temp_data, temp_seg, temp_roi = self.load_data(data_names[0], data_names[0].replace('_0000', '').replace('images', 'labels'))
            temp_data = np.expand_dims(temp_data, axis=(0, 1))
            temp_seg = np.expand_dims(temp_seg, axis=(0, 1))
            temp_roi = np.expand_dims(temp_roi, axis=(0, 1))
            return {'data': temp_data, 'seg': temp_seg, 'roi_name': data_names[0].replace('_0000', '').replace('images', 'roi'), 'data_name': data_names[0], 'seg_name': data_names[0].replace('_0000', '').replace('images', 'labels'), 'roi': temp_roi}
        inputs = []
        masks = []
        rois = []

        for data_name in data_names:
            temp_data, temp_seg, temp_roi = self.load_data(data_name, data_name.replace('_0000', '').replace('images', 'labels'))
            temp_data = np.expand_dims(temp_data, axis=(0, 1))
            temp_seg = np.expand_dims(temp_seg, axis=(0, 1))
            temp_roi = np.expand_dims(temp_roi, axis=(0, 1))
            temp_data, temp_seg, temp_roi = self.down_size(temp_data, temp_seg, temp_roi)
            inputs.append(temp_data)
            masks.append(temp_seg)
            rois.append(temp_roi)

        if self.batch_size > 1:
            inputs = self.amount_padding(inputs)
            masks = self.amount_padding(masks)
            rois = self.amount_padding(rois)
        else:
            inputs = inputs[0]
            masks = masks[0]
            rois = rois[0]

        complement_masks = self.generate_complement_mask(masks)
        
        combined_data = np.concatenate((inputs, rois), axis=1).astype(np.float32)
        combined_seg = np.concatenate((complement_masks, masks), axis=1).astype(np.float32)
        roi_name = data_name.replace('_0000', '').replace('images', 'roi')
        roi = rois

        return {'data': combined_data, 'seg': combined_seg, 'roi_name': roi_name, 'data_name': data_name, 'seg_name': data_name.replace('_0000', '').replace('images', 'labels'), 'roi': roi}
