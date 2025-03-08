from batchgenerators.transforms.spatial_transforms import SpatialTransform_2
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.abstract_transforms import Compose
import numpy as np

def get_augmentations():
    # Spatial transformations
    spatial_transform = SpatialTransform_2(
        patch_size=None,
        patch_center_dist_from_border=None,
        do_elastic_deform=False,
        do_rotation=True,
        angle_x=(-10 / 360. * 2 * np.pi, 10 / 360. * 2 * np.pi),
        angle_y=(-10 / 360. * 2 * np.pi, 10 / 360. * 2 * np.pi),
        angle_z=(-10 / 360. * 2 * np.pi, 10 / 360. * 2 * np.pi),
        scale=(0.75, 1.25),
        border_mode_data='constant',
        border_cval_data=0,
        border_mode_seg='constant',
        border_cval_seg=0,
        order_seg=0,
        order_data=3,
        do_scale=True,
        random_crop=False,
        p_rot_per_sample=0.2,
        p_scale_per_sample=0.2,
        independent_scale_for_each_axis=False
    )

    # Note: per-channel transformations have not been applied due to passing of roi mask

    # Noise transformations
    gaussian_noise = GaussianNoiseTransform((0, 0.1), p_per_sample=0.15)
    gaussian_blur = GaussianBlurTransform((0.5, 1.0), different_sigma_per_channel=False, p_per_sample=0.2, p_per_channel=0.5)

    # Color transformations
    brightness_multiplicative = BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15)
    contrast_augmentation = ContrastAugmentationTransform(p_per_sample=0.15)
    gamma_transform_retain_stats = GammaTransform((0.7, 1.5), invert_image=True, per_channel=False, retain_stats=True, p_per_sample=0.1)
    gamma_transform = GammaTransform((0.7, 1.5), invert_image=False, per_channel=False, retain_stats=True, p_per_sample=0.3)

    # Resample transformations
    low_resolution_simulation = SimulateLowResolutionTransform(
        zoom_range=(0.5, 1.0),
        per_channel=False,
        p_per_channel=0.5,
        order_downsample=0,
        order_upsample=3,
        p_per_sample=0.25,
        channels=[0]
    )

    # Compose all transformations
    all_transforms = [
        spatial_transform,
        gaussian_noise,
        gaussian_blur,
        brightness_multiplicative,
        contrast_augmentation,
        low_resolution_simulation,
        gamma_transform_retain_stats,
        gamma_transform
    ]
    
    transforms = Compose(all_transforms)
    return transforms
