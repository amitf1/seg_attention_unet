import numpy as np
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    NormalizeIntensityd,
    RandAffined,
    SpatialPadd
)
from CONFIG import ROI_SIZE

def get_transforms(train=True):
    if train:
        transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], ensure_channel_first=True), 

                NormalizeIntensityd(keys=["image"]),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear", "nearest")),
                SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=ROI_SIZE,
                ),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=ROI_SIZE,
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=ROI_SIZE,
                    rotate_range=(0, 0, np.pi/15),
                    scale_range=(0.1, 0.1, 0.1))
            ]
        )
    else:
        transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                NormalizeIntensityd(keys=["image"]),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear", "nearest")),
                SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=ROI_SIZE,
                )
            ]
        )
    return transforms
