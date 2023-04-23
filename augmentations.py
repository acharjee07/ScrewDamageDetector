
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.3),
    A.ShiftScaleRotate(p=0.3),
    A.RandomResizedCrop(height=1024, width=1024, scale=(0.8, 1.0), ratio=(3.0/4.0, 4.0/3.0), p=0.3),
    A.CoarseDropout(max_holes=8, max_height=64, max_width=64, p=0.3),
    ToTensorV2(),


    
 
])

valid_transforms = A.Compose([
    A.Resize(256, 256),

    ToTensorV2(),
 
])