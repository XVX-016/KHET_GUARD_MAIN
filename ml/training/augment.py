"""
Data augmentation pipeline using Albumentations for robust training.
Implements RandAugment, MixUp, CutMix, and custom agricultural augmentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgriculturalAugmentation:
    """Custom augmentation pipeline for agricultural images."""
    
    def __init__(self, 
                 image_size: int = 224,
                 use_randaugment: bool = True,
                 use_mixup: bool = True,
                 use_cutmix: bool = True,
                 mixup_alpha: float = 0.2,
                 cutmix_alpha: float = 1.0):
        
        self.image_size = image_size
        self.use_randaugment = use_randaugment
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        
        # Base augmentations
        self.base_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Training augmentations
        self.train_transform = self._create_train_augmentations()
        
        # Validation augmentations (no augmentation)
        self.val_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _create_train_augmentations(self) -> A.Compose:
        """Create training augmentation pipeline."""
        transforms = [
            # Geometric augmentations
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Transpose(p=0.2)
            ], p=0.7),
            
            A.OneOf([
                A.Rotate(limit=15, p=0.5),
                A.RandomRotate90(p=0.2)
            ], p=0.6),
            
            A.OneOf([
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=0.5
                ),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3)
            ], p=0.5),
            
            # Photometric augmentations
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3)
            ], p=0.7),
            
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3)
            ], p=0.6),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.MotionBlur(blur_limit=3, p=0.3)
            ], p=0.4),
            
            # Agricultural-specific augmentations
            A.OneOf([
                A.RandomShadow(p=0.3),
                A.RandomSunFlare(
                    src_radius=100,
                    flare_roi=(0, 0, 1, 0.5),
                    num_flare_circles_lower=6,
                    num_flare_circles_upper=10,
                    p=0.2
                ),
                A.RandomRain(
                    slant_lower=-10,
                    slant_upper=10,
                    drop_length=20,
                    drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=1,
                    brightness_coefficient=0.7,
                    rain_type="drizzle",
                    p=0.2
                )
            ], p=0.3),
            
            # RandAugment (if enabled)
            A.RandAugment(
                num_ops=2,
                magnitude=9,
                p=0.5 if self.use_randaugment else 0
            ),
            
            # Final transforms
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        return A.Compose(transforms)
    
    def apply_augmentation(self, image: np.ndarray, mask: Optional[np.ndarray] = None, 
                          is_training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply augmentation to image and optional mask."""
        transform = self.train_transform if is_training else self.val_transform
        
        if mask is not None:
            augmented = transform(image=image, mask=mask)
            return augmented['image'], augmented['mask']
        else:
            augmented = transform(image=image)
            return augmented['image'], None

class MixUpCutMix:
    """MixUp and CutMix augmentation implementation."""
    
    def __init__(self, mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
    
    def mixup_data(self, x: torch.Tensor, y: torch.Tensor, alpha: float = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply MixUp augmentation."""
        if alpha is None:
            alpha = self.mixup_alpha
        
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def cutmix_data(self, x: torch.Tensor, y: torch.Tensor, alpha: float = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation."""
        if alpha is None:
            alpha = self.cutmix_alpha
        
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        y_a, y_b = y, y[index]
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        
        return x, y_a, y_b, lam
    
    def _rand_bbox(self, size: Tuple[int, ...], lam: float) -> Tuple[int, int, int, int]:
        """Generate random bounding box for CutMix."""
        W = size[-1]
        H = size[-2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

class AugmentedDataset(Dataset):
    """Dataset with augmentation support."""
    
    def __init__(self, 
                 data: List[Dict],
                 augmentation: AgriculturalAugmentation,
                 is_training: bool = True,
                 use_mixup: bool = False,
                 use_cutmix: bool = False):
        
        self.data = data
        self.augmentation = augmentation
        self.is_training = is_training
        self.use_mixup = use_mixup and is_training
        self.use_cutmix = use_cutmix and is_training
        
        if self.use_mixup or self.use_cutmix:
            self.mixup_cutmix = MixUpCutMix()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Load image
        from PIL import Image
        image = Image.open(item['path']).convert('RGB')
        image = np.array(image)
        
        # Apply augmentation
        image_tensor, _ = self.augmentation.apply_augmentation(
            image, is_training=self.is_training
        )
        
        # Get label
        label = torch.tensor(item['class_id'], dtype=torch.long)
        
        return {
            'image': image_tensor,
            'label': label,
            'class_name': item['class'],
            'path': item['path']
        }
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for MixUp/CutMix."""
        images = torch.stack([item['image'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        
        if self.is_training and (self.use_mixup or self.use_cutmix):
            if self.use_mixup and np.random.random() < 0.5:
                images, labels_a, labels_b, lam = self.mixup_cutmix.mixup_data(images, labels)
                return {
                    'images': images,
                    'labels_a': labels_a,
                    'labels_b': labels_b,
                    'lam': lam,
                    'use_mixup': True
                }
            elif self.use_cutmix and np.random.random() < 0.5:
                images, labels_a, labels_b, lam = self.mixup_cutmix.cutmix_data(images, labels)
                return {
                    'images': images,
                    'labels_a': labels_a,
                    'labels_b': labels_b,
                    'lam': lam,
                    'use_cutmix': True
                }
        
        return {
            'images': images,
            'labels': labels,
            'use_mixup': False,
            'use_cutmix': False
        }

def create_augmentation_pipeline(config: Dict) -> AgriculturalAugmentation:
    """Create augmentation pipeline from config."""
    return AgriculturalAugmentation(
        image_size=config.get('image_size', 224),
        use_randaugment=config.get('use_randaugment', True),
        use_mixup=config.get('use_mixup', True),
        use_cutmix=config.get('use_cutmix', True),
        mixup_alpha=config.get('mixup_alpha', 0.2),
        cutmix_alpha=config.get('cutmix_alpha', 1.0)
    )

if __name__ == "__main__":
    # Test augmentation pipeline
    import matplotlib.pyplot as plt
    
    # Create sample image
    sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test augmentation
    aug = AgriculturalAugmentation()
    
    # Apply training augmentation
    augmented_image, _ = aug.apply_augmentation(sample_image, is_training=True)
    
    print(f"Original shape: {sample_image.shape}")
    print(f"Augmented shape: {augmented_image.shape}")
    print("Augmentation pipeline created successfully!")
