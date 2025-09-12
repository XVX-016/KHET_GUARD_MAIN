"""
Data preparation and validation pipeline for Khet Guard ML models.
Handles PlantVillage, cattle breeds, and pest datasets with validation.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split, StratifiedKFold
import great_expectations as ge
from great_expectations.dataset import PandasDataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validates dataset integrity and quality."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.expectations = []
    
    def validate_image_quality(self, image_path: str) -> Dict[str, bool]:
        """Validate single image quality."""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                return {
                    'valid_format': img.format in ['JPEG', 'PNG', 'JPG'],
                    'min_size': width >= 256 and height >= 256,
                    'not_corrupted': True,
                    'has_content': width * height > 0
                }
        except Exception as e:
            logger.warning(f"Image validation failed for {image_path}: {e}")
            return {
                'valid_format': False,
                'min_size': False,
                'not_corrupted': False,
                'has_content': False
            }
    
    def validate_dataset(self, dataset_name: str) -> Dict[str, any]:
        """Validate entire dataset."""
        dataset_path = self.data_dir / dataset_name
        if not dataset_path.exists():
            raise ValueError(f"Dataset {dataset_name} not found at {dataset_path}")
        
        # Collect all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(dataset_path.rglob(ext))
        
        logger.info(f"Found {len(image_files)} images in {dataset_name}")
        
        # Validate each image
        validation_results = []
        for img_path in image_files:
            result = self.validate_image_quality(str(img_path))
            result['path'] = str(img_path)
            validation_results.append(result)
        
        # Create validation summary
        df = pd.DataFrame(validation_results)
        summary = {
            'total_images': len(df),
            'valid_images': df['valid_format'].sum(),
            'min_size_ok': df['min_size'].sum(),
            'not_corrupted': df['not_corrupted'].sum(),
            'has_content': df['has_content'].sum()
        }
        
        # Filter valid images
        valid_df = df[df['valid_format'] & df['min_size'] & df['not_corrupted'] & df['has_content']]
        
        logger.info(f"Validation summary for {dataset_name}: {summary}")
        return {
            'summary': summary,
            'valid_images': valid_df['path'].tolist(),
            'invalid_images': df[~(df['valid_format'] & df['min_size'] & df['not_corrupted'] & df['has_content'])]['path'].tolist()
        }

class DatasetPreprocessor:
    """Preprocesses datasets for training."""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_plantvillage_dataset(self) -> Dict[str, List[str]]:
        """Process PlantVillage dataset."""
        logger.info("Processing PlantVillage dataset...")
        
        # Map to our existing structure
        plantvillage_path = self.data_dir / "plantvillage dataset"
        grayscale_path = plantvillage_path / "grayscale"
        segmented_path = plantvillage_path / "segmented"
        
        # Create class mapping
        class_mapping = {}
        class_idx = 0
        
        # Process grayscale images
        for class_dir in grayscale_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                class_mapping[class_name] = class_idx
                class_idx += 1
        
        # Create dataset splits
        dataset = {'train': [], 'val': [], 'test': []}
        
        for class_name, class_id in class_mapping.items():
            class_images = list((grayscale_path / class_name).glob('*.JPG'))
            class_images.extend(list((grayscale_path / class_name).glob('*.jpg')))
            
            if len(class_images) == 0:
                continue
            
            # Split images
            train_imgs, temp_imgs = train_test_split(class_images, test_size=0.3, random_state=42)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
            
            # Add to dataset
            for img_path in train_imgs:
                dataset['train'].append({
                    'path': str(img_path),
                    'class': class_name,
                    'class_id': class_id,
                    'dataset': 'plantvillage'
                })
            
            for img_path in val_imgs:
                dataset['val'].append({
                    'path': str(img_path),
                    'class': class_name,
                    'class_id': class_id,
                    'dataset': 'plantvillage'
                })
            
            for img_path in test_imgs:
                dataset['test'].append({
                    'path': str(img_path),
                    'class': class_name,
                    'class_id': class_id,
                    'dataset': 'plantvillage'
                })
        
        # Save metadata
        metadata = {
            'class_mapping': class_mapping,
            'num_classes': len(class_mapping),
            'dataset_stats': {
                split: len(images) for split, images in dataset.items()
            }
        }
        
        with open(self.output_dir / 'plantvillage_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save dataset splits
        for split, images in dataset.items():
            df = pd.DataFrame(images)
            df.to_csv(self.output_dir / f'plantvillage_{split}.csv', index=False)
        
        logger.info(f"PlantVillage dataset processed: {metadata['dataset_stats']}")
        return dataset
    
    def create_cattle_dataset(self) -> Dict[str, List[str]]:
        """Process cattle breeds dataset."""
        logger.info("Processing cattle breeds dataset...")
        
        cattle_path = self.data_dir / "Indian_bovine_breeds"
        class_mapping = {}
        class_idx = 0
        
        # Create class mapping
        for breed_dir in cattle_path.iterdir():
            if breed_dir.is_dir():
                breed_name = breed_dir.name
                class_mapping[breed_name] = class_idx
                class_idx += 1
        
        # Create dataset splits
        dataset = {'train': [], 'val': [], 'test': []}
        
        for breed_name, breed_id in class_mapping.items():
            breed_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                breed_images.extend(list((cattle_path / breed_name).glob(ext)))
            
            if len(breed_images) == 0:
                continue
            
            # Split images
            train_imgs, temp_imgs = train_test_split(breed_images, test_size=0.3, random_state=42)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
            
            # Add to dataset
            for img_path in train_imgs:
                dataset['train'].append({
                    'path': str(img_path),
                    'class': breed_name,
                    'class_id': breed_id,
                    'dataset': 'cattle'
                })
            
            for img_path in val_imgs:
                dataset['val'].append({
                    'path': str(img_path),
                    'class': breed_name,
                    'class_id': breed_id,
                    'dataset': 'cattle'
                })
            
            for img_path in test_imgs:
                dataset['test'].append({
                    'path': str(img_path),
                    'class': breed_name,
                    'class_id': breed_id,
                    'dataset': 'cattle'
                })
        
        # Save metadata
        metadata = {
            'class_mapping': class_mapping,
            'num_classes': len(class_mapping),
            'dataset_stats': {
                split: len(images) for split, images in dataset.items()
            }
        }
        
        with open(self.output_dir / 'cattle_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save dataset splits
        for split, images in dataset.items():
            df = pd.DataFrame(images)
            df.to_csv(self.output_dir / f'cattle_{split}.csv', index=False)
        
        logger.info(f"Cattle dataset processed: {metadata['dataset_stats']}")
        return dataset

def main():
    """Main data preparation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare datasets for training')
    parser.add_argument('--data_dir', type=str, default='../', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./processed_data', help='Output directory')
    parser.add_argument('--validate', action='store_true', help='Run validation only')
    
    args = parser.parse_args()
    
    # Initialize validator and preprocessor
    validator = DataValidator(args.data_dir)
    preprocessor = DatasetPreprocessor(args.data_dir, args.output_dir)
    
    if args.validate:
        # Run validation
        logger.info("Running dataset validation...")
        plantvillage_result = validator.validate_dataset("plantvillage dataset")
        cattle_result = validator.validate_dataset("Indian_bovine_breeds")
        
        logger.info("Validation complete!")
        return
    
    # Process datasets
    logger.info("Starting data preparation...")
    
    # Process PlantVillage
    plantvillage_dataset = preprocessor.create_plantvillage_dataset()
    
    # Process cattle breeds
    cattle_dataset = preprocessor.create_cattle_dataset()
    
    logger.info("Data preparation complete!")

if __name__ == "__main__":
    main()
