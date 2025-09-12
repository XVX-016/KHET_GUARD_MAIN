import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_plant_disease_dataset():
    """Prepare plant disease dataset from existing plantvillage dataset"""
    source_dir = Path("plantvillage dataset/grayscale")
    target_dir = Path("datasets/plant_disease")
    
    # Create train/val/test directories
    for split in ["train", "val", "test"]:
        (target_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"Processing class: {class_name}")
        
        # Get all images in this class
        images = list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpg"))
        
        if len(images) < 10:  # Skip classes with too few images
            print(f"Skipping {class_name} - only {len(images)} images")
            continue
            
        # Split into train/val/test (70/20/10)
        train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.33, random_state=42)
        
        # Create class directories in each split
        for split in ["train", "val", "test"]:
            (target_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Copy images to respective splits
        for img_list, split in [(train_imgs, "train"), (val_imgs, "val"), (test_imgs, "test")]:
            for img_path in img_list:
                target_path = target_dir / split / class_name / img_path.name
                shutil.copy2(img_path, target_path)
        
        print(f"  Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

def prepare_cattle_dataset():
    """Prepare cattle dataset from existing Indian_bovine_breeds"""
    source_dir = Path("Indian_bovine_breeds")
    target_dir = Path("datasets/cattle")
    
    # Create train/val/test directories
    for split in ["train", "val", "test"]:
        (target_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Get all breed directories
    breed_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    
    for breed_dir in breed_dirs:
        breed_name = breed_dir.name
        print(f"Processing breed: {breed_name}")
        
        # Get all images in this breed
        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            images.extend(list(breed_dir.glob(ext)))
        
        if len(images) < 5:  # Skip breeds with too few images
            print(f"Skipping {breed_name} - only {len(images)} images")
            continue
            
        # Split into train/val/test (70/20/10)
        train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.33, random_state=42)
        
        # Create breed directories in each split
        for split in ["train", "val", "test"]:
            (target_dir / split / breed_name).mkdir(parents=True, exist_ok=True)
        
        # Copy images to respective splits
        for img_list, split in [(train_imgs, "train"), (val_imgs, "val"), (test_imgs, "test")]:
            for img_path in img_list:
                target_path = target_dir / split / breed_name / img_path.name
                shutil.copy2(img_path, target_path)
        
        print(f"  Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

if __name__ == "__main__":
    print("Preparing plant disease dataset...")
    prepare_plant_disease_dataset()
    
    print("\nPreparing cattle dataset...")
    prepare_cattle_dataset()
    
    print("\nDataset preparation complete!")
