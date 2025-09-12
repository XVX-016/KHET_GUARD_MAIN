"""
Data loading and augmentation utilities
--------------------------------------

Provides:
- create_training_pipeline() function that loads NPZ data and creates tf.data.Datasets
- Data augmentation pipelines
- Train/validation split functionality
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def create_training_pipeline(data_path: str,
                             batch_size: int = 32,
                             augmentation_strength: str = "medium") -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Loads NPZ data and creates tf.data.Datasets with augmentation.
    
    Args:
        data_path: Path to NPZ file containing 'images', 'metadata', 'labels'
        batch_size: Batch size for training
        augmentation_strength: 'light', 'medium', or 'heavy'
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info(f"Loading data from {data_path}")
    
    # Load NPZ data
    try:
        data = np.load(data_path)
        images = data["images"]       # shape: (num_samples, H, W, 3)
        metadata = data["metadata"]   # shape: (num_samples, metadata_dim)
        labels = data["labels"]       # shape: (num_samples,)
        
        logger.info(f"Loaded {len(images)} samples")
        logger.info(f"Image shape: {images.shape}")
        logger.info(f"Metadata shape: {metadata.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    # Normalize images to [0, 1] if needed
    if images.max() > 1.0:
        images = images.astype(np.float32) / 255.0
    
    # Train/validation split (80/20)
    split_idx = int(0.8 * len(images))
    
    train_images = images[:split_idx]
    train_metadata = metadata[:split_idx]
    train_labels = labels[:split_idx]
    
    val_images = images[split_idx:]
    val_metadata = metadata[split_idx:]
    val_labels = labels[split_idx:]
    
    logger.info(f"Train samples: {len(train_images)}, Val samples: {len(val_images)}")
    
    # Create datasets
    train_ds = create_dataset(
        train_images, train_metadata, train_labels,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
        augmentation_strength=augmentation_strength
    )
    
    val_ds = create_dataset(
        val_images, val_metadata, val_labels,
        batch_size=batch_size,
        shuffle=False,
        augment=False
    )
    
    return train_ds, val_ds


def create_dataset(images: np.ndarray,
                   metadata: np.ndarray,
                   labels: np.ndarray,
                   batch_size: int,
                   shuffle: bool = True,
                   augment: bool = False,
                   augmentation_strength: str = "medium") -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from numpy arrays.
    
    Args:
        images: Image array (N, H, W, C)
        metadata: Metadata array (N, metadata_dim)
        labels: Labels array (N,)
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        augment: Whether to apply augmentation
        augmentation_strength: Strength of augmentation
    
    Returns:
        tf.data.Dataset
    """
    
    def _generator():
        for i in range(len(images)):
            yield {
                "image_input": images[i],
                "metadata_input": metadata[i]
            }, labels[i]
    
    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        _generator,
        output_signature=(
            {
                "image_input": tf.TensorSpec(shape=images.shape[1:], dtype=tf.float32),
                "metadata_input": tf.TensorSpec(shape=metadata.shape[1:], dtype=tf.float32)
            },
            tf.TensorSpec(shape=(), dtype=tf.int64)
        )
    )
    
    # Apply augmentation if requested
    if augment:
        dataset = dataset.map(
            lambda inputs, labels: (augment_data(inputs, augmentation_strength), labels),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    # Repeat dataset to avoid running out of data
    dataset = dataset.repeat()
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def augment_data(inputs: Dict[str, tf.Tensor], strength: str = "medium") -> Dict[str, tf.Tensor]:
    """
    Apply data augmentation to inputs.
    
    Args:
        inputs: Dictionary containing 'image_input' and 'metadata_input'
        strength: 'light', 'medium', or 'heavy'
    
    Returns:
        Augmented inputs
    """
    image = inputs["image_input"]
    metadata = inputs["metadata_input"]
    
    # Random horizontal flip
    if strength in ["medium", "heavy"]:
        image = tf.image.random_flip_left_right(image)
    
    # Random vertical flip (for some agricultural images)
    if strength == "heavy":
        image = tf.image.random_flip_up_down(image)
    
    # Random rotation
    if strength in ["medium", "heavy"]:
        angle = tf.random.uniform([], -0.1, 0.1)  # ±5.7 degrees
        image = tf.image.rot90(image, k=tf.cast(angle * 18 / np.pi, tf.int32))
    
    # Random brightness
    if strength in ["medium", "heavy"]:
        image = tf.image.random_brightness(image, max_delta=0.1)
    
    # Random contrast
    if strength == "heavy":
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    # Random saturation
    if strength == "heavy":
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    
    # Ensure values are in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return {
        "image_input": image,
        "metadata_input": metadata
    }


def create_sample_data(output_path: str, num_samples: int = 1000, num_classes: int = 38):
    """
    Create sample NPZ data for testing.
    
    Args:
        output_path: Path to save the NPZ file
        num_samples: Number of samples to generate
        num_classes: Number of classes
    """
    logger.info(f"Creating sample data with {num_samples} samples and {num_classes} classes")
    
    # Generate random data
    images = np.random.rand(num_samples, 224, 224, 3).astype(np.float32)
    metadata = np.random.rand(num_samples, 16).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_samples)
    
    # Save to NPZ
    np.savez(
        output_path,
        images=images,
        metadata=metadata,
        labels=labels
    )
    
    logger.info(f"Sample data saved to {output_path}")


if __name__ == "__main__":
    # Create sample datasets for testing
    import os
    os.makedirs("data/processed", exist_ok=True)
    
    create_sample_data("data/processed/plantvillage_color.npz", 1000, 38)
    create_sample_data("data/processed/pest_dataset.npz", 500, 20)
    create_sample_data("data/processed/cattle_dataset.npz", 800, 41)
    
    print("Sample datasets created!")
