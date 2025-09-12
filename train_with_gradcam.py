"""
Unified Multi-Task Training with Grad-CAM & Pesticide Mapping
-------------------------------------------------------------

- Crop Disease, Pest, and Cattle Breed training
- Grad-CAM hooks for explainable AI
- Pesticide recommendations for Crop/Pest
- TensorBoard logging & best-model checkpoints
- Drop-in models for FastAPI API

Author: Khet Guard ML Team
"""

import os
import json
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import tensorflow as tf

from fusion_model import create_fusion_model
from utils.augment import create_training_pipeline
from utils.pesticide_map import load_pesticide_map

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION --- #
CONFIG = {
    "datasets": {
        "crop": "data/processed/plantvillage_color.npz",
        "pest": "data/processed/pest_dataset.npz",
        "cattle": "data/processed/cattle_dataset.npz"
    },
    "models": {
        "crop": {"num_classes": 38, "gradcam": True, "pesticide": True},
        "pest": {"num_classes": 20, "gradcam": True, "pesticide": True},
        "cattle": {"num_classes": 41, "gradcam": True, "pesticide": False}
    },
    "training": {
        "batch_size": 32,
        "epochs": 50,
        "patience": 10,
        "augmentation_strength": "medium",
        "learning_rate": 1e-4
    },
    "output_dir": "model/exports"
}

output_dir = Path(CONFIG["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

# Load pesticide mapping once
PESTICIDE_MAP = load_pesticide_map("ml/recommender/pesticide_map.json")


# --- TRAINING HELPER --- #
def train_model(name: str, dataset_path: str, num_classes: int, gradcam: bool, pesticide: bool) -> dict:
    """
    Train a single model with optional Grad-CAM and pesticide mapping
    Returns dict with model path and Grad-CAM info
    """
    logger.info(f"--- Starting training for {name} ---")

    # Load data
    train_ds, val_ds = create_training_pipeline(
        data_path=dataset_path,
        batch_size=CONFIG["training"]["batch_size"],
        augmentation_strength=CONFIG["training"]["augmentation_strength"]
    )

    # Create model
    model = create_fusion_model(
        num_classes=num_classes,
        image_size=(224, 224),
        metadata_dim=16,
        dropout_rate=0.3,
        learning_rate=CONFIG["training"]["learning_rate"],
        gradcam=gradcam
    )

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = output_dir / f"{name}_best_model_{timestamp}.h5"
    log_dir = output_dir / f"logs_{name}_{timestamp}"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=CONFIG["training"]["patience"],
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir))
    ]

    # Calculate steps per epoch (since we use repeat(), we need to calculate based on original data size)
    # Get original data size from the dataset
    data = np.load(dataset_path)
    original_train_size = int(0.8 * len(data["images"]))
    original_val_size = len(data["images"]) - original_train_size
    
    train_steps = original_train_size // CONFIG["training"]["batch_size"]
    val_steps = original_val_size // CONFIG["training"]["batch_size"]
    
    # Train model
    history = model.train(
        train_ds,
        val_ds,
        epochs=CONFIG["training"]["epochs"],
        callbacks=callbacks,
        steps_per_epoch=train_steps,
        validation_steps=val_steps
    )

    # Save final model
    final_model_path = output_dir / f"{name}_final_model.keras"
    model.save_model(str(final_model_path))
    logger.info(f"{name.capitalize()} model training complete! Saved to {final_model_path}")

    # Attach Grad-CAM and pesticide info if relevant
    extra_info = {}
    if gradcam:
        extra_info["gradcam_enabled"] = True
    if pesticide:
        extra_info["pesticide_map"] = PESTICIDE_MAP

    return {
        "model_path": str(final_model_path),
        "checkpoint_path": str(checkpoint_path),
        "log_dir": str(log_dir),
        **extra_info
    }


# --- MAIN PIPELINE --- #
def main():
    trained_models = {}
    for name, dataset_path in CONFIG["datasets"].items():
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset not found: {dataset_path}. Skipping {name} training.")
            continue

        model_info = CONFIG["models"][name]
        trained_models[name] = train_model(
            name=name,
            dataset_path=dataset_path,
            num_classes=model_info["num_classes"],
            gradcam=model_info.get("gradcam", False),
            pesticide=model_info.get("pesticide", False)
        )

    # Save metadata about trained models
    models_json = output_dir / "trained_models_with_gradcam.json"
    with open(models_json, "w") as f:
        json.dump(trained_models, f, indent=2)

    logger.info(f"All trained models metadata saved to {models_json}")


if __name__ == "__main__":
    main()
