"""
Fusion Model for Crop Disease Detection with Grad-CAM Support
------------------------------------------------------------

Provides:
- CropDiseaseFusionModel class with Grad-CAM capabilities
- create_fusion_model() function
- TensorFlow/Keras implementation
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
from typing import Optional, Tuple


class CropDiseaseFusionModel:
    """Fusion model for crop disease detection with Grad-CAM support."""

    def __init__(self, model: tf.keras.Model, learning_rate: float = 1e-4, gradcam: bool = False):
        self.model = model
        self.learning_rate = learning_rate
        self.gradcam_enabled = gradcam
        self.gradcam_model = None
        
        # Compile the model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        # Create Grad-CAM model if enabled
        if self.gradcam_enabled:
            self._create_gradcam_model()

    def _create_gradcam_model(self):
        """Create a model for Grad-CAM visualization."""
        # Get the last convolutional layer
        conv_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, (layers.Conv2D, layers.Conv2DTranspose)):
                conv_layer = layer
                break
        
        if conv_layer is None:
            # Fallback to a dense layer if no conv layer found
            for layer in reversed(self.model.layers):
                if isinstance(layer, layers.Dense):
                    conv_layer = layer
                    break
        
        if conv_layer:
            self.gradcam_model = models.Model(
                inputs=self.model.inputs,
                outputs=[conv_layer.output, self.model.output]
            )

    def get_model_info(self):
        """Return model summary."""
        return self.model.summary()

    def train(self, train_dataset, val_dataset, epochs=10, callbacks=None, steps_per_epoch=None, validation_steps=None):
        """Train the model."""
        return self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=1
        )

    def evaluate(self, dataset):
        """Evaluate the model."""
        return self.model.evaluate(dataset, return_dict=True)

    def predict(self, inputs):
        """Make predictions."""
        return self.model.predict(inputs)

    def generate_gradcam(self, image: np.ndarray, metadata: np.ndarray, class_index: int = None) -> np.ndarray:
        """
        Generate Grad-CAM visualization.
        
        Args:
            image: Input image (1, H, W, C)
            metadata: Input metadata (1, metadata_dim)
            class_index: Class index for Grad-CAM (if None, uses predicted class)
        
        Returns:
            Grad-CAM heatmap
        """
        if not self.gradcam_enabled or self.gradcam_model is None:
            raise ValueError("Grad-CAM not enabled for this model")
        
        # Prepare inputs
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        if len(metadata.shape) == 1:
            metadata = np.expand_dims(metadata, 0)
        
        inputs = {"image_input": image, "metadata_input": metadata}
        
        # Get predictions
        predictions = self.model.predict(inputs)
        if class_index is None:
            class_index = np.argmax(predictions[0])
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.gradcam_model(inputs)
            loss = predictions[:, class_index]
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        # Resize to input image size
        heatmap = tf.image.resize(heatmap[..., tf.newaxis], [image.shape[1], image.shape[2]])
        heatmap = tf.squeeze(heatmap)
        
        return heatmap.numpy()

    def save_model(self, path: str):
        """Save the model."""
        # Ensure proper file extension
        if not path.endswith('.keras') and not path.endswith('.h5'):
            path = f"{path}.keras"
        
        self.model.save(path)
        
        # Save Grad-CAM model separately if enabled
        if self.gradcam_enabled and self.gradcam_model is not None:
            gradcam_path = f"{path}_gradcam.keras"
            self.gradcam_model.save(gradcam_path)


def create_fusion_model(num_classes: int = 38,
                        model_type: str = "standard",
                        image_size: Tuple[int, int] = (224, 224),
                        metadata_dim: int = 16,
                        dropout_rate: float = 0.3,
                        learning_rate: float = 1e-4,
                        gradcam: bool = False) -> CropDiseaseFusionModel:
    """
    Creates a fusion model for crop disease detection.
    
    Args:
        num_classes: Number of output classes
        model_type: Type of model architecture
        image_size: Input image size (H, W)
        metadata_dim: Dimension of metadata input
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        gradcam: Enable Grad-CAM visualization
    
    Returns:
        CropDiseaseFusionModel instance
    """
    
    # Image branch - CNN backbone
    inputs_img = tf.keras.Input(shape=(image_size[0], image_size[1], 3), name="image_input")
    
    # Feature extraction layers
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs_img)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Metadata branch
    inputs_meta = tf.keras.Input(shape=(metadata_dim,), name="metadata_input")
    y = layers.Dense(64, activation="relu")(inputs_meta)
    y = layers.Dropout(dropout_rate)(y)
    y = layers.Dense(32, activation="relu")(y)
    
    # Fusion
    z = layers.Concatenate()([x, y])
    z = layers.Dropout(dropout_rate)(z)
    z = layers.Dense(128, activation="relu")(z)
    z = layers.Dropout(dropout_rate)(z)
    outputs = layers.Dense(num_classes, activation="softmax")(z)
    
    # Create model
    model = models.Model(inputs=[inputs_img, inputs_meta], outputs=outputs)
    
    return CropDiseaseFusionModel(model=model, learning_rate=learning_rate, gradcam=gradcam)
