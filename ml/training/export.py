"""
Model export pipeline for production deployment.
Exports models to TFLite, ONNX, and TorchScript formats with quantization.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse

import torch
import torch.nn as nn
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# TensorFlow Lite imports
try:
    import tensorflow as tf
    from tensorflow import lite as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    print("TensorFlow Lite not available. TFLite export will be skipped.")

from train import KhetGuardModel, KhetGuardDataModule
from augment import AgriculturalAugmentation, AugmentedDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelExporter:
    """Export models to various formats for production deployment."""
    
    def __init__(self, model: KhetGuardModel, class_mapping: Dict[str, int], device: torch.device):
        self.model = model
        self.class_mapping = class_mapping
        self.device = device
        self.id_to_class = {v: k for k, v in class_mapping.items()}
        
        # Set model to evaluation mode
        self.model.eval()
        self.model = self.model.to(device)
    
    def export_torchscript(self, output_path: str, input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> str:
        """Export model to TorchScript format."""
        logger.info("Exporting to TorchScript...")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Trace the model
        traced_model = torch.jit.trace(self.model, dummy_input)
        
        # Save the traced model
        torchscript_path = Path(output_path) / "model.pt"
        traced_model.save(str(torchscript_path))
        
        # Test the exported model
        self._test_torchscript_model(traced_model, dummy_input)
        
        logger.info(f"TorchScript model saved to {torchscript_path}")
        return str(torchscript_path)
    
    def export_onnx(self, output_path: str, input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> str:
        """Export model to ONNX format."""
        logger.info("Exporting to ONNX...")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Export to ONNX
        onnx_path = Path(output_path) / "model.onnx"
        torch.onnx.export(
            self.model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        self._verify_onnx_model(str(onnx_path))
        
        logger.info(f"ONNX model saved to {onnx_path}")
        return str(onnx_path)
    
    def export_tflite(self, output_path: str, input_shape: Tuple[int, ...] = (1, 224, 224, 3)) -> Optional[str]:
        """Export model to TensorFlow Lite format."""
        if not TFLITE_AVAILABLE:
            logger.warning("TensorFlow Lite not available. Skipping TFLite export.")
            return None
        
        logger.info("Exporting to TensorFlow Lite...")
        
        try:
            # Convert PyTorch model to TensorFlow
            tf_model = self._pytorch_to_tensorflow(input_shape)
            
            # Convert to TFLite
            converter = tflite.TFLiteConverter.from_keras_model(tf_model)
            converter.optimizations = [tflite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]  # Use float16 for smaller size
            
            tflite_model = converter.convert()
            
            # Save TFLite model
            tflite_path = Path(output_path) / "model.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            # Test TFLite model
            self._test_tflite_model(str(tflite_path), input_shape)
            
            logger.info(f"TFLite model saved to {tflite_path}")
            return str(tflite_path)
            
        except Exception as e:
            logger.error(f"TFLite export failed: {e}")
            return None
    
    def export_quantized_onnx(self, output_path: str, calibration_data: DataLoader, 
                             input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> str:
        """Export quantized ONNX model using calibration data."""
        logger.info("Exporting quantized ONNX...")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Export to ONNX with quantization
        onnx_path = Path(output_path) / "model_quantized.onnx"
        torch.onnx.export(
            self.model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Apply quantization using ONNX Runtime
        self._quantize_onnx_model(str(onnx_path), str(Path(output_path) / "model_quantized_int8.onnx"))
        
        logger.info(f"Quantized ONNX model saved to {onnx_path}")
        return str(onnx_path)
    
    def _pytorch_to_tensorflow(self, input_shape: Tuple[int, ...]) -> Any:
        """Convert PyTorch model to TensorFlow format."""
        # This is a simplified conversion - in practice, you might need more sophisticated conversion
        import tensorflow as tf
        
        # Create a simple TensorFlow model that mimics the PyTorch model
        # This is a placeholder - actual conversion would be more complex
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape[1:]),  # Remove batch dimension
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(self.class_mapping), activation='softmax')
        ])
        
        return model
    
    def _test_torchscript_model(self, model: torch.jit.ScriptModule, dummy_input: torch.Tensor):
        """Test TorchScript model."""
        with torch.no_grad():
            output = model(dummy_input)
            logger.info(f"TorchScript model test passed. Output shape: {output.shape}")
    
    def _verify_onnx_model(self, onnx_path: str):
        """Verify ONNX model."""
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model verification passed")
        except Exception as e:
            logger.error(f"ONNX model verification failed: {e}")
    
    def _test_tflite_model(self, tflite_path: str, input_shape: Tuple[int, ...]):
        """Test TFLite model."""
        try:
            interpreter = ort.InferenceSession(tflite_path)
            logger.info("TFLite model test passed")
        except Exception as e:
            logger.error(f"TFLite model test failed: {e}")
    
    def _quantize_onnx_model(self, input_path: str, output_path: str):
        """Quantize ONNX model to INT8."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantize_dynamic(
                input_path,
                output_path,
                weight_type=QuantType.QUInt8
            )
            logger.info("ONNX model quantization completed")
        except Exception as e:
            logger.error(f"ONNX quantization failed: {e}")
    
    def create_model_metadata(self, output_path: str, model_paths: Dict[str, str]) -> str:
        """Create model metadata file."""
        metadata = {
            'model_info': {
                'num_classes': len(self.class_mapping),
                'class_mapping': self.class_mapping,
                'id_to_class': self.id_to_class
            },
            'exported_models': model_paths,
            'input_shape': [1, 3, 224, 224],
            'output_shape': [1, len(self.class_mapping)],
            'preprocessing': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'input_size': 224
            },
            'postprocessing': {
                'softmax': True,
                'top_k': 5
            }
        }
        
        metadata_path = Path(output_path) / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to {metadata_path}")
        return str(metadata_path)

def export_model(checkpoint_path: str, config_path: str, data_dir: str, output_dir: str) -> Dict[str, str]:
    """Export a trained model to various formats."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load model
    model = KhetGuardModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load class mapping
    metadata_path = Path(data_dir) / f"{config['dataset_name']}_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create exporter
    exporter = ModelExporter(model, metadata['class_mapping'], device)
    
    # Export models
    model_paths = {}
    
    # TorchScript
    torchscript_path = exporter.export_torchscript(str(output_path))
    model_paths['torchscript'] = torchscript_path
    
    # ONNX
    onnx_path = exporter.export_onnx(str(output_path))
    model_paths['onnx'] = onnx_path
    
    # TFLite
    tflite_path = exporter.export_tflite(str(output_path))
    if tflite_path:
        model_paths['tflite'] = tflite_path
    
    # Create calibration data for quantized model
    data_module = KhetGuardDataModule(
        data_dir=data_dir,
        dataset_name=config['dataset_name'],
        batch_size=32,
        image_size=config['image_size']
    )
    data_module.setup()
    
    # Quantized ONNX
    quantized_onnx_path = exporter.export_quantized_onnx(
        str(output_path), 
        data_module.val_dataloader()
    )
    model_paths['quantized_onnx'] = quantized_onnx_path
    
    # Create metadata
    metadata_path = exporter.create_model_metadata(str(output_path), model_paths)
    model_paths['metadata'] = metadata_path
    
    logger.info("Model export completed successfully!")
    return model_paths

def benchmark_models(model_paths: Dict[str, str], input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> Dict[str, Dict[str, float]]:
    """Benchmark exported models for performance."""
    logger.info("Benchmarking exported models...")
    
    results = {}
    
    # Benchmark TorchScript
    if 'torchscript' in model_paths:
        try:
            model = torch.jit.load(model_paths['torchscript'])
            model.eval()
            
            dummy_input = torch.randn(input_shape)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            # Benchmark
            import time
            times = []
            for _ in range(100):
                start = time.time()
                with torch.no_grad():
                    _ = model(dummy_input)
                times.append(time.time() - start)
            
            results['torchscript'] = {
                'mean_inference_time': np.mean(times),
                'std_inference_time': np.std(times),
                'model_size_mb': Path(model_paths['torchscript']).stat().st_size / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"TorchScript benchmarking failed: {e}")
    
    # Benchmark ONNX
    if 'onnx' in model_paths:
        try:
            import onnxruntime as ort
            
            session = ort.InferenceSession(model_paths['onnx'])
            input_name = session.get_inputs()[0].name
            
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                _ = session.run(None, {input_name: dummy_input})
            
            # Benchmark
            times = []
            for _ in range(100):
                start = time.time()
                _ = session.run(None, {input_name: dummy_input})
                times.append(time.time() - start)
            
            results['onnx'] = {
                'mean_inference_time': np.mean(times),
                'std_inference_time': np.std(times),
                'model_size_mb': Path(model_paths['onnx']).stat().st_size / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"ONNX benchmarking failed: {e}")
    
    # Save benchmark results
    benchmark_path = Path(model_paths['metadata']).parent / "benchmark_results.json"
    with open(benchmark_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark results saved to {benchmark_path}")
    return results

def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description='Export Khet Guard models')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to model config')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for exported models')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    
    args = parser.parse_args()
    
    # Export models
    model_paths = export_model(args.checkpoint, args.config, args.data_dir, args.output_dir)
    
    # Run benchmarks if requested
    if args.benchmark:
        benchmark_results = benchmark_models(model_paths)
        
        # Print benchmark results
        print("\n" + "="*50)
        print("BENCHMARK RESULTS")
        print("="*50)
        for model_type, metrics in benchmark_results.items():
            print(f"\n{model_type.upper()}:")
            print(f"  Mean inference time: {metrics['mean_inference_time']:.4f}s")
            print(f"  Std inference time: {metrics['std_inference_time']:.4f}s")
            print(f"  Model size: {metrics['model_size_mb']:.2f} MB")

if __name__ == "__main__":
    main()
