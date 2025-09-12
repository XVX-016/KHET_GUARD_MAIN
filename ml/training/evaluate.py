"""
Model evaluation and calibration pipeline.
Implements calibration, uncertainty estimation, and comprehensive evaluation metrics.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from train import KhetGuardModel, KhetGuardDataModule
from augment import AgriculturalAugmentation, AugmentedDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCalibrator:
    """Temperature scaling calibration for model outputs."""
    
    def __init__(self, model: KhetGuardModel, device: torch.device):
        self.model = model
        self.device = device
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.model.eval()
    
    def calibrate(self, val_loader: DataLoader) -> float:
        """Calibrate model using validation set."""
        self.model.eval()
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(images)
                logits_list.append(logits)
                labels_list.append(labels)
        
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        # Temperature scaling
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        
        return self.temperature.item()
    
    def predict_calibrated(self, images: torch.Tensor) -> torch.Tensor:
        """Get calibrated predictions."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(images)
            calibrated_probs = F.softmax(logits / self.temperature, dim=1)
        return calibrated_probs

class UncertaintyEstimator:
    """Uncertainty estimation using MC Dropout and ensemble methods."""
    
    def __init__(self, model: KhetGuardModel, device: torch.device, n_samples: int = 10):
        self.model = model
        self.device = device
        self.n_samples = n_samples
        self.model.eval()
    
    def predict_with_uncertainty(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict with epistemic and aleatoric uncertainty."""
        self.model.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(images)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        
        predictions = torch.stack(predictions)  # [n_samples, batch_size, num_classes]
        
        # Epistemic uncertainty (model uncertainty)
        mean_pred = torch.mean(predictions, dim=0)
        epistemic_uncertainty = torch.std(predictions, dim=0)
        
        # Aleatoric uncertainty (data uncertainty)
        aleatoric_uncertainty = torch.mean(predictions * (1 - predictions), dim=0)
        
        return mean_pred, epistemic_uncertainty, aleatoric_uncertainty

class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, model: KhetGuardModel, class_mapping: Dict[str, int], device: torch.device):
        self.model = model
        self.class_mapping = class_mapping
        self.device = device
        self.id_to_class = {v: k for k, v in class_mapping.items()}
        
        # Initialize calibrator and uncertainty estimator
        self.calibrator = ModelCalibrator(model, device)
        self.uncertainty_estimator = UncertaintyEstimator(model, device)
    
    def evaluate_model(self, test_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        logger.info("Starting model evaluation...")
        
        # Calibrate model
        logger.info("Calibrating model...")
        temperature = self.calibrator.calibrate(val_loader)
        logger.info(f"Calibration temperature: {temperature:.4f}")
        
        # Get predictions
        logger.info("Getting predictions...")
        predictions, labels, probabilities, uncertainties = self._get_predictions(test_loader)
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = self._calculate_metrics(labels, predictions, probabilities)
        
        # Add calibration metrics
        metrics['calibration'] = self._calculate_calibration_metrics(labels, probabilities)
        
        # Add uncertainty metrics
        metrics['uncertainty'] = self._calculate_uncertainty_metrics(uncertainties)
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        self._generate_visualizations(labels, predictions, probabilities, uncertainties, metrics)
        
        return metrics
    
    def _get_predictions(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Get model predictions with uncertainty."""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_epistemic_uncertainty = []
        all_aleatoric_uncertainty = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Get calibrated predictions
                probs = self.calibrator.predict_calibrated(images)
                predictions = torch.argmax(probs, dim=1)
                
                # Get uncertainty estimates
                mean_pred, epistemic_unc, aleatoric_unc = self.uncertainty_estimator.predict_with_uncertainty(images)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                all_epistemic_uncertainty.extend(epistemic_unc.cpu().numpy())
                all_aleatoric_uncertainty.extend(aleatoric_unc.cpu().numpy())
        
        uncertainties = {
            'epistemic': np.array(all_epistemic_uncertainty),
            'aleatoric': np.array(all_aleatoric_uncertainty)
        }
        
        return (
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probabilities),
            uncertainties
        )
    
    def _calculate_metrics(self, labels: np.ndarray, predictions: np.ndarray, 
                          probabilities: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        # Basic metrics
        accuracy = np.mean(predictions == labels)
        
        # Per-class metrics
        report = classification_report(labels, predictions, 
                                    target_names=list(self.id_to_class.values()),
                                    output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # ROC AUC (one-vs-rest)
        try:
            roc_auc = roc_auc_score(labels, probabilities, multi_class='ovr', average='macro')
        except:
            roc_auc = 0.0
        
        # Average precision
        try:
            avg_precision = average_precision_score(labels, probabilities, average='macro')
        except:
            avg_precision = 0.0
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'roc_auc': roc_auc,
            'average_precision': avg_precision
        }
    
    def _calculate_calibration_metrics(self, labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
        """Calculate calibration metrics."""
        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities.max(axis=1) > bin_lower) & (probabilities.max(axis=1) <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (labels[in_bin] == probabilities[in_bin].argmax(axis=1)).mean()
                avg_confidence_in_bin = probabilities[in_bin].max(axis=1).mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            'ece': ece,
            'temperature': self.calibrator.temperature.item()
        }
    
    def _calculate_uncertainty_metrics(self, uncertainties: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate uncertainty metrics."""
        epistemic = uncertainties['epistemic']
        aleatoric = uncertainties['aleatoric']
        
        return {
            'mean_epistemic_uncertainty': np.mean(epistemic),
            'mean_aleatoric_uncertainty': np.mean(aleatoric),
            'epistemic_std': np.std(epistemic),
            'aleatoric_std': np.std(aleatoric)
        }
    
    def _generate_visualizations(self, labels: np.ndarray, predictions: np.ndarray,
                                probabilities: np.ndarray, uncertainties: Dict[str, np.ndarray],
                                metrics: Dict[str, Any]):
        """Generate evaluation visualizations."""
        output_dir = Path('evaluation_plots')
        output_dir.mkdir(exist_ok=True)
        
        # Confusion Matrix
        plt.figure(figsize=(12, 10))
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.id_to_class.values()),
                   yticklabels=list(self.id_to_class.values()))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calibration Plot
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(self.id_to_class.values()):
            class_probs = probabilities[:, i]
            class_labels = (labels == i).astype(int)
            
            if len(np.unique(class_labels)) > 1:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    class_labels, class_probs, n_bins=10
                )
                plt.plot(mean_predicted_value, fraction_of_positives, 
                        marker='o', label=f'{class_name}')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'calibration_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Uncertainty Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epistemic = uncertainties['epistemic']
        aleatoric = uncertainties['aleatoric']
        
        ax1.hist(epistemic.flatten(), bins=50, alpha=0.7, label='Epistemic')
        ax1.set_xlabel('Epistemic Uncertainty')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Epistemic Uncertainty Distribution')
        ax1.legend()
        
        ax2.hist(aleatoric.flatten(), bins=50, alpha=0.7, label='Aleatoric', color='orange')
        ax2.set_xlabel('Aleatoric Uncertainty')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Aleatoric Uncertainty Distribution')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'uncertainty_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Per-class Performance
        report = metrics['classification_report']
        classes = list(self.id_to_class.values())
        precision = [report[cls]['precision'] for cls in classes if cls in report]
        recall = [report[cls]['recall'] for cls in classes if cls in report]
        f1 = [report[cls]['f1-score'] for cls in classes if cls in report]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")

def evaluate_model(checkpoint_path: str, config_path: str, data_dir: str) -> Dict[str, Any]:
    """Evaluate a trained model."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load model
    model = KhetGuardModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create data module
    data_module = KhetGuardDataModule(
        data_dir=data_dir,
        dataset_name=config['dataset_name'],
        batch_size=32,
        image_size=config['image_size']
    )
    data_module.setup()
    
    # Load class mapping
    metadata_path = Path(data_dir) / f"{config['dataset_name']}_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create evaluator
    evaluator = ModelEvaluator(model, metadata['class_mapping'], device)
    
    # Evaluate model
    metrics = evaluator.evaluate_model(data_module.test_dataloader(), data_module.val_dataloader())
    
    # Save results
    output_path = Path('evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {output_path}")
    return metrics

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Khet Guard models')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to model config')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    
    args = parser.parse_args()
    
    metrics = evaluate_model(args.checkpoint, args.config, args.data_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"ECE: {metrics['calibration']['ece']:.4f}")
    print(f"Temperature: {metrics['calibration']['temperature']:.4f}")
    print(f"Mean Epistemic Uncertainty: {metrics['uncertainty']['mean_epistemic_uncertainty']:.4f}")
    print(f"Mean Aleatoric Uncertainty: {metrics['uncertainty']['mean_aleatoric_uncertainty']:.4f}")

if __name__ == "__main__":
    main()
