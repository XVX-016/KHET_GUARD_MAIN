"""
Production training pipeline for Khet Guard ML models.
Supports plant disease detection and cattle breed classification.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import torchmetrics
import optuna
from optuna.integration import PyTorchLightningPruningCallback

import wandb
import timm
from torchvision import models
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_prep import DatasetPreprocessor
from augment import AgriculturalAugmentation, AugmentedDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KhetGuardModel(pl.LightningModule):
    """PyTorch Lightning model for Khet Guard."""
    
    def __init__(self, 
                 num_classes: int,
                 model_name: str = 'efficientnet_b0',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 dropout_rate: float = 0.2,
                 use_focal_loss: bool = False,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 class_weights: Optional[List[float]] = None):
        
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Load pretrained model
        if model_name.startswith('efficientnet'):
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
            feature_dim = self.backbone.num_features
        else:
            # Use torchvision models
            if model_name == 'resnet50':
                self.backbone = models.resnet50(pretrained=True)
                feature_dim = self.backbone.fc.in_features
                self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            elif model_name == 'resnet101':
                self.backbone = models.resnet101(pretrained=True)
                feature_dim = self.backbone.fc.in_features
                self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.val_precision = torchmetrics.Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.val_recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average='macro')
        
        # For uncertainty estimation
        self.dropout_samples = 10
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        return self.classifier(features)
    
    def forward_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty estimation using MC Dropout."""
        self.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.dropout_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)
        
        return mean_pred, uncertainty
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with MixUp/CutMix support."""
        images = batch['images']
        labels = batch['labels']
        
        # Handle MixUp/CutMix
        if batch.get('use_mixup', False):
            labels_a, labels_b = batch['labels_a'], batch['labels_b']
            lam = batch['lam']
            
            logits = self.forward(images)
            loss = lam * self.criterion(logits, labels_a) + (1 - lam) * self.criterion(logits, labels_b)
            
            # Use labels_a for accuracy calculation
            preds = torch.argmax(logits, dim=1)
            acc = self.train_accuracy(preds, labels_a)
            
        elif batch.get('use_cutmix', False):
            labels_a, labels_b = batch['labels_a'], batch['labels_b']
            lam = batch['lam']
            
            logits = self.forward(images)
            loss = lam * self.criterion(logits, labels_a) + (1 - lam) * self.criterion(logits, labels_b)
            
            # Use labels_a for accuracy calculation
            preds = torch.argmax(logits, dim=1)
            acc = self.train_accuracy(preds, labels_a)
            
        else:
            logits = self.forward(images)
            loss = self.criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            acc = self.train_accuracy(preds, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        images = batch['images']
        labels = batch['labels']
        
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        self.val_accuracy(preds, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

class KhetGuardDataModule(pl.LightningDataModule):
    """Data module for Khet Guard training."""
    
    def __init__(self, 
                 data_dir: str,
                 dataset_name: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 image_size: int = 224,
                 use_mixup: bool = True,
                 use_cutmix: bool = True):
        
        super().__init__()
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.class_mapping = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        # Load metadata
        metadata_path = self.data_dir / f'{self.dataset_name}_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.class_mapping = metadata['class_mapping']
        
        # Load dataset splits
        for split in ['train', 'val', 'test']:
            csv_path = self.data_dir / f'{self.dataset_name}_{split}.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                data = df.to_dict('records')
                
                if split == 'train':
                    self.train_data = data
                elif split == 'val':
                    self.val_data = data
                elif split == 'test':
                    self.test_data = data
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        augmentation = AgriculturalAugmentation(
            image_size=self.image_size,
            use_mixup=self.use_mixup,
            use_cutmix=self.use_cutmix
        )
        
        dataset = AugmentedDataset(
            self.train_data,
            augmentation,
            is_training=True,
            use_mixup=self.use_mixup,
            use_cutmix=self.use_cutmix
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        augmentation = AgriculturalAugmentation(image_size=self.image_size)
        
        dataset = AugmentedDataset(
            self.val_data,
            augmentation,
            is_training=False
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        augmentation = AgriculturalAugmentation(image_size=self.image_size)
        
        dataset = AugmentedDataset(
            self.test_data,
            augmentation,
            is_training=False
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

def objective(trial: optuna.Trial, config: Dict) -> float:
    """Optuna objective function for hyperparameter optimization."""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Update config
    config['learning_rate'] = learning_rate
    config['weight_decay'] = weight_decay
    config['dropout_rate'] = dropout_rate
    config['batch_size'] = batch_size
    
    # Create data module
    data_module = KhetGuardDataModule(
        data_dir=config['data_dir'],
        dataset_name=config['dataset_name'],
        batch_size=batch_size,
        image_size=config['image_size']
    )
    
    # Create model
    model = KhetGuardModel(
        num_classes=config['num_classes'],
        model_name=config['model_name'],
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='auto',
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_f1')]
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Return validation F1 score
    return trainer.callback_metrics['val_f1'].item()

def train_model(config: Dict) -> str:
    """Train model with given configuration."""
    
    # Set random seeds
    pl.seed_everything(config.get('seed', 42))
    
    # Create data module
    data_module = KhetGuardDataModule(
        data_dir=config['data_dir'],
        dataset_name=config['dataset_name'],
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        use_mixup=config.get('use_mixup', True),
        use_cutmix=config.get('use_cutmix', True)
    )
    
    # Load class mapping
    metadata_path = Path(config['data_dir']) / f"{config['dataset_name']}_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create model
    model = KhetGuardModel(
        num_classes=metadata['num_classes'],
        model_name=config['model_name'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        dropout_rate=config['dropout_rate'],
        use_focal_loss=config.get('use_focal_loss', False)
    )
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config['output_dir'],
            filename='best-{epoch:02d}-{val_f1:.2f}',
            monitor='val_f1',
            mode='max',
            save_top_k=1,
            save_last=True
        ),
        EarlyStopping(
            monitor='val_f1',
            patience=config.get('patience', 10),
            mode='max'
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Create logger
    if config.get('use_wandb', False):
        logger = WandbLogger(
            project=config['project_name'],
            name=config['run_name'],
            save_dir=config['output_dir']
        )
    else:
        logger = TensorBoardLogger(
            save_dir=config['output_dir'],
            name=config['run_name']
        )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='auto',
        devices=config.get('devices', 1),
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        check_val_every_n_epoch=1
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module)
    
    # Save model info
    model_info = {
        'model_name': config['model_name'],
        'num_classes': metadata['num_classes'],
        'class_mapping': metadata['class_mapping'],
        'best_checkpoint': str(Path(config['output_dir']) / 'best.ckpt'),
        'config': config
    }
    
    with open(Path(config['output_dir']) / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"Training completed. Model saved to {config['output_dir']}")
    return str(Path(config['output_dir']) / 'best.ckpt')

def run_hyperparameter_optimization(config: Dict) -> Dict:
    """Run hyperparameter optimization with Optuna."""
    
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(
        lambda trial: objective(trial, config),
        n_trials=config.get('n_trials', 20)
    )
    
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    return study.best_params

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Khet Guard models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--hpo', action='store_true', help='Run hyperparameter optimization')
    parser.add_argument('--data_dir', type=str, default='./processed_data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line args
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    
    # Create output directory
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    if args.hpo:
        # Run hyperparameter optimization
        best_params = run_hyperparameter_optimization(config)
        config.update(best_params)
        
        # Save best config
        with open(Path(config['output_dir']) / 'best_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    # Train model
    checkpoint_path = train_model(config)
    logger.info(f"Training completed. Checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()
