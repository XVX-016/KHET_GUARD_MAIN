#!/usr/bin/env python3
"""Create mock models for testing export functionality"""

import torch
import torch.nn as nn
import os
from pathlib import Path

class MockDiseaseModel(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class MockCattleModel(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def create_mock_models():
    # Create directories
    disease_dir = Path("artifacts/disease_pest")
    cattle_dir = Path("artifacts/cattle")
    disease_dir.mkdir(parents=True, exist_ok=True)
    cattle_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock models
    disease_model = MockDiseaseModel()
    cattle_model = MockCattleModel()
    
    # Save as Lightning checkpoints
    disease_ckpt = {
        'state_dict': disease_model.state_dict(),
        'hyper_parameters': {'num_classes': 38}
    }
    cattle_ckpt = {
        'state_dict': cattle_model.state_dict(),
        'hyper_parameters': {'num_classes': 50}
    }
    
    torch.save(disease_ckpt, disease_dir / "best.ckpt")
    torch.save(cattle_ckpt, cattle_dir / "best.ckpt")
    
    print(f"Created mock disease model: {disease_dir / 'best.ckpt'}")
    print(f"Created mock cattle model: {cattle_dir / 'best.ckpt'}")

if __name__ == "__main__":
    create_mock_models()
