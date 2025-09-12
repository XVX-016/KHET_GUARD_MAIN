import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pytorch_lightning.loggers import WandbLogger


class LitModel(pl.LightningModule):
    def __init__(self, arch: str = "efficientnet_b4", lr: float = 3e-4, num_classes: int = 10):
        super().__init__()
        self.save_hyperparameters()

        if arch == "efficientnet_b4":
            backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
            in_features = backbone.classifier[1].in_features
            backbone.classifier[1] = nn.Linear(in_features, num_classes)
            self.model = backbone
        else:
            raise ValueError(f"Unsupported arch: {arch}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def create_loaders(dataset_root: str, batch_size: int) -> tuple[DataLoader, DataLoader, list[str]]:
    tfm_train = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
    ])
    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=tfm_train)
    val_ds = datasets.ImageFolder(val_dir, transform=tfm_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, train_ds.classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--arch", type=str, default="efficientnet_b4")
    parser.add_argument("--wandb-project", type=str, default="khet_guard")
    parser.add_argument("--output", type=str, default="../artifacts/disease_pest")
    args = parser.parse_args()

    train_loader, val_loader, classes = create_loaders(args.dataset, args.batch_size)
    model = LitModel(arch=args.arch, lr=args.lr, num_classes=len(classes))

    Path(args.output).mkdir(parents=True, exist_ok=True)
    logger = WandbLogger(project=args.wandb_project, name=f"{args.arch}_disease_pest")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
    )
    trainer.fit(model, train_loader, val_loader)

    # Save checkpoint
    ckpt_path = os.path.join(args.output, "best.ckpt")
    trainer.save_checkpoint(ckpt_path)


if __name__ == "__main__":
    main()


