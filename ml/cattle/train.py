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


class LitCattle(pl.LightningModule):
    def __init__(self, arch: str = "resnet50", lr: float = 1e-4, num_classes: int = 10):
        super().__init__()
        self.save_hyperparameters()

        if arch == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, num_classes)
            self.model = backbone
        else:
            raise ValueError(f"Unsupported arch: {arch}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


def create_loaders(dataset_root: str, batch_size: int) -> tuple[DataLoader, DataLoader, list[str]]:
    tfm_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(os.path.join(dataset_root, "train"), transform=tfm_train)
    val_ds = datasets.ImageFolder(os.path.join(dataset_root, "val"), transform=tfm_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, train_ds.classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--wandb-project", type=str, default="khet_guard")
    parser.add_argument("--output", type=str, default="../artifacts/cattle")
    args = parser.parse_args()

    train_loader, val_loader, classes = create_loaders(args.dataset, args.batch_size)
    model = LitCattle(arch=args.arch, lr=args.lr, num_classes=len(classes))

    Path(args.output).mkdir(parents=True, exist_ok=True)
    logger = WandbLogger(project=args.wandb_project, name=f"{args.arch}_cattle")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
    )
    trainer.fit(model, train_loader, val_loader)

    ckpt_path = os.path.join(args.output, "best.ckpt")
    trainer.save_checkpoint(ckpt_path)


if __name__ == "__main__":
    main()


