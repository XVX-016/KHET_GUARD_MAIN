import argparse
import os
from pathlib import Path

import torch


def export_model(checkpoint_path: str, out_dir: str, formats: list[str] | None = None):
    formats = formats or ["onnx", "torchscript"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle Lightning checkpoint format
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # Create a simple model architecture that matches the saved state dict
        num_classes = checkpoint.get('hyper_parameters', {}).get('num_classes', 10)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(64, num_classes)
        )
        # Wrap in a module with 'backbone' attribute to match saved state dict
        class ModelWrapper(torch.nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
            def forward(self, x):
                return self.backbone(x)
        
        wrapped_model = ModelWrapper(model)
        wrapped_model.load_state_dict(checkpoint['state_dict'])
        model = wrapped_model
    else:
        model = checkpoint
        if hasattr(model, "model"):
            model = model.model
    
    model.eval()
    dummy = torch.randn(1, 3, 380, 380)

    if "onnx" in formats:
        onnx_path = os.path.join(out_dir, "model.onnx")
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        print(f"Saved ONNX to {onnx_path}")

    if "torchscript" in formats:
        traced = torch.jit.trace(model, dummy)
        ts_path = os.path.join(out_dir, "model.pt")
        traced.save(ts_path)
        print(f"Saved TorchScript to {ts_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--formats", nargs="*", default=["onnx", "torchscript"])
    args = parser.parse_args()
    export_model(args.checkpoint, args.output, args.formats)


if __name__ == "__main__":
    main()


