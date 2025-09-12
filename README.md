# KHET Guard - Main

This is the minimal repository for KHET Guard. Large assets (datasets, model exports, artifacts, logs, and local environments) are intentionally excluded to keep the repository size small.

## Structure
- `apps/mobile`: React Native app (without native build folders)
- `functions`: Serverless/Cloud functions sources
- `ml`: ML training/serving code (datasets and artifacts excluded)
- `services`: Auxiliary services
- `infra`/`terraform`: Infrastructure-as-code
- `docs`: Documentation

## Getting Started
1. Create a Python environment and install dependencies:
```bash
python -m venv .venv
.venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
```
2. Install JavaScript deps where applicable:
```bash
npm install
```

## Datasets and Models
Datasets and model artifacts are not included. See `ml/prepare_datasets.py` and `docs/` for instructions to obtain or generate them.

## Notes
- Ensure any secrets/configs are provided via environment variables or secure secret managers.
- Refer to `DEPLOYMENT_GUIDE.md` or `docs/DEPLOYMENT_GUIDE.md` for deployment steps.
