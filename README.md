# KHET Guard - Main

This is the minimal repository for KHET Guard. Large assets (datasets, model exports, artifacts, logs, and local environments) are intentionally excluded to keep the repository size small.



<p align="center">
  <img src="https://github.com/user-attachments/assets/a5ec58fc-5a1f-46de-a8ce-6deb63f45fb0" alt="Screenshot 1" width="300"/>
  <img src="https://github.com/user-attachments/assets/b77c2fdf-36b3-44bb-a87c-f547b1a54263" alt="Screenshot 2" width="300"/>
  <img src="https://github.com/user-attachments/assets/71ca8694-210e-4721-aa6c-3fd30c12eb40" alt="Screenshot 3" width="300"/>
  <img src="https://github.com/user-attachments/assets/6d2d48b7-f321-40a6-a266-84c1aa3b7680" alt="Screenshot 4" width="300"/>
</p>

---

## Structure
- `apps/mobile`: React Native app (without native build folders)
- `functions`: Serverless/Cloud functions sources
- `ml`: ML training/serving code
- `services`: Auxiliary services
- `infra`/`terraform`: Infrastructure-as-code
- `docs`: Documentation

## Getting Started
1. Create a Python environment and install dependencies:
```bash
python -m venv .venv
.venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt



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
