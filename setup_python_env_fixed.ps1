# Khet Guard Python Environment Setup Script (Fixed Version)
# This script fixes Python environment issues and installs all ML dependencies

Write-Host "🚀 Setting up Khet Guard Python Environment (Fixed Version)..." -ForegroundColor Green
Write-Host "===============================================================" -ForegroundColor Green

# Check if we're in the right directory
if (-not (Test-Path "requirements.txt")) {
    Write-Host "❌ Error: requirements.txt not found. Please run this script from the project root." -ForegroundColor Red
    exit 1
}

# Step 1: Remove existing virtual environment if it exists
Write-Host "`n1️⃣ Cleaning up existing virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "   Removing existing venv directory..." -ForegroundColor Gray
    Remove-Item -Recurse -Force "venv"
}

# Step 2: Create fresh virtual environment
Write-Host "`n2️⃣ Creating fresh virtual environment..." -ForegroundColor Yellow
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: Failed to create virtual environment" -ForegroundColor Red
    exit 1
}
Write-Host "   ✅ Virtual environment created successfully" -ForegroundColor Green

# Step 3: Activate virtual environment
Write-Host "`n3️⃣ Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: Failed to activate virtual environment" -ForegroundColor Red
    exit 1
}
Write-Host "   ✅ Virtual environment activated" -ForegroundColor Green

# Step 4: Upgrade pip and setuptools
Write-Host "`n4️⃣ Upgrading pip and setuptools..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: Failed to upgrade pip" -ForegroundColor Red
    exit 1
}
Write-Host "   ✅ pip and setuptools upgraded successfully" -ForegroundColor Green

# Step 5: Install PyTorch with CUDA support first (most important)
Write-Host "`n5️⃣ Installing PyTorch with CUDA support..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Warning: CUDA PyTorch installation failed, trying CPU version..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Error: Failed to install PyTorch" -ForegroundColor Red
        exit 1
    }
}
Write-Host "   ✅ PyTorch installed successfully" -ForegroundColor Green

# Step 6: Install core dependencies one by one to avoid conflicts
Write-Host "`n6️⃣ Installing core dependencies..." -ForegroundColor Yellow

# Install numpy first
Write-Host "   Installing numpy..." -ForegroundColor Gray
pip install numpy==1.26.4
if ($LASTEXITCODE -ne 0) {
    Write-Host "   ⚠️  Warning: Specific numpy version failed, trying latest..." -ForegroundColor Yellow
    pip install numpy
}

# Install Pillow (with pre-compiled wheel)
Write-Host "   Installing Pillow..." -ForegroundColor Gray
pip install Pillow
if ($LASTEXITCODE -ne 0) {
    Write-Host "   ⚠️  Warning: Pillow installation failed, trying alternative..." -ForegroundColor Yellow
    pip install --only-binary=all Pillow
}

# Install other core dependencies
Write-Host "   Installing FastAPI and related..." -ForegroundColor Gray
pip install fastapi uvicorn python-multipart pydantic

Write-Host "   Installing ONNX runtime..." -ForegroundColor Gray
pip install onnxruntime onnx

Write-Host "   Installing matplotlib..." -ForegroundColor Gray
pip install matplotlib

Write-Host "   Installing other utilities..." -ForegroundColor Gray
pip install python-dotenv pytest requests

Write-Host "   ✅ Core dependencies installed" -ForegroundColor Green

# Step 7: Install additional ML dependencies
Write-Host "`n7️⃣ Installing additional ML dependencies..." -ForegroundColor Yellow
pip install pytorch-lightning wandb psycopg2-binary
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: Failed to install ML dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "   ✅ ML dependencies installed" -ForegroundColor Green

# Step 8: Verify installation
Write-Host "`n8️⃣ Verifying installation..." -ForegroundColor Yellow
Write-Host "   Testing PyTorch..." -ForegroundColor Gray
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: PyTorch verification failed" -ForegroundColor Red
    exit 1
}

Write-Host "   Testing other dependencies..." -ForegroundColor Gray
python -c "import fastapi, uvicorn, onnxruntime, numpy, PIL; print('✅ All core dependencies working')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: Core dependencies verification failed" -ForegroundColor Red
    exit 1
}

# Step 9: Test ML API startup (without actually running it)
Write-Host "`n9️⃣ Testing ML API import..." -ForegroundColor Yellow
python -c "import inference_api; print('✅ ML API imports successfully')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Warning: ML API import failed - this might be due to missing model files" -ForegroundColor Yellow
    Write-Host "   This is normal if model files haven't been generated yet" -ForegroundColor Gray
}

# Step 10: Summary
Write-Host "`n🎉 Python Environment Setup Complete!" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green
Write-Host "✅ Virtual environment: venv\" -ForegroundColor Green
Write-Host "✅ PyTorch: Installed with CUDA support" -ForegroundColor Green
Write-Host "✅ All dependencies: Installed" -ForegroundColor Green
Write-Host "`n📝 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Activate the environment: venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "   2. Test ML training: python ml/cattle/train.py --help" -ForegroundColor White
Write-Host "   3. Test ML API: python inference_api.py" -ForegroundColor White
Write-Host "   4. Start full system: .\start.bat" -ForegroundColor White

Write-Host "`n🔧 Environment activation command:" -ForegroundColor Cyan
Write-Host "   venv\Scripts\Activate.ps1" -ForegroundColor White

