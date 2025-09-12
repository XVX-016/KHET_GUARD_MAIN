@echo off
REM Khet Guard Startup Script for Windows

echo 🚀 Starting Khet Guard ML System...

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Start services
echo 📦 Starting PostgreSQL and ML API...
docker-compose -f docker-compose.simple.yml up -d

REM Wait for services to be ready
echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Test ML API
echo 🧪 Testing ML API...
curl -f http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ ML API is running at http://localhost:8000
) else (
    echo ❌ ML API failed to start. Check logs with: docker-compose logs ml-api
)

REM Test Database
echo 🧪 Testing Database...
docker-compose -f docker-compose.simple.yml exec postgres pg_isready -U postgres >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ PostgreSQL is running on localhost:5432
) else (
    echo ❌ PostgreSQL failed to start. Check logs with: docker-compose logs postgres
)

echo.
echo 🎉 Khet Guard is ready!
echo.
echo 📡 API Endpoints:
echo    Health: http://localhost:8000/health
echo    Docs: http://localhost:8000/docs
echo    Disease Prediction: http://localhost:8000/predict/disease_pest
echo    Cattle Prediction: http://localhost:8000/predict/cattle
echo.
echo 🗄️ Database:
echo    Host: localhost:5432
echo    Database: khet_guard
echo    User: postgres
echo    Password: khet_guard_password
echo.
echo 🛠️ Management Commands:
echo    View logs: docker-compose -f docker-compose.simple.yml logs
echo    Stop services: docker-compose -f docker-compose.simple.yml down
echo    Restart: docker-compose -f docker-compose.simple.yml restart
echo.
pause
