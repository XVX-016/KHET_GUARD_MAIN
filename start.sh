#!/bin/bash

# Khet Guard Startup Script

echo "🚀 Starting Khet Guard ML System..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Start services
echo "📦 Starting PostgreSQL and ML API..."
docker-compose -f docker-compose.simple.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Test ML API
echo "🧪 Testing ML API..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ ML API is running at http://localhost:8000"
else
    echo "❌ ML API failed to start. Check logs with: docker-compose logs ml-api"
fi

# Test Database
echo "🧪 Testing Database..."
if docker-compose -f docker-compose.simple.yml exec postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo "✅ PostgreSQL is running on localhost:5432"
else
    echo "❌ PostgreSQL failed to start. Check logs with: docker-compose logs postgres"
fi

echo ""
echo "🎉 Khet Guard is ready!"
echo ""
echo "📡 API Endpoints:"
echo "   Health: http://localhost:8000/health"
echo "   Docs: http://localhost:8000/docs"
echo "   Disease Prediction: http://localhost:8000/predict/disease_pest"
echo "   Cattle Prediction: http://localhost:8000/predict/cattle"
echo ""
echo "🗄️ Database:"
echo "   Host: localhost:5432"
echo "   Database: khet_guard"
echo "   User: postgres"
echo "   Password: khet_guard_password"
echo ""
echo "🛠️ Management Commands:"
echo "   View logs: docker-compose -f docker-compose.simple.yml logs"
echo "   Stop services: docker-compose -f docker-compose.simple.yml down"
echo "   Restart: docker-compose -f docker-compose.simple.yml restart"
