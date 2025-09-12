# 🚀 Khet Guard - Complete Deployment Guide

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   📱 Mobile App │    │   🌐 Backend    │    │   🤖 ML API     │
│   (React Native)│◄──►│   (FastAPI)     │◄──►│   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   📦 PostgreSQL │    │   📊 Monitoring │
                       │   (Database)    │    │   (Prometheus)  │
                       └─────────────────┘    └─────────────────┘
```

---

## 🚀 **Quick Start with Docker Compose**

### 1. **Prerequisites**
```bash
# Install Docker and Docker Compose
# Ensure you have at least 4GB RAM available
```

### 2. **Clone and Setup**
```bash
git clone <your-repo>
cd khet-guard-main
```

### 3. **Start All Services**
```bash
# Start the complete stack
docker-compose up -d

# Check status
docker-compose ps
```

### 4. **Verify Services**
```bash
# ML API Health
curl http://localhost:8000/health

# Database Connection
docker-compose exec postgres psql -U postgres -d khet_guard -c "SELECT COUNT(*) FROM users;"

# Grafana Dashboard
open http://localhost:3000 (admin/admin)
```

---

## 📊 **Service Endpoints**

| Service | URL | Purpose |
|---------|-----|---------|
| **ML API** | http://localhost:8000 | Disease/Cattle prediction |
| **PostgreSQL** | localhost:5432 | Database |
| **Redis** | localhost:6379 | Caching |
| **Prometheus** | http://localhost:9090 | Metrics |
| **Grafana** | http://localhost:3000 | Dashboards |

---

## 🔧 **Configuration**

### **Environment Variables**
Create `.env` file:
```ini
# Database
POSTGRES_DB=khet_guard
POSTGRES_USER=postgres
POSTGRES_PASSWORD=khet_guard_password

# ML API
DISEASE_MODEL=/app/ml/artifacts/disease_pest/exports/model.onnx
CATTLE_MODEL=/app/ml/artifacts/cattle/exports/model.onnx
DISEASE_LABELS=/app/ml/artifacts/disease_pest/labels.json
CATTLE_LABELS=/app/ml/artifacts/cattle/labels.json
PESTICIDE_MAP=/app/ml/recommender/pesticide_map.json

# Monitoring
GRAFANA_ADMIN_PASSWORD=admin
```

---

## 🗄️ **Database Management**

### **Schema Setup**
```bash
# Schema is automatically created on first run
# Manual setup if needed:
docker-compose exec postgres psql -U postgres -d khet_guard -f /docker-entrypoint-initdb.d/01-schema.sql
```

### **Seed Data**
```bash
# Seed data is automatically loaded
# Manual setup if needed:
docker-compose exec postgres psql -U postgres -d khet_guard -f /docker-entrypoint-initdb.d/02-seed.sql
```

### **Database Backup**
```bash
# Create backup
docker-compose exec postgres pg_dump -U postgres khet_guard > backup.sql

# Restore backup
docker-compose exec -T postgres psql -U postgres khet_guard < backup.sql
```

---

## 📈 **Monitoring & Observability**

### **Grafana Dashboards**
1. **ML API Performance**
   - Request latency
   - Error rates
   - Model inference times

2. **Database Metrics**
   - Connection pool
   - Query performance
   - Storage usage

3. **System Resources**
   - CPU/Memory usage
   - Disk I/O
   - Network traffic

### **Prometheus Alerts**
```yaml
# Example alert rules
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High error rate detected"
```

---

## 🔄 **Development Workflow**

### **Local Development**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black .
isort .

# Type checking
mypy .
```

### **API Testing**
```bash
# Run the test suite
python test_api.py

# Manual testing
curl -X POST "http://localhost:8000/predict/disease_pest" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

---

## 🚀 **Production Deployment**

### **Kubernetes Deployment**
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=khet-guard

# Scale the ML API
kubectl scale deployment ml-api --replicas=3
```

### **AWS/GCP Deployment**
```bash
# Build and push Docker image
docker build -t khet-guard/ml-api:latest .
docker push khet-guard/ml-api:latest

# Deploy with Terraform
cd terraform/
terraform init
terraform plan
terraform apply
```

---

## 🔒 **Security Considerations**

### **API Security**
- ✅ Input validation
- ✅ Rate limiting
- ✅ Authentication tokens
- ✅ HTTPS enforcement

### **Database Security**
- ✅ Encrypted connections
- ✅ User access controls
- ✅ Regular backups
- ✅ Audit logging

### **Container Security**
- ✅ Non-root users
- ✅ Minimal base images
- ✅ Security scanning
- ✅ Regular updates

---

## 🐛 **Troubleshooting**

### **Common Issues**

**ML API not starting:**
```bash
# Check logs
docker-compose logs ml-api

# Verify model files
ls -la ml/artifacts/*/exports/
```

**Database connection issues:**
```bash
# Check PostgreSQL status
docker-compose exec postgres pg_isready -U postgres

# Test connection
docker-compose exec postgres psql -U postgres -d khet_guard -c "SELECT 1;"
```

**High memory usage:**
```bash
# Check resource usage
docker stats

# Scale down if needed
docker-compose up --scale ml-api=1
```

### **Performance Tuning**

**Database Optimization:**
```sql
-- Add indexes for better performance
CREATE INDEX CONCURRENTLY idx_scans_user_created ON scans(user_id, created_at);
CREATE INDEX CONCURRENTLY idx_analytics_region_disease ON analytics(region, disease_name);
```

**ML API Optimization:**
```python
# Enable model caching
# Use ONNX Runtime optimizations
# Implement request batching
```

---

## 📋 **Maintenance Tasks**

### **Daily**
- [ ] Check service health
- [ ] Monitor error rates
- [ ] Review resource usage

### **Weekly**
- [ ] Database backup
- [ ] Security updates
- [ ] Performance review

### **Monthly**
- [ ] Model retraining
- [ ] Capacity planning
- [ ] Security audit

---

## 🎯 **Next Steps**

1. **Mobile App Integration**
   - Connect React Native app to ML API
   - Implement offline queue
   - Add push notifications

2. **Advanced Features**
   - Real-time model updates
   - A/B testing framework
   - Advanced analytics

3. **Scaling**
   - Load balancing
   - Auto-scaling
   - Multi-region deployment

---

✅ **Your Khet Guard system is now production-ready!** 🎉

For support, check the logs or create an issue in the repository.
