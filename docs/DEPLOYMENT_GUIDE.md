# Khet Guard Deployment Guide

## Prerequisites

### System Requirements
- **Kubernetes**: 1.28+
- **Docker**: 20.10+
- **Terraform**: 1.0+
- **AWS CLI**: 2.0+
- **kubectl**: 1.28+
- **Helm**: 3.0+

### AWS Account Setup
1. Create AWS account with appropriate permissions
2. Configure AWS CLI:
   ```bash
   aws configure
   ```
3. Create IAM user with required permissions:
   - EKS cluster management
   - RDS management
   - S3 access
   - VPC management

### Required Secrets
Create the following secrets in your environment:
- `OPENWEATHER_API_KEY`: OpenWeather API key
- `GEE_SERVICE_ACCOUNT_JSON`: Google Earth Engine service account JSON
- `WANDB_API_KEY`: Weights & Biases API key
- `SENTRY_DSN`: Sentry DSN for error tracking
- `DB_PASSWORD`: Database password
- `JWT_SECRET`: JWT signing secret

## Infrastructure Deployment

### 1. Clone Repository
```bash
git clone https://github.com/your-org/khet-guard.git
cd khet-guard
```

### 2. Configure Terraform
```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:
```hcl
project_name = "khet-guard"
environment = "production"
aws_region = "us-west-2"
db_password = "your-secure-password"
```

### 3. Deploy Infrastructure
```bash
# Initialize Terraform
terraform init

# Plan deployment
terraform plan

# Deploy infrastructure
terraform apply
```

### 4. Configure kubectl
```bash
# Get cluster credentials
aws eks update-kubeconfig --region us-west-2 --name khet-guard-cluster

# Verify connection
kubectl get nodes
```

## Application Deployment

### 1. Create Namespace
```bash
kubectl create namespace khet-guard
```

### 2. Deploy Secrets
```bash
# Create secrets
kubectl create secret generic app-secrets \
  --from-literal=openweather-api-key=$OPENWEATHER_API_KEY \
  --from-literal=db-password=$DB_PASSWORD \
  --from-literal=jwt-secret=$JWT_SECRET \
  --from-literal=sentry-dsn=$SENTRY_DSN \
  -n khet-guard

# Create GEE service account secret
kubectl create secret generic gee-service-account \
  --from-file=service-account.json=$GEE_SERVICE_ACCOUNT_JSON \
  -n khet-guard
```

### 3. Deploy Database
```bash
# Deploy PostgreSQL
kubectl apply -f k8s/postgres.yaml

# Wait for database to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n khet-guard --timeout=300s
```

### 4. Deploy Redis
```bash
# Deploy Redis
kubectl apply -f k8s/redis.yaml

# Wait for Redis to be ready
kubectl wait --for=condition=ready pod -l app=redis -n khet-guard --timeout=300s
```

### 5. Deploy ML Service
```bash
# Deploy ML serving service
kubectl apply -f k8s/ml-service.yaml

# Wait for service to be ready
kubectl wait --for=condition=ready pod -l app=ml-service -n khet-guard --timeout=300s
```

### 6. Deploy Cloud Functions
```bash
# Deploy Cloud Functions
kubectl apply -f k8s/cloud-functions.yaml

# Wait for functions to be ready
kubectl wait --for=condition=ready pod -l app=cloud-functions -n khet-guard --timeout=300s
```

### 7. Deploy GEE Service
```bash
# Deploy GEE microservice
kubectl apply -f k8s/gee-service.yaml

# Wait for service to be ready
kubectl wait --for=condition=ready pod -l app=gee-service -n khet-guard --timeout=300s
```

### 8. Deploy Monitoring Stack
```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Deploy Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values k8s/prometheus-values.yaml

# Deploy Grafana
helm install grafana grafana/grafana \
  --namespace monitoring \
  --set adminPassword=admin \
  --values k8s/grafana-values.yaml
```

### 9. Deploy Ingress
```bash
# Deploy NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/aws/deploy.yaml

# Deploy ingress rules
kubectl apply -f k8s/ingress.yaml
```

## Model Deployment

### 1. Train Models
```bash
# Navigate to training directory
cd ml/training

# Install dependencies
pip install -r requirements.txt

# Prepare data
python data_prep.py --data_dir ../../ --output_dir ./processed_data

# Train plant disease model
python train.py --config configs/plant_disease_config.yaml --data_dir ./processed_data --output_dir ./outputs/plant_disease

# Train cattle breed model
python train.py --config configs/cattle_breed_config.yaml --data_dir ./processed_data --output_dir ./outputs/cattle_breed
```

### 2. Export Models
```bash
# Export plant disease model
python export.py \
  --checkpoint ./outputs/plant_disease/best.ckpt \
  --config ./outputs/plant_disease/model_info.json \
  --data_dir ./processed_data \
  --output_dir ./exported_models/plant_disease

# Export cattle breed model
python export.py \
  --checkpoint ./outputs/cattle_breed/best.ckpt \
  --config ./outputs/cattle_breed/model_info.json \
  --data_dir ./processed_data \
  --output_dir ./exported_models/cattle_breed
```

### 3. Upload Models to S3
```bash
# Get S3 bucket name from Terraform output
BUCKET_NAME=$(terraform output -raw models_bucket_name)

# Upload models
aws s3 cp ./exported_models/plant_disease/ s3://$BUCKET_NAME/plant_disease/v1.0.0/ --recursive
aws s3 cp ./exported_models/cattle_breed/ s3://$BUCKET_NAME/cattle_breed/v1.0.0/ --recursive
```

### 4. Update Model Configuration
```bash
# Update model version in ML service
kubectl set env deployment/ml-service MODEL_VERSION=v1.0.0 -n khet-guard

# Restart ML service to load new models
kubectl rollout restart deployment/ml-service -n khet-guard
```

## Mobile App Deployment

### 1. Configure Environment
```bash
cd apps/mobile

# Create .env file
cp .env.example .env
```

Edit `.env`:
```env
EXPO_PUBLIC_FUNCTIONS_URL=https://your-api-domain.com
EXPO_PUBLIC_FIREBASE_API_KEY=your-firebase-api-key
EXPO_PUBLIC_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
EXPO_PUBLIC_FIREBASE_PROJECT_ID=your-project-id
EXPO_PUBLIC_FIREBASE_STORAGE_BUCKET=your-project.appspot.com
EXPO_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=123456789
EXPO_PUBLIC_FIREBASE_APP_ID=1:123456789:android:abcdef
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Build for Android
```bash
# Build APK
npx expo build:android

# Build AAB (for Play Store)
npx expo build:android --type app-bundle
```

### 4. Deploy to Play Store
```bash
# Upload to Play Console
# Follow Google Play Console instructions
```

## Verification

### 1. Check Service Health
```bash
# Check all pods are running
kubectl get pods -n khet-guard

# Check service endpoints
kubectl get services -n khet-guard

# Check ingress
kubectl get ingress -n khet-guard
```

### 2. Test API Endpoints
```bash
# Get API URL
API_URL=$(kubectl get ingress khet-guard-api -n khet-guard -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

# Test health endpoint
curl http://$API_URL/health

# Test plant disease prediction
curl -X POST http://$API_URL/predict/plant_disease \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "base64-encoded-image"}'

# Test crop recommendation
curl -X POST http://$API_URL/recommend/crop \
  -H "Content-Type: application/json" \
  -d '{"latitude": 28.6, "longitude": 77.2}'
```

### 3. Check Monitoring
```bash
# Get Grafana URL
GRAFANA_URL=$(kubectl get service grafana -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

# Access Grafana dashboard
echo "Grafana URL: http://$GRAFANA_URL"
echo "Username: admin"
echo "Password: admin"
```

### 4. Test Mobile App
1. Install APK on Android device
2. Test plant disease detection
3. Test cattle breed classification
4. Test crop recommendation
5. Test offline functionality

## Scaling

### Horizontal Pod Autoscaling
```bash
# Create HPA for ML service
kubectl autoscale deployment ml-service --cpu-percent=70 --min=2 --max=10 -n khet-guard

# Check HPA status
kubectl get hpa -n khet-guard
```

### Vertical Pod Autoscaling
```bash
# Install VPA
kubectl apply -f k8s/vpa.yaml

# Check VPA status
kubectl get vpa -n khet-guard
```

## Backup and Recovery

### 1. Database Backup
```bash
# Create backup job
kubectl apply -f k8s/backup-job.yaml

# Manual backup
kubectl exec -it postgres-0 -n khet-guard -- pg_dump -U khetguard khetguard > backup_$(date +%Y%m%d_%H%M%S).sql
```

### 2. Model Backup
```bash
# Sync models to backup bucket
aws s3 sync s3://khet-guard-models s3://khet-guard-models-backup
```

### 3. Configuration Backup
```bash
# Backup Kubernetes configurations
kubectl get all -n khet-guard -o yaml > khet-guard-backup.yaml
```

## Troubleshooting

### Common Issues

#### Pod Not Starting
```bash
# Check pod status
kubectl describe pod <pod-name> -n khet-guard

# Check logs
kubectl logs <pod-name> -n khet-guard

# Check events
kubectl get events -n khet-guard --sort-by='.lastTimestamp'
```

#### Service Not Accessible
```bash
# Check service endpoints
kubectl get endpoints -n khet-guard

# Check ingress
kubectl describe ingress khet-guard-api -n khet-guard

# Check DNS
nslookup <service-name>
```

#### Database Connection Issues
```bash
# Check database status
kubectl get pods -l app=postgres -n khet-guard

# Check database logs
kubectl logs -f postgres-0 -n khet-guard

# Test connection
kubectl exec -it postgres-0 -n khet-guard -- psql -U khetguard -d khetguard -c "SELECT 1;"
```

### Performance Issues

#### High CPU Usage
```bash
# Check resource usage
kubectl top pods -n khet-guard

# Scale up if needed
kubectl scale deployment ml-service --replicas=5 -n khet-guard
```

#### High Memory Usage
```bash
# Check memory usage
kubectl top pods -n khet-guard

# Check for memory leaks
kubectl logs -f deployment/ml-service -n khet-guard | grep -i memory
```

## Security Hardening

### 1. Network Policies
```bash
# Apply network policies
kubectl apply -f k8s/network-policies.yaml
```

### 2. Pod Security Policies
```bash
# Apply pod security policies
kubectl apply -f k8s/pod-security-policies.yaml
```

### 3. RBAC Configuration
```bash
# Apply RBAC rules
kubectl apply -f k8s/rbac.yaml
```

### 4. Secrets Management
```bash
# Use external secret management
kubectl apply -f k8s/external-secrets.yaml
```

## Maintenance

### Regular Tasks

#### Daily
- Check service health
- Review error logs
- Monitor resource usage
- Check backup status

#### Weekly
- Review security logs
- Update dependencies
- Check model performance
- Review monitoring dashboards

#### Monthly
- Security audit
- Performance review
- Capacity planning
- Disaster recovery test

### Updates

#### Application Updates
```bash
# Update image tags
kubectl set image deployment/ml-service ml-service=ghcr.io/khet-guard/serving:v1.1.0 -n khet-guard

# Monitor rollout
kubectl rollout status deployment/ml-service -n khet-guard
```

#### Infrastructure Updates
```bash
# Update Terraform
cd terraform
terraform plan
terraform apply
```

## Support

### Documentation
- Architecture: `docs/ARCHITECTURE.md`
- Operational Runbook: `docs/OPERATIONAL_RUNBOOK.md`
- API Documentation: `docs/API.md`

### Contact
- **Technical Support**: support@khet-guard.com
- **Emergency**: +1-XXX-XXX-XXXX
- **Slack**: #khet-guard-support

---

**Last Updated**: [Date]
**Version**: 1.0
**Next Review**: [Date + 3 months]
