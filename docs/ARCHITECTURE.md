# Khet Guard Architecture

## Overview

Khet Guard is a production-ready agricultural AI platform that provides plant disease detection, cattle breed classification, and crop recommendation services. The system is designed for government-grade deployment with robust monitoring, security, and scalability.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mobile App    │    │   Web Portal    │    │   Admin Panel   │
│   (React Native)│    │   (React)       │    │   (React)       │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │     API Gateway           │
                    │   (Load Balancer)         │
                    └─────────────┬─────────────┘
                                 │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────┴─────────┐    ┌─────────┴─────────┐    ┌─────────┴─────────┐
│  ML Service       │    │  Cloud Functions  │    │  GEE Service      │
│  (FastAPI)        │    │  (Node.js)        │    │  (Python)         │
└─────────┬─────────┘    └─────────┬─────────┘    └─────────┬─────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │     Data Layer            │
                    │  (PostgreSQL + Redis)     │
                    └─────────────┬─────────────┘
                                 │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────┴─────────┐    ┌─────────┴─────────┐    ┌─────────┴─────────┐
│  Model Storage    │    │  Data Storage     │    │  Monitoring       │
│  (S3)             │    │  (S3)             │    │  (Prometheus)     │
└───────────────────┘    └───────────────────┘    └───────────────────┘
```

## Components

### 1. Mobile Application (React Native + Expo)

**Features:**
- Plant disease detection via camera
- Cattle breed classification
- Crop recommendation based on location
- Offline support with upload queuing
- Multi-language support (Hindi/English)
- Dark/light mode toggle
- 3D animations for engagement

**Key Services:**
- `ImageProcessor`: Handles image capture, validation, and preprocessing
- `ErrorHandler`: Centralized error handling with offline queuing
- `LocationService`: GPS and map-based location selection

### 2. ML Training Pipeline

**Components:**
- `data_prep.py`: Dataset preparation and validation
- `augment.py`: Advanced data augmentation (RandAugment, MixUp, CutMix)
- `train.py`: PyTorch Lightning training with Optuna HPO
- `evaluate.py`: Model evaluation with calibration and uncertainty
- `export.py`: Model export to TFLite, ONNX, TorchScript

**Features:**
- Transfer learning with EfficientNet/ResNet
- Class imbalance handling (focal loss, weighted sampling)
- Model calibration (temperature scaling)
- Uncertainty estimation (MC Dropout, ensemble)
- Explainability (Grad-CAM integration)
- Experiment tracking (Weights & Biases)

### 3. ML Serving Service (FastAPI)

**Endpoints:**
- `POST /predict/plant_disease`: Plant disease detection
- `POST /predict/cattle_breed`: Cattle breed classification
- `POST /recommend/crop`: Crop recommendation
- `GET /models/status`: Model status and metadata

**Features:**
- ONNX model inference
- Uncertainty quantification
- Authentication (JWT tokens)
- Prometheus metrics
- Sentry error tracking
- Redis caching
- Health checks

### 4. Cloud Functions (Node.js)

**Functions:**
- `recommendCrop`: Integrates SoilGrids, OpenWeather, GEE
- `detectPlantDisease`: Server-side plant disease detection
- `detectCattle`: Server-side cattle breed classification
- `locationData`: Cached environmental data retrieval

**Integrations:**
- SoilGrids API for soil data
- OpenWeather API for weather data
- GEE microservice for NDVI data
- Firestore for data caching

### 5. GEE Microservice (Python FastAPI)

**Features:**
- Google Earth Engine integration
- NDVI/NDWI calculation
- Historical data analysis
- Export functionality

### 6. Data Storage

**PostgreSQL:**
- User data and authentication
- Location metadata
- Scan results and history
- Model metadata

**Redis:**
- Session management
- API response caching
- Upload queue management
- Real-time data

**S3:**
- Model artifacts storage
- Image data storage
- Backup and archival

### 7. Monitoring & Observability

**Prometheus:**
- Metrics collection
- Custom business metrics
- Model performance metrics

**Grafana:**
- Dashboards for system health
- Model performance visualization
- Alert management

**Drift Detection:**
- Input distribution monitoring
- Model performance tracking
- Automated alerting

**Sentry:**
- Error tracking and alerting
- Performance monitoring
- User session tracking

## Security Architecture

### Authentication & Authorization
- JWT-based authentication
- Role-based access control (RBAC)
- Service-to-service authentication (mTLS)
- API key management

### Data Protection
- Encryption at rest (AES-256)
- Encryption in transit (TLS 1.3)
- PII data anonymization
- Secure key management (AWS Secrets Manager)

### Network Security
- VPC with private subnets
- Security groups with least privilege
- WAF for API protection
- DDoS protection

### Compliance
- Data retention policies
- Audit logging
- GDPR compliance features
- Government security standards

## Deployment Architecture

### Infrastructure (Terraform)
- AWS EKS cluster
- RDS PostgreSQL
- ElastiCache Redis
- S3 buckets for storage
- IAM roles and policies

### CI/CD Pipeline
- GitHub Actions workflows
- Automated testing
- Security scanning
- Model training automation
- Blue-green deployments

### Monitoring Stack
- Prometheus + Grafana
- ELK stack for logging
- AlertManager for notifications
- Custom drift detection

## Scalability & Performance

### Horizontal Scaling
- Kubernetes HPA for auto-scaling
- Load balancers for traffic distribution
- Database read replicas
- Redis cluster mode

### Performance Optimization
- Model quantization (INT8)
- ONNX runtime optimization
- CDN for static assets
- Database query optimization

### Caching Strategy
- Redis for API responses
- CDN for static content
- Database query caching
- Model inference caching

## Data Flow

### Plant Disease Detection
1. User captures image via mobile app
2. Image preprocessing and validation
3. Upload to ML service (with offline queuing)
4. ONNX model inference
5. Uncertainty quantification
6. Results display with confidence scores

### Crop Recommendation
1. User provides location (GPS or map pin)
2. Cloud Function fetches environmental data:
   - Soil data from SoilGrids
   - Weather data from OpenWeather
   - NDVI data from GEE service
3. Rule-based recommendation engine
4. Results with reasoning and uncertainty

### Model Training Pipeline
1. Data preparation and validation
2. Augmentation and preprocessing
3. Hyperparameter optimization
4. Model training with monitoring
5. Evaluation and calibration
6. Model export and deployment
7. A/B testing and rollback capability

## Disaster Recovery

### Backup Strategy
- Automated database backups
- S3 cross-region replication
- Model artifact versioning
- Configuration backup

### Recovery Procedures
- RTO: 4 hours
- RPO: 1 hour
- Multi-region deployment
- Automated failover

## Cost Optimization

### Resource Management
- Spot instances for training
- Reserved instances for production
- Auto-scaling based on demand
- Storage lifecycle policies

### Monitoring
- Cost allocation tags
- Resource utilization monitoring
- Automated cost alerts
- Regular cost reviews

## Future Enhancements

### Planned Features
- Real-time video analysis
- Edge deployment support
- Advanced ML models (Vision Transformers)
- Multi-modal data integration
- Advanced analytics dashboard

### Technical Improvements
- Service mesh implementation
- Advanced monitoring (Jaeger tracing)
- Machine learning operations (MLOps)
- Automated model retraining
- Advanced security features
