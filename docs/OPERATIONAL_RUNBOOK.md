# Khet Guard Operational Runbook

## Table of Contents
1. [Incident Response](#incident-response)
2. [Model Management](#model-management)
3. [Deployment Procedures](#deployment-procedures)
4. [Monitoring & Alerting](#monitoring--alerting)
5. [Security Procedures](#security-procedures)
6. [Data Management](#data-management)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Emergency Contacts](#emergency-contacts)

## Incident Response

### Severity Levels

**P0 - Critical (Response: 15 minutes)**
- Complete service outage
- Data loss or corruption
- Security breach
- Model serving failures

**P1 - High (Response: 1 hour)**
- Partial service degradation
- Performance issues
- High error rates
- Model accuracy degradation

**P2 - Medium (Response: 4 hours)**
- Minor service issues
- Non-critical feature failures
- Monitoring alerts

**P3 - Low (Response: 24 hours)**
- Documentation issues
- Feature requests
- Minor bugs

### Incident Response Process

1. **Detection & Acknowledgment**
   ```bash
   # Check service status
   kubectl get pods -n khet-guard
   kubectl logs -f deployment/ml-service -n khet-guard
   
   # Check metrics
   curl http://prometheus:9090/api/v1/query?query=up
   ```

2. **Initial Assessment**
   - Identify affected components
   - Check error logs and metrics
   - Determine severity level
   - Notify stakeholders

3. **Containment**
   - Isolate affected systems
   - Implement temporary fixes
   - Scale resources if needed

4. **Resolution**
   - Apply permanent fixes
   - Verify system stability
   - Monitor for recurrence

5. **Post-Incident**
   - Document lessons learned
   - Update runbooks
   - Conduct post-mortem

### Common Incidents

#### Service Outage
```bash
# Check pod status
kubectl get pods -n khet-guard

# Restart failing pods
kubectl rollout restart deployment/ml-service -n khet-guard

# Check resource usage
kubectl top pods -n khet-guard
kubectl describe pod <pod-name> -n khet-guard
```

#### High Error Rate
```bash
# Check error logs
kubectl logs -f deployment/ml-service -n khet-guard | grep ERROR

# Check metrics
curl "http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~'5..'}[5m])"

# Scale up if needed
kubectl scale deployment ml-service --replicas=5 -n khet-guard
```

#### Model Performance Degradation
```bash
# Check model metrics
curl "http://prometheus:9090/api/v1/query?query=model_accuracy"

# Check drift detection
kubectl logs -f deployment/drift-detector -n khet-guard

# Rollback model if needed
kubectl set image deployment/ml-service model=ghcr.io/khet-guard/model:v1.2.0 -n khet-guard
```

## Model Management

### Model Deployment

1. **Prepare Model**
   ```bash
   # Train and export model
   cd ml/training
   python train.py --config configs/plant_disease_config.yaml
   python export.py --checkpoint outputs/best.ckpt --output_dir exported_models
   ```

2. **Upload to S3**
   ```bash
   aws s3 cp exported_models/ s3://khet-guard-models/v1.3.0/ --recursive
   ```

3. **Update Kubernetes Deployment**
   ```yaml
   # Update model version in deployment
   spec:
     template:
       spec:
         containers:
         - name: ml-service
           image: ghcr.io/khet-guard/serving:latest
           env:
           - name: MODEL_VERSION
             value: "v1.3.0"
   ```

4. **Deploy with Canary**
   ```bash
   # Deploy canary version
   kubectl apply -f k8s/canary-deployment.yaml
   
   # Monitor canary performance
   kubectl get canary -n khet-guard
   
   # Promote to production
   kubectl promote canary ml-service -n khet-guard
   ```

### Model Rollback

1. **Identify Previous Version**
   ```bash
   kubectl rollout history deployment/ml-service -n khet-guard
   ```

2. **Rollback to Previous Version**
   ```bash
   kubectl rollout undo deployment/ml-service -n khet-guard
   ```

3. **Verify Rollback**
   ```bash
   kubectl rollout status deployment/ml-service -n khet-guard
   kubectl get pods -n khet-guard
   ```

### Model Retraining

1. **Trigger Retraining**
   ```bash
   # Manual trigger
   kubectl create job retrain-plant-disease --from=cronjob/retrain-plant-disease -n khet-guard
   
   # Check training status
   kubectl logs -f job/retrain-plant-disease -n khet-guard
   ```

2. **Monitor Training**
   ```bash
   # Check Weights & Biases
   # Monitor resource usage
   kubectl top pods -n khet-guard
   ```

3. **Deploy New Model**
   ```bash
   # Follow model deployment process
   # A/B test if needed
   ```

## Deployment Procedures

### Blue-Green Deployment

1. **Prepare Green Environment**
   ```bash
   # Deploy to green environment
   kubectl apply -f k8s/green-deployment.yaml
   
   # Verify green deployment
   kubectl get pods -l environment=green -n khet-guard
   ```

2. **Switch Traffic**
   ```bash
   # Update service to point to green
   kubectl patch service ml-service -n khet-guard -p '{"spec":{"selector":{"environment":"green"}}}'
   ```

3. **Monitor and Cleanup**
   ```bash
   # Monitor green environment
   # Clean up blue environment after verification
   kubectl delete deployment ml-service-blue -n khet-guard
   ```

### Rolling Updates

1. **Update Deployment**
   ```bash
   kubectl set image deployment/ml-service ml-service=ghcr.io/khet-guard/serving:v1.3.0 -n khet-guard
   ```

2. **Monitor Update**
   ```bash
   kubectl rollout status deployment/ml-service -n khet-guard
   ```

3. **Rollback if Needed**
   ```bash
   kubectl rollout undo deployment/ml-service -n khet-guard
   ```

## Monitoring & Alerting

### Key Metrics to Monitor

**System Metrics:**
- CPU usage > 80%
- Memory usage > 90%
- Disk usage > 85%
- Network latency > 100ms

**Application Metrics:**
- Request rate
- Error rate > 5%
- Response time > 2s
- Model inference time > 5s

**Business Metrics:**
- Model accuracy
- User satisfaction
- API usage
- Data quality

### Alerting Rules

**Critical Alerts:**
```yaml
- alert: ServiceDown
  expr: up == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Service is down"

- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "High error rate detected"
```

**Warning Alerts:**
```yaml
- alert: HighResponseTime
  expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High response time detected"
```

### Dashboard Access

**Grafana Dashboards:**
- System Overview: `http://grafana:3000/d/system-overview`
- ML Service: `http://grafana:3000/d/ml-service`
- Model Performance: `http://grafana:3000/d/model-performance`

**Prometheus Queries:**
```promql
# Service availability
up

# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m])

# Model accuracy
model_accuracy

# Drift detection
model_drift_score
```

## Security Procedures

### Security Incident Response

1. **Immediate Actions**
   - Isolate affected systems
   - Preserve evidence
   - Notify security team
   - Document timeline

2. **Investigation**
   - Analyze logs and metrics
   - Identify attack vector
   - Assess data exposure
   - Determine scope

3. **Containment**
   - Block malicious IPs
   - Revoke compromised credentials
   - Update security rules
   - Patch vulnerabilities

4. **Recovery**
   - Restore from clean backups
   - Update security configurations
   - Monitor for recurrence
   - Conduct security review

### Access Management

**User Access:**
```bash
# Add user to cluster
kubectl create clusterrolebinding user-binding --clusterrole=view --user=username

# Remove user access
kubectl delete clusterrolebinding user-binding
```

**Service Account Management:**
```bash
# Create service account
kubectl create serviceaccount ml-service-account -n khet-guard

# Bind to role
kubectl create rolebinding ml-service-binding --role=ml-service-role --serviceaccount=khet-guard:ml-service-account -n khet-guard
```

### Secret Management

**Create Secret:**
```bash
kubectl create secret generic ml-secrets \
  --from-literal=db-password=password \
  --from-literal=api-key=key \
  -n khet-guard
```

**Update Secret:**
```bash
kubectl create secret generic ml-secrets \
  --from-literal=db-password=new-password \
  --from-literal=api-key=new-key \
  -n khet-guard \
  --dry-run=client -o yaml | kubectl apply -f -
```

## Data Management

### Backup Procedures

**Database Backup:**
```bash
# Create backup
kubectl exec -it postgres-0 -n khet-guard -- pg_dump -U khetguard khetguard > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore from backup
kubectl exec -i postgres-0 -n khet-guard -- psql -U khetguard khetguard < backup_20240101_120000.sql
```

**S3 Backup:**
```bash
# Sync to backup bucket
aws s3 sync s3://khet-guard-models s3://khet-guard-models-backup
aws s3 sync s3://khet-guard-data s3://khet-guard-data-backup
```

### Data Retention

**Log Retention:**
- Application logs: 30 days
- Audit logs: 1 year
- Error logs: 90 days

**Data Retention:**
- User data: 7 years (compliance)
- Model artifacts: 1 year
- Training data: 2 years
- Inference data: 30 days

### Data Quality Monitoring

**Check Data Quality:**
```bash
# Run data quality checks
kubectl create job data-quality-check --from=cronjob/data-quality-check -n khet-guard

# Check results
kubectl logs -f job/data-quality-check -n khet-guard
```

## Troubleshooting Guide

### Common Issues

#### Pod CrashLoopBackOff
```bash
# Check pod logs
kubectl logs <pod-name> -n khet-guard

# Check pod description
kubectl describe pod <pod-name> -n khet-guard

# Check resource limits
kubectl top pod <pod-name> -n khet-guard
```

#### Service Unavailable
```bash
# Check service endpoints
kubectl get endpoints -n khet-guard

# Check service selector
kubectl get service ml-service -n khet-guard -o yaml

# Check pod labels
kubectl get pods --show-labels -n khet-guard
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

#### Model Loading Failures
```bash
# Check model files in S3
aws s3 ls s3://khet-guard-models/

# Check model service logs
kubectl logs -f deployment/ml-service -n khet-guard

# Verify model metadata
kubectl exec -it deployment/ml-service -n khet-guard -- cat /app/models/plant_disease_metadata.json
```

### Performance Issues

#### High CPU Usage
```bash
# Check CPU usage
kubectl top pods -n khet-guard

# Scale up if needed
kubectl scale deployment ml-service --replicas=5 -n khet-guard

# Check resource limits
kubectl describe pod <pod-name> -n khet-guard
```

#### High Memory Usage
```bash
# Check memory usage
kubectl top pods -n khet-guard

# Check for memory leaks
kubectl logs -f deployment/ml-service -n khet-guard | grep -i memory

# Restart if needed
kubectl rollout restart deployment/ml-service -n khet-guard
```

#### Slow Response Times
```bash
# Check response time metrics
curl "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"

# Check database performance
kubectl exec -it postgres-0 -n khet-guard -- psql -U khetguard -d khetguard -c "SELECT * FROM pg_stat_activity;"
```

## Emergency Contacts

### On-Call Rotation
- **Primary**: [Name] - [Phone] - [Email]
- **Secondary**: [Name] - [Phone] - [Email]
- **Escalation**: [Name] - [Phone] - [Email]

### Escalation Matrix
1. **Level 1**: On-call engineer (15 min response)
2. **Level 2**: Senior engineer (30 min response)
3. **Level 3**: Engineering manager (1 hour response)
4. **Level 4**: CTO (2 hour response)

### External Contacts
- **AWS Support**: [Support Case ID]
- **Security Team**: [Email]
- **Legal Team**: [Email]
- **Compliance Team**: [Email]

### Communication Channels
- **Slack**: #khet-guard-alerts
- **PagerDuty**: [Integration URL]
- **Email**: alerts@khet-guard.com
- **Phone**: [Emergency hotline]

---

**Last Updated**: [Date]
**Version**: 1.0
**Next Review**: [Date + 3 months]
