# DEPLOYMENT.md

## Deployment Guide for UNISOLAR Solar Platform

This guide explains how to deploy the UNISOLAR Solar Power Generation Prediction Platform using the CI/CD pipeline.

## Deployment Environments

### Staging Environment

**Trigger:** Automatic on push to `develop` branch

**Process:**
```bash
# Commit to develop triggers automatic deployment
git add .
git commit -m "Add new feature"
git push origin develop
```

**Checks:**
- All CI tests must pass
- Code quality checks must pass
- Docker images must build successfully

**Access:** `https://staging.solar.example.com`

### Production Environment

**Trigger:** Manual approval + git tag on `main` branch

**Process:**
```bash
# Create semantic version tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Then approve deployment in GitHub
# Go to: Actions → Deploy → Approve and Run
```

**Checks:**
- All CI tests must pass
- Code quality checks must pass
- Docker images must build successfully
- Manual approval from maintainers
- Production deployment requires backup

**Access:** `https://solar.example.com`

## Deployment Architecture

### Infrastructure Components

```
┌─────────────────────────────────────────────────┐
│           GitHub Actions (CI/CD)                │
│  ┌─────────────────────────────────────────┐   │
│  │ 1. Test (Python, Frontend, API)         │   │
│  │ 2. Build (Docker images)                │   │
│  │ 3. Security Scan (CodeQL, Bandit, etc.) │   │
│  │ 4. Deploy (Staging → Production)        │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
         ↓                              ↓
    ┌─────────────┐          ┌──────────────────┐
    │   Staging   │          │   Production     │
    │  Environment│          │  Environment     │
    └─────────────┘          └──────────────────┘
         ↓                              ↓
    ┌─────────────┐          ┌──────────────────┐
    │ Docker      │          │ Kubernetes or    │
    │ Compose     │          │ Cloud Provider   │
    │ (Test env)  │          │ (Production)     │
    └─────────────┘          └──────────────────┘
         ↓                              ↓
    Frontend/Backend              Frontend/Backend
    MLflow/Postgres               MLflow/Postgres
    Redis/Prometheus              Redis/Prometheus
```

## Deployment Methods

### Method 1: Docker Compose (Local/Staging)

**Prerequisites:**
- Docker and Docker Compose installed
- 8GB RAM, 20GB disk space

**Steps:**
```bash
# Clone repository
git clone https://github.com/yourusername/solar-gemini.git
cd solar-gemini

# Create .env file
cat > .env << EOF
DB_USER=solar
DB_PASSWORD=$(openssl rand -base64 32)
DB_NAME=solar_db
GRAFANA_PASSWORD=$(openssl rand -base64 32)
EOF

# Start services
docker-compose up -d

# Verify services
docker-compose ps

# View logs
docker-compose logs -f backend
```

**Access:**
- Frontend: http://localhost
- Backend API: http://localhost:8000/api/v1
- MLflow: http://localhost:5000
- Grafana: http://localhost:3000

**Stop services:**
```bash
docker-compose down
```

### Method 2: Kubernetes (Production)

**Prerequisites:**
- Kubernetes cluster (1.24+)
- kubectl configured
- Docker images pushed to registry

**Deploy manifest** (`k8s/deployment.yaml`):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: solar-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: solar-backend
  template:
    metadata:
      labels:
        app: solar-backend
    spec:
      containers:
      - name: backend
        image: ghcr.io/yourusername/solar-gemini-backend:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: solar-secrets
              key: database_url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

**Deploy:**
```bash
# Create secrets
kubectl create secret generic solar-secrets \
  --from-literal=database_url='postgresql://...' \
  --from-literal=api_key='...'

# Create configmap
kubectl create configmap solar-config \
  --from-file=configs/

# Deploy
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods
kubectl logs solar-backend-xxxxx
```

### Method 3: AWS/GCP/Azure

**AWS ECS Deployment:**
```bash
# Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker tag solar-backend:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/solar:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/solar:latest

# Update ECS service
aws ecs update-service --cluster solar --service solar-backend --force-new-deployment
```

**GCP Cloud Run:**
```bash
# Push to GCR
docker tag solar-backend:latest gcr.io/PROJECT_ID/solar-backend:latest
docker push gcr.io/PROJECT_ID/solar-backend:latest

# Deploy
gcloud run deploy solar-backend \
  --image gcr.io/PROJECT_ID/solar-backend:latest \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2
```

## Deployment Checklist

### Pre-deployment

- [ ] All tests pass in CI
- [ ] Code review completed
- [ ] Security scan passed
- [ ] Database migrations prepared
- [ ] Backup created
- [ ] Rollback plan documented
- [ ] Deployment window scheduled
- [ ] Stakeholders notified

### During Deployment

- [ ] Monitor CI/CD pipeline
- [ ] Check for deployment errors
- [ ] Verify health checks
- [ ] Monitor application logs
- [ ] Check database connectivity
- [ ] Verify API endpoints
- [ ] Test critical user flows
- [ ] Monitor system resources

### Post-deployment

- [ ] Verify all services running
- [ ] Check application metrics
- [ ] Monitor error rates
- [ ] Verify data integrity
- [ ] Smoke tests passed
- [ ] User acceptance testing
- [ ] Update deployment documentation
- [ ] Notify stakeholders

## Rollback Procedure

If deployment fails or causes issues:

```bash
# Kubernetes rollback
kubectl rollout undo deployment/solar-backend

# Docker Compose rollback
docker-compose down
git checkout previous-tag
docker-compose up -d

# AWS ECS rollback
aws ecs update-service --cluster solar --service solar-backend \
  --task-definition solar-backend:previous-revision
```

## Monitoring & Alerting

### Key Metrics to Monitor

- **Application Health:** HTTP 5xx errors, API response times
- **Database:** Query performance, connection pool usage
- **ML Models:** Prediction accuracy, inference latency
- **Infrastructure:** CPU, memory, disk usage
- **Security:** Failed authentication attempts, unauthorized access

### Grafana Dashboards

Access at: http://monitoring.example.com:3000

Pre-configured dashboards:
- System Overview
- Application Performance
- Database Metrics
- ML Model Metrics
- API Latency

### Alerting Rules

Example Prometheus alert:
```yaml
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
  annotations:
    summary: "High error rate detected"
    action: "Check application logs and restart if needed"
```

## Troubleshooting

### Service not starting

```bash
# Check logs
docker-compose logs backend

# Check health
curl http://localhost:8000/api/v1/health

# Restart service
docker-compose restart backend
```

### Database connection issues

```bash
# Verify database running
docker-compose ps db

# Check connection string
docker-compose exec backend env | grep DATABASE

# Restart database
docker-compose down db
docker-compose up -d db
```

### Out of memory

```bash
# Check resource usage
docker stats

# Increase limits in docker-compose.yml
# or k8s deployment.yaml

# Restart services
docker-compose down
docker-compose up -d
```

## Performance Optimization

### Docker Compose

- Use SSD for data volumes
- Set appropriate resource limits
- Enable Docker BuildKit caching
- Use health checks

### Kubernetes

- Use HPA (Horizontal Pod Autoscaling)
- Set resource requests and limits
- Use PodDisruptionBudgets
- Enable pod affinity rules

### Database

- Create appropriate indexes
- Enable query caching
- Monitor slow queries
- Use connection pooling (pgBouncer)

## Security Considerations

1. **Secrets Management:** Use GitHub Secrets, AWS Secrets Manager, or HashiCorp Vault
2. **Network Security:** Use VPC, security groups, network policies
3. **Container Security:** Scan images, run as non-root, use read-only filesystems
4. **Access Control:** RBAC in Kubernetes, IAM in cloud providers
5. **Audit Logging:** Enable CloudTrail, audit logs, container registries

## Disaster Recovery

### Backup Strategy

- Daily database backups
- Model checkpoint backups
- Configuration backups
- Cross-region replication

### Recovery Time Objective (RTO)

- Staging: 1 hour
- Production: 30 minutes

### Recovery Point Objective (RPO)

- Database: 1 hour
- Models: 4 hours
- Configuration: real-time

## References

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [GitHub Actions Deployment](https://docs.github.com/actions/deployment)
- [AWS ECS Deployment](https://docs.aws.amazon.com/ecs/)
- [GCP Cloud Run Documentation](https://cloud.google.com/run/docs)
