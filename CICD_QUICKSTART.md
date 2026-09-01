# CICD Quick Start Guide

## Overview

This guide helps you quickly get started with the UNISOLAR Solar Platform's CI/CD pipeline.

## What's Included

✅ **GitHub Actions Workflows**
- Continuous Integration (tests, linting, coverage)
- Docker image building and pushing
- Code quality and security scanning
- Performance benchmarking
- Automated deployment

✅ **Docker Setup**
- Multi-stage Dockerfile for backend (FastAPI)
- Multi-stage Dockerfile for frontend (React + Nginx)
- Docker Compose for local development
- Nginx configuration with API proxy
- Health checks and logging

✅ **Monitoring & Observability**
- Prometheus metrics collection
- Grafana dashboards
- MLflow experiment tracking
- Redis caching

✅ **Security**
- CodeQL static analysis
- Bandit Python security scanner
- OWASP Dependency Check
- Dependabot automated updates

## Quick Start (5 minutes)

### 1. Prerequisites

```bash
# Check you have these installed
git --version          # >= 2.30
docker --version       # >= 24.0
docker-compose --version  # >= 2.0
npm --version          # >= 18 (for frontend)
python --version       # >= 3.11
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/solar-gemini.git
cd solar-gemini

# Create .env file for local development
cat > .env << EOF
DB_USER=solar
DB_PASSWORD=dev_password_123
DB_NAME=solar_db
GRAFANA_PASSWORD=admin_password_123
EOF

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm ci
cd ..
```

### 3. Start Local Development Environment

#### Option A: Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# Check services are running
docker-compose ps

# View logs
docker-compose logs -f backend
```

#### Option B: Local Python Development

```bash
# In terminal 1: Start backend
cd solar-gemini
python -m uvicorn api.main:app --reload

# In terminal 2: Start frontend dev server
cd solar-gemini/frontend
npm run dev
```

### 4. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Frontend | http://localhost | - |
| Backend API | http://localhost:8000 | - |
| API Docs | http://localhost:8000/docs | - |
| MLflow | http://localhost:5000 | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin / admin_password_123 |
| PostgreSQL | localhost:5432 | solar / dev_password_123 |

### 5. Run Tests

```bash
# Python backend tests
pytest tests/ -v --cov=src

# Frontend tests (if configured)
cd frontend
npm test

# Frontend linting
npm run lint
```

## GitHub Setup

### 1. Configure GitHub Secrets

Go to **Settings → Secrets and variables → Actions** and add:

```
STAGING_DEPLOY_KEY        # (SSH private key for staging server)
STAGING_DEPLOY_HOST       # (staging.example.com)
PROD_DEPLOY_KEY           # (SSH private key for production)
PROD_DEPLOY_HOST          # (solar.example.com)
DB_PASSWORD               # (PostgreSQL password)
GRAFANA_PASSWORD          # (Grafana admin password)
```

### 2. Enable GitHub Actions

1. Go to **Settings → Actions → General**
2. Enable "Allow all actions and reusable workflows"
3. Save

### 3. Setup Dependabot (Optional)

1. Go to **Settings → Code security and analysis**
2. Enable "Dependabot alerts"
3. Enable "Dependabot security updates"

### 4. Setup Code Scanning (Optional)

1. Go to **Settings → Code security and analysis**
2. Enable "CodeQL"

## Workflow Overview

### 1. On Pull Request

**Triggered by:** `git push` to feature branch or creation of pull request

**Steps:**
1. Run Python linting (Ruff, mypy)
2. Run Python tests with coverage
3. Build frontend
4. Run API integration tests
5. Run ML model tests
6. Run security scanning
7. Post results as PR comment

**Status Badge:**
```markdown
[![CI](https://github.com/yourusername/solar-gemini/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/solar-gemini/actions/workflows/ci.yml)
```

### 2. On Merge to Main

**Triggered by:** `git push` to `main` or merge PR

**Steps:**
1. All PR checks (tests, linting, security)
2. Build Docker images
3. Push Docker images to registry
4. Run performance benchmarks
5. Wait for manual approval
6. Deploy to production

### 3. On Git Tag (Release)

**Triggered by:** `git tag v*` and `git push --tags`

**Steps:**
1. All CI checks
2. Build Docker images
3. Tag images with version number
4. Push to registry
5. Create GitHub Release
6. Deploy to production (requires approval)

## Common Tasks

### Build Docker Images Locally

```bash
# Backend
docker build -f docker/Dockerfile.backend -t solar-backend:latest .

# Frontend
docker build -f docker/Dockerfile.frontend -t solar-frontend:latest .

# Run locally
docker run -p 8000:8000 solar-backend:latest
docker run -p 80:80 solar-frontend:latest
```

### Run Linting and Formatting

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files

# Auto-fix issues
ruff check src/ --fix
ruff format src/
cd frontend && npm run lint -- --fix
```

### Deploy Changes

```bash
# Development/Staging
git add .
git commit -m "feat: add new feature"
git push origin develop

# Production
git tag -a v1.0.1 -m "Release version 1.0.1"
git push origin v1.0.1
# Then approve deployment in GitHub Actions
```

### View Logs

```bash
# Docker Compose logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f db

# GitHub Actions logs
# Go to: Actions tab → Click workflow run → Click job
```

### Monitor Performance

1. **Prometheus:** http://localhost:9090
   - Query metrics (http_requests_total, model_inference_time_ms, etc.)

2. **Grafana:** http://localhost:3000
   - Pre-built dashboards for system, application, and ML metrics

3. **MLflow:** http://localhost:5000
   - View experiment runs and model metrics

## Troubleshooting

### Docker won't start

```bash
# Check Docker daemon
docker ps

# Restart Docker
# On Windows: Restart Docker Desktop
# On Linux: sudo systemctl restart docker
# On Mac: Restart Docker Desktop

# Clean up and retry
docker system prune -a
docker-compose up -d
```

### Port already in use

```bash
# Find process using port 8000
# Windows
netstat -ano | findstr :8000

# Linux/Mac
lsof -i :8000

# Kill process or change port in docker-compose.yml
```

### Tests failing

```bash
# Clear caches
rm -rf .pytest_cache __pycache__
rm -rf frontend/node_modules

# Reinstall dependencies
pip install -r requirements.txt
cd frontend && npm ci && cd ..

# Run tests with verbose output
pytest tests/ -vv
```

### Performance issues

```bash
# Check resource usage
docker stats

# Increase Docker memory limit
# Windows/Mac: Docker Desktop → Settings → Resources

# Check what's consuming resources
docker top container_name
```

## Next Steps

1. **Read documentation:**
   - [CI_CD_SETUP.md](CI_CD_SETUP.md) - Detailed CI/CD configuration
   - [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment strategies
   - [README.md](README.md) - Project overview

2. **Configure deployment:**
   - Set GitHub Secrets
   - Update deployment scripts
   - Configure monitoring

3. **Start developing:**
   - Create feature branch
   - Make changes
   - Push and create PR
   - Review CI results

## Support

For issues or questions:
1. Check [CI_CD_SETUP.md](CI_CD_SETUP.md) troubleshooting section
2. Check [DEPLOYMENT.md](DEPLOYMENT.md) troubleshooting section
3. Review GitHub Actions logs
4. Check Docker logs: `docker-compose logs`

## Quick Reference

```bash
# Development
docker-compose up -d                    # Start services
docker-compose down                     # Stop services
docker-compose logs -f backend          # View logs
pytest tests/ -v                        # Run tests
cd frontend && npm run dev              # Start frontend dev server

# Deployment
git tag v1.0.0 && git push --tags      # Create release
git push origin develop                 # Deploy to staging

# Cleanup
docker-compose down -v                  # Remove volumes
docker system prune -a                  # Clean Docker

# Monitoring
curl http://localhost:8000/api/v1/health    # Check API
docker-compose ps                       # Check services
```

---

**Need help?** See [CI_CD_SETUP.md](CI_CD_SETUP.md) or [DEPLOYMENT.md](DEPLOYMENT.md)
