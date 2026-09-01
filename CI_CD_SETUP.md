# CI/CD Pipeline Configuration Reference

This document describes the complete CI/CD pipeline for the UNISOLAR Solar Power Generation Prediction Platform.

## Overview

The CI/CD pipeline uses **GitHub Actions** to automate testing, building, security scanning, and deployment workflows. The pipeline is designed to ensure code quality, security, and reliability across the entire stack (Python backend, TypeScript frontend, and ML models).

## Workflow Files

### 1. **Continuous Integration (`.github/workflows/ci.yml`)**

Runs on every push to `main`/`develop` and pull requests. Tests all code changes.

**Jobs:**
- **Python Lint & Test** — Ruff linting, mypy type checking, pytest unit tests, coverage reporting
- **Frontend Build & Lint** — npm dependencies, oxlint, TypeScript compilation, Vite build
- **API Integration Tests** — FastAPI endpoint tests
- **ML Model Tests** — Model training/inference tests, feature engineering tests
- **Data & Cross-site Tests** — Data validation, cross-site evaluation tests
- **Security Dependency Check** — pip-audit and safety vulnerability scanning

**Triggers:**
- Push to `main` or `develop`
- Pull requests to `main` or `develop`
- Daily schedule (2 AM UTC)

**Outputs:**
- Test reports (JUnit XML)
- Coverage reports (Codecov)
- Artifacts (frontend build, HTML coverage)

---

### 2. **Docker Build & Push (`.github/workflows/docker-build.yml`)**

Builds and pushes Docker images for backend and frontend to GitHub Container Registry.

**Jobs:**
- Backend Docker image build
- Frontend Docker image build

**Image Naming:**
- Backend: `ghcr.io/owner/repo-backend:tag`
- Frontend: `ghcr.io/owner/repo-frontend:tag`

**Tags:**
- Branch names (e.g., `main`, `develop`)
- Semantic versions (e.g., `v1.0.0`, `1.0`)
- Git SHA (commit hash)
- `latest` (for default branch)

**Triggers:**
- Push to `main` or `develop`
- Git tags (e.g., `v1.0.0`)
- Pull requests (builds without pushing)

---

### 3. **Code Quality & Security (`.github/workflows/code-quality.yml`)**

Advanced security and quality analysis using industry-standard tools.

**Jobs:**
- **CodeQL Analysis** — GitHub's SAST (Static Application Security Testing)
- **Bandit Security Scan** — Python security linter
- **OWASP Dependency Check** — Known vulnerability scanning
- **Radon Code Complexity** — Cyclomatic complexity, maintainability index

**Artifacts:**
- CodeQL alerts (GitHub Security tab)
- Bandit report (JSON format)
- Dependency Check report (JSON format)
- Complexity metrics

**Triggers:**
- Push to `main` or `develop`
- Pull requests to `main` or `develop`

---

### 4. **Deploy (`.github/workflows/deploy.yml`)**

Handles staging and production deployments with environment protection.

**Jobs:**
- **Build Verification** — Pre-deployment Docker image build cache
- **Deploy to Staging** — Auto-deploy on `develop` branch
- **Deploy to Production** — Manual approval required, triggered by git tags

**Environment Protection:**
- Production requires review before deployment
- Secrets management (deploy keys, hosts) via GitHub secrets

**Triggers:**
- Push to `develop` (staging)
- Git tags on `main` (production, requires manual approval)
- Manual workflow dispatch

---

### 5. **Performance & Benchmarks (`.github/workflows/performance.yml`)**

Tracks ML model performance and API response times across versions.

**Jobs:**
- **ML Model Benchmarks** — Model inference speed, memory usage
- **API Performance Tests** — HTTP endpoint response times
- **Frontend Build Performance** — Build time, bundle size analysis

**Outputs:**
- Benchmark history (GitHub Pages)
- Performance trends over time

**Triggers:**
- Push to `main` or `develop`
- Pull requests to `main` or `develop`
- Weekly schedule (Sunday midnight UTC)

---

## Docker Configuration

### Backend Dockerfile (`docker/Dockerfile.backend`)

Multi-stage build optimizing for production:
1. **Builder stage** — Compiles Python wheels with all dependencies
2. **Runtime stage** — Minimal image with only runtime requirements

**Features:**
- Non-root user (solar:1000)
- Health checks
- Environment variable configuration
- FastAPI server on port 8000

### Frontend Dockerfile (`docker/Dockerfile.frontend`)

Multi-stage build with Nginx serving:
1. **Builder stage** — Node.js build environment, npm install & build
2. **Runtime stage** — Nginx Alpine image serving static files

**Features:**
- Non-root user (nginx:1000)
- Gzip compression
- Health checks
- API proxy to backend
- Security headers (CSP, X-Frame-Options, etc.)

### Docker Compose (`docker-compose.yml`)

Complete local development environment:
- **Backend** (FastAPI)
- **Frontend** (Nginx)
- **PostgreSQL** database
- **MLflow** tracking server
- **Redis** cache
- **Prometheus** metrics
- **Grafana** dashboards

**Usage:**
```bash
docker-compose up -d
# Frontend: http://localhost
# Backend API: http://localhost:8000/api/v1
# MLflow: http://localhost:5000
# Grafana: http://localhost:3000
```

---

## Dependency Management

### Dependabot (`.github/dependabot.yml`)

Automated dependency updates:
- **Python** (pip) — Weekly updates every Monday at 4 AM UTC
- **Node.js** (npm) — Weekly updates every Monday at 4 AM UTC
- **GitHub Actions** — Weekly updates every Monday at 4 AM UTC

**Configuration:**
- Max 5 open PRs per ecosystem
- Auto-reviewer assignment
- Commit message prefixes (chore/ci)
- Labels for filtering

---

## Security

### Secret Management

Required GitHub Secrets (set in repository settings):

```
STAGING_DEPLOY_KEY        # SSH private key for staging
STAGING_DEPLOY_HOST       # Staging server hostname
PROD_DEPLOY_KEY           # SSH private key for production
PROD_DEPLOY_HOST          # Production server hostname
GRAFANA_PASSWORD          # Initial Grafana admin password
DB_PASSWORD               # PostgreSQL password
```

### Security Scanning

1. **CodeQL** — Detects code patterns, data flows, configurations issues
2. **Bandit** — Scans Python code for common security issues
3. **OWASP Dependency Check** — Identifies known vulnerabilities in dependencies
4. **pip-audit** — Checks Python packages for known vulnerabilities
5. **Safety** — Additional Python vulnerability database

### Deployment Protection

- **Staging** — Automatic deployment on `develop` push
- **Production** — Manual approval required, only on versioned releases (git tags)
- **Secrets** — Never logged or exposed in workflow output
- **Credentials** — GitHub-provided token for package registry authentication

---

## Metrics & Monitoring

### Codecov Integration

- Automatic coverage report uploads after tests
- Pull request comments with coverage changes
- Coverage trends over time
- Per-file coverage analysis

### Benchmark Tracking

- Stores ML model performance metrics
- Tracks API response times
- Records build times
- GitHub Pages history for trend analysis

### Prometheus & Grafana

- Collects metrics from backend API
- Visualizes system performance
- Tracks model inference latency
- Monitors resource usage (CPU, memory)

---

## Local Development

### Running Tests Locally

```bash
# Python tests
pytest tests/ -v --cov=src

# Frontend tests
cd frontend && npm run build

# Linting
ruff check src/
cd frontend && npm run lint
```

### Docker Development

```bash
# Build and run locally
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

### Pre-commit Hooks

Install pre-commit to run checks before commits:

```bash
pip install pre-commit
pre-commit install
```

---

## Troubleshooting

### CI Failures

1. **Python tests fail** — Check Python version (3.11+), run `pip install -r requirements.txt`
2. **Frontend build fails** — Check Node.js version (20+), run `npm ci` in `frontend/`
3. **Docker build fails** — Check Dockerfile syntax, ensure all COPY paths exist
4. **Deployment fails** — Verify secrets are set, check deployment key permissions

### Performance Issues

1. **Slow tests** — Use `pytest -n auto` for parallel testing
2. **Docker build slow** — Use layer caching, minimize image size
3. **Frontend build slow** — Check for large dependencies, use tree-shaking

### Coverage Issues

1. **Coverage not uploaded** — Check Codecov token in secrets
2. **Low coverage** — Write more tests, run `coverage html` to identify gaps

---

## References

- [GitHub Actions Documentation](https://docs.github.com/actions)
- [Docker Documentation](https://docs.docker.com)
- [pytest Documentation](https://docs.pytest.org)
- [GitHub Container Registry](https://docs.github.com/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [Dependabot Documentation](https://docs.github.com/code-security/dependabot)
