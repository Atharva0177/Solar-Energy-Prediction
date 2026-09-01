#!/bin/bash

# ============================================================
# CI/CD Pipeline Setup Script
# ============================================================
#
# This script sets up the GitHub Actions CI/CD pipeline for
# the UNISOLAR Solar Power Generation Prediction Platform.
#
# Usage: bash scripts/setup_cicd.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Setting up CI/CD Pipeline for UNISOLAR Solar Platform"
echo "==========================================================="
echo ""

# ============================================================
# 1. Check prerequisites
# ============================================================

echo "✓ Checking prerequisites..."

if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

echo "✅ Git and Docker are installed"
echo ""

# ============================================================
# 2. Create .github workflows directory
# ============================================================

echo "✓ Setting up GitHub Actions workflows..."
mkdir -p "$PROJECT_ROOT/.github/workflows"
echo "✅ Workflows directory created"
echo ""

# ============================================================
# 3. Create Docker configuration
# ============================================================

echo "✓ Setting up Docker configuration..."
mkdir -p "$PROJECT_ROOT/docker"

# Verify Dockerfiles exist
if [ ! -f "$PROJECT_ROOT/docker/Dockerfile.backend" ]; then
    echo "⚠️  Dockerfile.backend not found"
fi

if [ ! -f "$PROJECT_ROOT/docker/Dockerfile.frontend" ]; then
    echo "⚠️  Dockerfile.frontend not found"
fi

echo "✅ Docker configuration ready"
echo ""

# ============================================================
# 4. Validate workflow files
# ============================================================

echo "✓ Validating workflow files..."

WORKFLOW_FILES=(
    "$PROJECT_ROOT/.github/workflows/ci.yml"
    "$PROJECT_ROOT/.github/workflows/docker-build.yml"
    "$PROJECT_ROOT/.github/workflows/code-quality.yml"
    "$PROJECT_ROOT/.github/workflows/deploy.yml"
    "$PROJECT_ROOT/.github/workflows/performance.yml"
)

for workflow in "${WORKFLOW_FILES[@]}"; do
    if [ -f "$workflow" ]; then
        echo "  ✅ $(basename "$workflow")"
    else
        echo "  ❌ $(basename "$workflow") not found"
    fi
done

echo ""

# ============================================================
# 5. Test Docker build locally (optional)
# ============================================================

read -p "Build Docker images locally? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Building backend image..."
    docker build -f docker/Dockerfile.backend -t solar-backend:latest .
    echo "✅ Backend image built"
    
    echo ""
    echo "Building frontend image..."
    docker build -f docker/Dockerfile.frontend -t solar-frontend:latest .
    echo "✅ Frontend image built"
fi

echo ""

# ============================================================
# 6. Setup local docker-compose
# ============================================================

echo "✓ Docker Compose configuration..."
if [ -f "$PROJECT_ROOT/docker-compose.yml" ]; then
    echo "✅ docker-compose.yml found"
    echo ""
    read -p "Start Docker Compose services? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$PROJECT_ROOT"
        docker-compose up -d
        echo "✅ Services started"
        echo ""
        echo "Available services:"
        echo "  - Backend API: http://localhost:8000"
        echo "  - Frontend: http://localhost:80"
        echo "  - MLflow: http://localhost:5000"
        echo "  - Prometheus: http://localhost:9090"
        echo "  - Grafana: http://localhost:3000"
    fi
else
    echo "⚠️  docker-compose.yml not found"
fi

echo ""

# ============================================================
# 7. Setup GitHub secrets
# ============================================================

echo "✓ GitHub Secrets Configuration"
echo ""
echo "To complete CI/CD setup, configure these secrets in GitHub:"
echo ""
echo "  1. Go to: Settings → Secrets and variables → Actions"
echo "  2. Add the following secrets:"
echo ""
echo "  Required for deployment:"
echo "    - STAGING_DEPLOY_KEY (SSH private key)"
echo "    - STAGING_DEPLOY_HOST (hostname)"
echo "    - PROD_DEPLOY_KEY (SSH private key)"
echo "    - PROD_DEPLOY_HOST (hostname)"
echo ""
echo "  Required for local development:"
echo "    - DB_PASSWORD (PostgreSQL password)"
echo "    - GRAFANA_PASSWORD (Grafana admin password)"
echo ""

# ============================================================
# 8. Setup Dependabot
# ============================================================

echo "✓ Dependabot Configuration"
if [ -f "$PROJECT_ROOT/.github/dependabot.yml" ]; then
    echo "✅ Dependabot configuration found"
    echo ""
    echo "Dependabot will:"
    echo "  - Check for Python dependency updates (weekly)"
    echo "  - Check for npm dependency updates (weekly)"
    echo "  - Check for GitHub Actions updates (weekly)"
    echo "  - Create automated pull requests"
else
    echo "⚠️  Dependabot configuration not found"
fi

echo ""

# ============================================================
# 9. Summary
# ============================================================

echo "✅ CI/CD Pipeline Setup Complete!"
echo "==========================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Push to GitHub:"
echo "   git add .github/ docker/ docker-compose.yml CI_CD_SETUP.md"
echo "   git commit -m 'chore: add CI/CD pipeline'"
echo "   git push origin main"
echo ""
echo "2. Configure GitHub Secrets (see instructions above)"
echo ""
echo "3. Monitor workflows:"
echo "   - Go to: Settings → Actions"
echo "   - View workflow runs and logs"
echo ""
echo "4. Local development:"
echo "   docker-compose up -d"
echo ""
echo "5. Documentation:"
echo "   - See CI_CD_SETUP.md for detailed information"
echo "   - See .github/workflows/*.yml for workflow definitions"
echo ""
echo "Need help? See CI_CD_SETUP.md for troubleshooting"
echo ""
