# PowerShell CI/CD Setup Script
# Run with: powershell -ExecutionPolicy Bypass -File scripts/setup_cicd.ps1

param(
    [switch]$BuildDocker = $false,
    [switch]$StartServices = $false,
    [switch]$Help = $false
)

if ($Help) {
    Write-Host @"
CI/CD Pipeline Setup Script
Usage: powershell -ExecutionPolicy Bypass -File scripts/setup_cicd.ps1 [options]

Options:
  -BuildDocker    Build Docker images locally
  -StartServices  Start Docker Compose services
  -Help          Show this help message

Examples:
  # Just setup without building
  powershell -ExecutionPolicy Bypass -File scripts/setup_cicd.ps1
  
  # Build Docker images
  powershell -ExecutionPolicy Bypass -File scripts/setup_cicd.ps1 -BuildDocker
  
  # Build and start services
  powershell -ExecutionPolicy Bypass -File scripts/setup_cicd.ps1 -BuildDocker -StartServices
"@
    exit 0
}

Write-Host "🚀 Setting up CI/CD Pipeline for UNISOLAR Solar Platform" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "✓ Checking prerequisites..." -ForegroundColor Green

$tools = @("git", "docker")
foreach ($tool in $tools) {
    try {
        $null = & $tool --version 2>$null
        Write-Host "  ✅ $tool is installed" -ForegroundColor Green
    }
    catch {
        Write-Host "  ❌ $tool is not installed" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Setup directories
Write-Host "✓ Setting up GitHub Actions workflows..." -ForegroundColor Green
$workflowDir = ".\.github\workflows"
if (-not (Test-Path $workflowDir)) {
    New-Item -ItemType Directory -Path $workflowDir -Force | Out-Null
}
Write-Host "  ✅ Workflows directory ready" -ForegroundColor Green

$dockerDir = ".\docker"
if (-not (Test-Path $dockerDir)) {
    New-Item -ItemType Directory -Path $dockerDir -Force | Out-Null
}
Write-Host "  ✅ Docker directory ready" -ForegroundColor Green

Write-Host ""

# Validate workflow files
Write-Host "✓ Validating workflow files..." -ForegroundColor Green
$workflows = @(
    ".\.github\workflows\ci.yml",
    ".\.github\workflows\docker-build.yml",
    ".\.github\workflows\code-quality.yml",
    ".\.github\workflows\deploy.yml",
    ".\.github\workflows\performance.yml"
)

foreach ($workflow in $workflows) {
    if (Test-Path $workflow) {
        Write-Host "  ✅ $(Split-Path $workflow -Leaf)" -ForegroundColor Green
    }
    else {
        Write-Host "  ⚠️  $(Split-Path $workflow -Leaf) not found" -ForegroundColor Yellow
    }
}

Write-Host ""

# Build Docker images
if ($BuildDocker) {
    Write-Host "✓ Building Docker images..." -ForegroundColor Green
    
    Write-Host "  Building backend image..." -ForegroundColor Cyan
    docker build -f docker/Dockerfile.backend -t solar-backend:latest .
    Write-Host "  ✅ Backend image built" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "  Building frontend image..." -ForegroundColor Cyan
    docker build -f docker/Dockerfile.frontend -t solar-frontend:latest .
    Write-Host "  ✅ Frontend image built" -ForegroundColor Green
    
    Write-Host ""
}

# Start services
if ($StartServices) {
    Write-Host "✓ Starting Docker Compose services..." -ForegroundColor Green
    docker-compose up -d
    Start-Sleep -Seconds 3
    Write-Host "  ✅ Services started" -ForegroundColor Green
    Write-Host ""
    Write-Host "Available services:" -ForegroundColor Cyan
    Write-Host "  - Backend API: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "  - Frontend: http://localhost:80" -ForegroundColor Cyan
    Write-Host "  - MLflow: http://localhost:5000" -ForegroundColor Cyan
    Write-Host "  - Prometheus: http://localhost:9090" -ForegroundColor Cyan
    Write-Host "  - Grafana: http://localhost:3000" -ForegroundColor Cyan
    Write-Host ""
}

# GitHub Secrets
Write-Host "✓ GitHub Secrets Configuration" -ForegroundColor Green
Write-Host ""
Write-Host "To complete CI/CD setup, configure these secrets in GitHub:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Go to: Settings → Secrets and variables → Actions" -ForegroundColor Cyan
Write-Host "  2. Add the following secrets:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Required for deployment:" -ForegroundColor Yellow
Write-Host "    - STAGING_DEPLOY_KEY (SSH private key)" -ForegroundColor Yellow
Write-Host "    - STAGING_DEPLOY_HOST (hostname)" -ForegroundColor Yellow
Write-Host "    - PROD_DEPLOY_KEY (SSH private key)" -ForegroundColor Yellow
Write-Host "    - PROD_DEPLOY_HOST (hostname)" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Required for Docker Compose:" -ForegroundColor Yellow
Write-Host "    - DB_PASSWORD (PostgreSQL password)" -ForegroundColor Yellow
Write-Host "    - GRAFANA_PASSWORD (Grafana admin password)" -ForegroundColor Yellow
Write-Host ""

# Summary
Write-Host "✅ CI/CD Pipeline Setup Complete!" -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Push to GitHub:" -ForegroundColor Cyan
Write-Host "   git add .github/ docker/ docker-compose.yml" -ForegroundColor Gray
Write-Host "   git commit -m 'chore: add CI/CD pipeline'" -ForegroundColor Gray
Write-Host "   git push origin main" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Configure GitHub Secrets (see instructions above)" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Monitor workflows:" -ForegroundColor Cyan
Write-Host "   - Go to: Settings → Actions" -ForegroundColor Gray
Write-Host "   - View workflow runs and logs" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Local development:" -ForegroundColor Cyan
Write-Host "   docker-compose up -d" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Documentation:" -ForegroundColor Cyan
Write-Host "   - See CI_CD_SETUP.md for detailed information" -ForegroundColor Gray
Write-Host "   - See .github/workflows/*.yml for workflow definitions" -ForegroundColor Gray
Write-Host ""
