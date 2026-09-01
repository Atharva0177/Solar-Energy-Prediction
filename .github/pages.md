# GitHub Pages Configuration for CI/CD Pipeline

The CI/CD pipeline automatically generates and publishes artifacts to GitHub Pages.

## Published Artifacts

### Coverage Reports

- **Location:** `https://yourusername.github.io/solar-gemini/coverage/`
- **Updated:** After every push to `main`
- **Contains:** HTML coverage report showing code coverage by file

### Benchmark Results

- **Location:** `https://yourusername.github.io/solar-gemini/benchmarks/`
- **Updated:** Weekly on Sunday
- **Contains:** Performance metrics and trends over time

### API Documentation

- **Location:** `https://yourusername.github.io/solar-gemini/api-docs/`
- **Updated:** After every push to `main`
- **Contains:** Generated Swagger/OpenAPI documentation

## Setup GitHub Pages

1. Go to **Settings → Pages**
2. Under "Build and deployment":
   - Select "Deploy from a branch"
   - Select branch: `gh-pages`
   - Select folder: `/ (root)`
3. Click "Save"

## Workflow Configuration

The CI/CD workflows automatically deploy to the `gh-pages` branch:

```yaml
- name: Deploy to GitHub Pages
  uses: actions/deploy-pages@v2
```

## Customization

To add more artifacts to GitHub Pages:

1. Update `.github/workflows/ci.yml`:
```yaml
- name: Upload coverage
  uses: actions/upload-pages-artifact@v2
  with:
    path: 'coverage/'
```

2. Artifacts will be published to:
   - `https://yourusername.github.io/solar-gemini/coverage/`

## Status Badges

Add to your README.md:

```markdown
[![CI/CD](https://github.com/yourusername/solar-gemini/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/solar-gemini/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)](https://yourusername.github.io/solar-gemini/coverage/)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue)](https://github.com/yourusername/packages)
[![License](https://img.shields.io/badge/license-Proprietary-red)](LICENSE)
```
