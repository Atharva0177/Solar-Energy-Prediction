## GitHub Secrets Setup

To complete the CI/CD pipeline configuration, you need to add secrets to GitHub.

### 1. Access GitHub Secrets

1. Go to your repository on GitHub
2. Click **Settings** (top right)
3. Click **Secrets and variables** → **Actions** (left sidebar)

### 2. Required Secrets

Add the following secrets (click "New repository secret"):

#### Deployment Secrets

| Secret Name | Description | Example |
|---|---|---|
| `STAGING_DEPLOY_KEY` | SSH private key for staging server | `-----BEGIN OPENSSH PRIVATE KEY-----\n...` |
| `STAGING_DEPLOY_HOST` | Staging server hostname | `staging.solar.example.com` |
| `PROD_DEPLOY_KEY` | SSH private key for production server | `-----BEGIN OPENSSH PRIVATE KEY-----\n...` |
| `PROD_DEPLOY_HOST` | Production server hostname | `solar.example.com` |

#### Docker/Container Registry Secrets

These are optional if using GitHub Container Registry (GHCR):

| Secret Name | Description |
|---|---|
| `REGISTRY_USERNAME` | Docker registry username (usually GitHub username) |
| `REGISTRY_TOKEN` | Docker registry password/token (GitHub PAT with packages scope) |

#### Environment Secrets

| Secret Name | Description | Example |
|---|---|---|
| `DB_PASSWORD` | PostgreSQL database password | `securepwd123!@#` |
| `GRAFANA_PASSWORD` | Grafana admin password | `grafana_admin_123` |

### 3. Generating SSH Keys for Deployment

If you don't have SSH keys yet:

**On your local machine:**

```bash
# Generate SSH key pair
ssh-keygen -t ed25519 -f ~/.ssh/staging_deploy -N ""

# Display private key (for STAGING_DEPLOY_KEY secret)
cat ~/.ssh/staging_deploy

# Display public key (for server's authorized_keys)
cat ~/.ssh/staging_deploy.pub
```

**On the staging server:**

```bash
# Add public key to authorized_keys
cat ~/.ssh/staging_deploy.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### 4. GitHub Personal Access Token (Optional)

If using Docker Hub or custom registry:

1. Go to **Settings → Developer settings → Personal access tokens**
2. Click **Generate new token (classic)**
3. Grant `write:packages` and `read:packages` scopes
4. Copy the token
5. Add as `REGISTRY_TOKEN` secret in Actions

### 5. Verifying Secrets

To verify a secret is set correctly:

1. Go to **Settings → Secrets and variables → Actions**
2. Each secret shows:
   - ✅ Secret name and update date
   - Value is masked (never visible)
   - Action to edit or delete

### 6. Using Secrets in Workflows

Secrets are automatically available in all workflows:

```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy
        env:
          DEPLOY_KEY: ${{ secrets.STAGING_DEPLOY_KEY }}
          DEPLOY_HOST: ${{ secrets.STAGING_DEPLOY_HOST }}
        run: |
          # Secrets are securely injected here
          # Never logged or printed
```

### 7. Best Practices

✅ **DO:**
- Use unique, strong passwords
- Rotate secrets regularly
- Use SSH keys (not passwords when possible)
- Store backups in a secure password manager
- Review secret access logs

❌ **DON'T:**
- Commit secrets to repository
- Print secrets in workflow logs
- Share secrets via Slack/email
- Use same secret across environments
- Commit `.env` files

### 8. Troubleshooting

**Secret not found error:**
```yaml
Error: secrets.STAGING_DEPLOY_KEY is not defined
```
Solution: Check secret name matches exactly (case-sensitive)

**Permission denied error:**
```
Permission denied (publickey).
```
Solution: Verify SSH key is in `~/.ssh/authorized_keys` on server

**Deployment fails silently:**
```bash
# Check workflow logs in GitHub
# Actions → Workflow run → Job logs
```

### 9. Rotating Secrets

Regularly update secrets:

1. Generate new SSH key pair
2. Update authorized_keys on servers
3. Update GitHub secret with new key
4. Remove old key from servers

### 10. Removing Secrets

To remove a secret:

1. Go to **Settings → Secrets and variables → Actions**
2. Find the secret
3. Click **Delete** (red trash icon)
4. Confirm deletion

Deleted secrets:
- Are removed from all future workflow runs
- Cannot be recovered
- Should be revoked on servers (e.g., remove from `authorized_keys`)
