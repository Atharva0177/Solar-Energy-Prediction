"""Production API entrypoint for Docker deployment.

Initializes the FastAPI application with ParquetStore for serving forecasts.
"""

from pathlib import Path
import sys

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Ensure artifacts directory exists for static file mounting
artifacts_dir = REPO_ROOT / "artifacts"
artifacts_dir.mkdir(parents=True, exist_ok=True)

from src.api.app import create_app
from src.api.store import ParquetStore

# Initialize store and create app
store = ParquetStore(REPO_ROOT)
app = create_app(store)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
