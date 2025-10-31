import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import uvicorn
from src.config import settings

def main():
    """Run FastAPI server"""
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()