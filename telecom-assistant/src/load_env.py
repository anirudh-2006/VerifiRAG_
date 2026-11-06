"""Load `.env` from the project root (folder that contains `main.py`)."""

from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_project_dotenv() -> bool:
    path = _PROJECT_ROOT / ".env"
    return load_dotenv(path, override=False)
