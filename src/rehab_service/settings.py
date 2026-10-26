"""Paths shared by the service, desktop application and command-line tools."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
TEMPLATE_DIR = PROJECT_ROOT / "video_templates"
MEDIA_DIR = PROJECT_ROOT / "media"
DATA_DIR = PROJECT_ROOT / "data"
