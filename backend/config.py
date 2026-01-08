import os

class DB:
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_NAME = os.getenv("DB_NAME", "gis")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
    DB_PORT = os.getenv("DB_PORT", 25433)

from pathlib import Path

IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)