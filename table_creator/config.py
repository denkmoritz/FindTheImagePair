import os
import re
from pathlib import Path

def generate_table_name(city_name: str, mly_score: float = None) -> str:
    """
    Generate a valid PostgreSQL table name from city name and optionally MLY score.
    Converts to lowercase and replaces spaces/special chars with underscores.
    Examples:
        "Cape Town", 0.9 -> "cape_town_mly0_9"
        "New York City", None -> "new_york_city"
        "São Paulo", 0.85 -> "sao_paulo_mly0_85"
    """
    # Convert to lowercase
    name = city_name.lower()
    # Replace spaces and special characters with underscores
    name = re.sub(r'[\s\-]+', '_', name)  # spaces and hyphens become underscores
    name = re.sub(r'[^\w_]+', '', name)    # remove other special chars
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Replace multiple underscores with single underscore
    name = re.sub(r'_+', '_', name)
    
    # Add MLY score if provided
    if mly_score is not None:
        # Format the MLY score: replace decimal point with underscore
        mly_str = f"mly{str(mly_score).replace('.', '_')}"
        name = f"{name}_{mly_str}"
    
    return name

class Variables:
    CITY = os.getenv("CITY", "Sao Paulo")
    CITY_EPSG = os.getenv("CITY_EPSG", "32723")
    INNER = int(os.getenv("INNER", 5))
    OUTER = int(os.getenv("OUTER", 20))
    CITY_NORMAL = generate_table_name(CITY)
    MLY_SCORE = float(os.getenv("MLY_SCORE", 0.9))
    
    # Generate table name from city and MLY score
    CITY_TABLE = generate_table_name(CITY, MLY_SCORE)
    
    # Bounding box (west, south, east, north)
    BBOX_WEST = float(os.getenv("BBOX_WEST", "-46.9"))
    BBOX_SOUTH = float(os.getenv("BBOX_SOUTH", "-23.8"))
    BBOX_EAST = float(os.getenv("BBOX_EAST", "-46.4"))
    BBOX_NORTH = float(os.getenv("BBOX_NORTH", "-23.3"))

# ============ DATABASE ============
class Config:
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_NAME = os.getenv("DB_NAME", "gis")
    DB_USER = os.getenv("DB_USER", "moritz")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "3004")
    DB_PORT = int(os.getenv("DB_PORT", "25432"))

# ============ DIRECTORIES ============
class Directories:
    # Main city directory (use CITY_TABLE to avoid spaces)
    CITY_DIR = Path(Variables.CITY_NORMAL)
    CITY_DIR.mkdir(exist_ok=True)
    
    # Subdirectories
    CACHE_DIR = CITY_DIR / f"tiles_cache_{Variables.CITY_NOMRAL}"
    IMG_DIR = CITY_DIR / f"downloaded_images_{Variables.CITY_NORMAL}"
    FILTERED_DIR = CITY_DIR / f"filtered_{Variables.CITY_NORMAL}"
    
    # Create all directories
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Files
    CONFIG_FILE = CITY_DIR / f"{Variables.CITY_NORMAL}-config.json"
    IMGIDS_FILE = CITY_DIR / "images-to-download.txt"
    JPGS_LIST_FILE = CITY_DIR / f"list-of-jpgs-{Variables.CITY_NORMAL}.txt"
    NPZ_LIST_FILE = CITY_DIR / f"list-of-npz-outputs-{Variables.CITY_NORMAL}.txt"
    OUT_LIST_FILE = CITY_DIR / f"list-of-out-files-{Variables.CITY_NORMAL}.txt"
    DB_FILE = CITY_DIR / f"{Variables.CITY_NORMAL}-tiles-database.pkl"