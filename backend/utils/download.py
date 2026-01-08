import os
from pathlib import Path
from typing import Iterable, Tuple, Dict, List
import requests
from dotenv import load_dotenv
from config import IMAGES_DIR

GRAPH_BASE = "https://graph.mapillary.com"
URL_FIELD = "thumb_original_url"

load_dotenv()
TOKEN = os.getenv("MAPILLARY_TOKEN")
if not TOKEN:
    raise RuntimeError("MAPILLARY_TOKEN not set. Put MAPILLARY_TOKEN=MLY|... in your env or .env")

def _local_path(dest_name: str, city: str) -> Path:
    """Save as images/{city}/{uuid}.jpg"""
    city_dir = IMAGES_DIR / city
    city_dir.mkdir(parents=True, exist_ok=True)
    return city_dir / f"{dest_name}.jpg"

def _find_in_existing_downloads(orig_id: str, city: str) -> Path | None:
    """
    Search for orig_id in root/table_creator/{city}/downloaded_images_{city}/*/*.jpg
    Returns Path if found, None otherwise
    """
    # Try relative path first
    existing_dir = Path("../table_creator") / city / f"downloaded_images_{city}"
    
    # Debug: print what we're searching for
    print(f"[DEBUG] Searching for {orig_id}.jpg in: {existing_dir.resolve()}", flush=True)
    
    if not existing_dir.exists():
        print(f"[DEBUG] Directory does not exist: {existing_dir.resolve()}", flush=True)
        return None
    
    # Search all subdirectories for a file matching orig_id.jpg
    search_pattern = f"{orig_id}.jpg"
    for subdir in existing_dir.glob("*"):
        if subdir.is_dir():
            image_path = subdir / search_pattern
            if image_path.exists() and image_path.stat().st_size > 0:
                print(f"[DEBUG] Found image at: {image_path.resolve()}", flush=True)
                return image_path
    
    print(f"[DEBUG] Image {orig_id}.jpg not found in any subdirectory", flush=True)
    return None

def _fetch_url_single(image_id: str) -> str | None:
    """Fetch thumb_original_url for a single image id (no batching)."""
    url = f"{GRAPH_BASE}/{image_id}?fields=id,{URL_FIELD}&access_token={TOKEN}"
    r = requests.get(url, timeout=12)
    if r.status_code != 200:
        print(f"[META-HTTP-{r.status_code}] id={image_id} body={r.text[:200]}", flush=True)
        return None
    j = r.json()
    u = j.get(URL_FIELD)
    if not u:
        print(f"[META-MISS] id={image_id} has no {URL_FIELD}", flush=True)
        return None
    return u

def download_pairs(pairs: Iterable[Tuple[str, str]], city: str) -> Dict:
    """
    pairs: (fetch_id -> Mapillary image id, dest_name -> UUID)
    city: City name (e.g., "berlin", "paris", "washington", "singapore")
    
    1. First checks root/table_creator/{city}/downloaded_images_{city}/ for existing images
    2. Copies found images to IMAGES_DIR
    3. Downloads missing images from Mapillary
    4. Saves as images/{city}/{uuid}.jpg
    """
    city = city.lower()
    
    # Create city-specific directory
    city_dir = IMAGES_DIR / city
    city_dir.mkdir(parents=True, exist_ok=True)
    
    # de-dupe + clean
    seen = set()
    uniq_pairs: List[Tuple[str, str]] = []
    for fid, dest in pairs:
        fid, dest = str(fid).strip(), str(dest).strip()
        if fid and dest and (fid, dest) not in seen:
            seen.add((fid, dest))
            uniq_pairs.append((fid, dest))
    
    # Check existing files (both in IMAGES_DIR and in table_creator)
    to_fetch = []
    skipped_existing = 0
    copied_from_cache = 0
    
    for fid, dest in uniq_pairs:
        path = _local_path(dest, city)
        
        # First check if already in IMAGES_DIR
        if path.exists() and path.stat().st_size > 0:
            skipped_existing += 1
            continue
        
        # Then check if in existing downloaded_images directory
        existing_path = _find_in_existing_downloads(fid, city)
        if existing_path:
            try:
                # Copy from existing downloads
                path.write_bytes(existing_path.read_bytes())
                copied_from_cache += 1
                print(f"Image {fid} copied from cache {existing_path} to {path}", flush=True)
                continue
            except Exception as e:
                print(f"[WARN] Failed to copy from cache for id={fid}: {e}", flush=True)
                # Fall through to fetch from Mapillary
        
        # Need to fetch from Mapillary
        to_fetch.append((fid, dest))
    
    # Download missing images from Mapillary
    downloaded = 0
    missing_meta, failed = [], []
    
    for fid, dest in to_fetch:
        # 1) per-id metadata call
        url = _fetch_url_single(fid)
        if not url:
            missing_meta.append(fid)
            continue
        
        # 2) download
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200 and resp.content:
                path = _local_path(dest, city)
                path.write_bytes(resp.content)
                downloaded += 1
                print(f"Image {fid} downloaded and saved under {path}", flush=True)
            else:
                failed.append((fid, dest))
                print(f"[FAIL] download HTTP {resp.status_code} for id={fid}", flush=True)
        except requests.RequestException as e:
            failed.append((fid, dest))
            print(f"[FAIL] exception for id={fid}: {e}", flush=True)
    
    return {
        "city": city,
        "requested_pairs": len(uniq_pairs),
        "skipped_existing": skipped_existing,
        "copied_from_cache": copied_from_cache,
        "attempted_download": len(to_fetch),
        "downloaded": downloaded,
        "missing_meta": missing_meta[:5],
        "failed": failed[:5],
        "images_dir": str(city_dir.resolve()),
    }