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
    Search for orig_id in images/{city}/downloaded_images_{city}/*/*.jpg
    Returns Path if found, None otherwise
    """
    existing_dir = Path("images") / city / f"downloaded_images_{city}"
    
    if not existing_dir.exists():
        return None
    
    # Search for {orig_id}.jpg in any subdirectory
    matches = list(existing_dir.glob(f"*/{orig_id}.jpg"))
    if matches:
        image_path = matches[0]
        if image_path.stat().st_size > 0:
            return image_path
    
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
    
    1. Checks images/{city}/{uuid}.jpg (for images from previous tables)
    2. Checks images/{city}/downloaded_images_{city}/*/{orig_id}.jpg (for cached downloads)
    3. Copies found cache images to IMAGES_DIR and deletes from cache
    4. Downloads missing images from Mapillary
    5. On future runs with overlapping UUIDs, images are found by UUID directly
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
    
    # Check existing files (multiple search strategies)
    to_fetch = []
    skipped_existing = 0
    copied_from_cache = 0
    
    for fid, dest in uniq_pairs:
        path = _local_path(dest, city)
        
        # Strategy 1: Check if already in IMAGES_DIR by UUID
        if path.exists() and path.stat().st_size > 0:
            skipped_existing += 1
            continue
        
        # Strategy 2: Check if in cache organized by orig_id
        cached_path = _find_in_existing_downloads(fid, city)
        if cached_path:
            try:
                # Copy from cache to main IMAGES_DIR (by UUID)
                path.write_bytes(cached_path.read_bytes())
                copied_from_cache += 1
                print(f"Image {fid} copied from cache {cached_path} to {path}", flush=True)
                
                # Delete from cache
                cached_path.unlink()
                print(f"Deleted from cache: {cached_path}", flush=True)
                continue
            except Exception as e:
                print(f"[WARN] Failed to copy from cache for id={fid}: {e}", flush=True)
                # Fall through to fetch from Mapillary
        
        # Strategy 3: Need to fetch from Mapillary
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