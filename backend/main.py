from config import *
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
import sqlite3

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from utils.plot_slice import plot_slice
from utils.query import run_query
from utils.create_slice import create_materialized_view
from utils.download import download_pairs
from utils.helper_db import *
from utils.pydantic_models import *

from city import *
from config import IMAGES_DIR

PKL_PATH = Path("latest_query.pkl")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: code before yield
    print("Loading cities from CSV...")
    load_cities_from_csv()
    
    yield
    
    # Shutdown: code after yield
    print("Cleaning up resources...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
    allow_credentials = True
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/available-cities/")
async def get_available_cities():
    """
    Return available cities with their table names.
    """
    csv_path = "available_tables_cities.csv"
    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Cities configuration file not found"
        )
    
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cities_list = []
            
            for row in reader:
                cities_list.append({
                    "city": row["city"],
                    "table_name": row["table_name"]
                })
        
        return JSONResponse(content=cities_list)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading cities file: {str(e)}"
        )

@app.post("/plot-slice/")
async def plot(data: PlotRequest):
    if data.inner_buffer >= data.outer_buffer:
        raise HTTPException(status_code=400, detail="Outer Ring must be greater than Inner Ring.")
    else:
        image = plot_slice(data.inner_buffer, data.outer_buffer)
        return JSONResponse(content={"image": f"data:image/png;base64,{image}"})

@app.post("/query/")
async def query(data: PlotRequest):
    # Validate city
    if not data.city:
        raise HTTPException(status_code=400, detail="City parameter is required")
    
    city = data.city.lower()
    
    # Validate against CSV-loaded cities
    if not is_valid_city(city):
        available = supported_cities()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid city. Must be one of: {available}"
        )
    
    epsg = get_epsg(city)
    mly_score = get_mly_score(city)
    full_table_name = get_table_name(city)
    
    # Extract location data
    lat = None
    lng = None
    radius_m = None
    
    if data.area:
        lng, lat = data.area.center
        radius_m = data.area.radius_m
    
    # Create materialized view
    create_materialized_view(
        city=full_table_name,
        inner=data.inner_buffer,
        outer=data.outer_buffer,
        epsg=epsg,
        mly_score=mly_score,
        lat=lat,
        lng=lng,
        radius_m=radius_m
    )
    
    # Run query and save results
    count, df = run_query(city=full_table_name, epsg=epsg)
    df.attrs['city'] = city
    df.to_pickle("latest_query.pkl")
    
    save_dir = Path("queries")
    save_dir.mkdir(exist_ok=True)
    
    # Helper function for safe numeric formatting
    def safe_num(val, decimals=5):
        if val is None:
            return "none"
        return str(round(val, decimals)).replace(".", "_")
    
    # Generate filename - optionally include mly_score in filename
    inner_str = safe_num(data.inner_buffer)
    outer_str = safe_num(data.outer_buffer)
    lat_str = safe_num(lat)
    lng_str = safe_num(lng)
    radius_str = safe_num(radius_m, 2)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Optional: Add mly_score to filename
    mly_str = mly_score if mly_score else "nomly"
    base_name = f"city_{city}_mly_{mly_str}_count_{count}_inner_{inner_str}_outer_{outer_str}_lat_{lat_str}_lng_{lng_str}_r_{radius_str}_{timestamp}"
    
    # Save as CSV
    csv_path = save_dir / f"{base_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Query saved to: {csv_path}")
    
    return JSONResponse(content={
        "count": int(count),
        "csv_path": str(csv_path),
        "city": city,
        "epsg": epsg,
        "mly_score": mly_score
    })

@app.post("/download/")
async def download(body: dict = None):
    """
    Downloads images for the current query.
    Expects optional body with city name, or reads from latest_query.pkl metadata.
    
    Pairs format: (fetch_id=orig_id, dest_name=uuid)
    - Searches for existing images in: final_project/table_creator/{city}/downloaded_images_{city}/*/{orig_id}.jpg
    - Copies found images to: final_project/backend/images/{city}/{uuid}.jpg
    - Downloads missing images from Mapillary
    """
    pkl = Path("latest_query.pkl")
    
    if not pkl.exists():
        raise HTTPException(status_code=400, detail="No latest_query.pkl found.")
    
    df = pd.read_pickle(pkl)
    
    if df.empty:
        raise HTTPException(status_code=400, detail="No rows found in latest_query.pkl")
    
    # Try to get city from request body, or from dataframe metadata
    city = None
    if body and "city" in body:
        city = body["city"].lower()
    elif hasattr(df, 'attrs') and 'city' in df.attrs:
        city = df.attrs['city']
    else:
        raise HTTPException(status_code=400, detail="City information not found. Please re-run query.")
    
    required = ["uuid", "relation_uuid", "orig_id", "relation_orig_id"]
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
    
    # Build pairs: (fetch_id=orig_id, dest_name=uuid)
    # The orig_id is the Mapillary image ID used to search for existing downloads
    # The uuid is the destination filename
    pairs = []
    
    for _, row in df.iterrows():
        orig_id = str(row["orig_id"]).strip() if pd.notna(row["orig_id"]) else None
        uuid = str(row["uuid"]).strip() if pd.notna(row["uuid"]) else None
        
        if orig_id and uuid:
            pairs.append((orig_id, uuid))
            print(f"Image pair: {orig_id} -> {uuid}")
        
        relation_orig_id = str(row["relation_orig_id"]).strip() if pd.notna(row["relation_orig_id"]) else None
        relation_uuid = str(row["relation_uuid"]).strip() if pd.notna(row["relation_uuid"]) else None
        
        if relation_orig_id and relation_uuid:
            pairs.append((relation_orig_id, relation_uuid))
            print(f"Image pair: {relation_orig_id} -> {relation_uuid}")
    
    if not pairs:
        raise HTTPException(status_code=400, detail="No valid image pairs found in dataframe.")
    
    print(f"\nDownloading {len(pairs)} images for {city}...")
    
    # Perform the download with city context
    result = download_pairs(pairs, city=city)
    
    return JSONResponse(content=result)

@app.post("/pairs/")
def get_pairs(body: dict):
    """
    Returns paginated pairs from latest_query.pkl.
    Pagination is numeric-based (cursor = row index).
    Now includes coordinate data for map display.
    """
    limit = body.get("limit", 50)
    cursor = body.get("cursor", 0)
    
    pkl = Path("latest_query.pkl")
    if not pkl.exists():
        raise HTTPException(status_code=400, detail="No latest_query.pkl found.")
    
    df = pd.read_pickle(pkl)
    if df.empty:
        raise HTTPException(status_code=400, detail="No rows found in latest_query.pkl")
    
    # Get city from dataframe metadata or request
    city = body.get("city")
    if not city and hasattr(df, 'attrs') and 'city' in df.attrs:
        city = df.attrs['city']
    
    required = ["uuid", "relation_uuid", "orig_id", "relation_orig_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
    
    # Check which pairs are already liked
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT uuid_1, uuid_2 FROM liked")
    liked_pairs = {(row[0], row[1]) for row in cur.fetchall()}
    con.close()
    
    start = int(cursor)
    end = start + int(limit)
    total = len(df)
    slice_df = df.iloc[start:end]
    
    items = []
    for _, row in slice_df.iterrows():
        uuid1 = row["uuid"]
        uuid2 = row["relation_uuid"]
        
        # Build left image data with coordinates
        left = {
            "uuid": uuid1, 
            "orig_id": row["orig_id"]
        }
        
        # Add lat/lng from lat_1 and lon_1
        if "lat_1" in row and pd.notna(row["lat_1"]):
            left["lat"] = float(row["lat_1"])
        if "lon_1" in row and pd.notna(row["lon_1"]):
            left["lng"] = float(row["lon_1"])
        
        # Build right image data with coordinates
        right = {
            "uuid": uuid2, 
            "orig_id": row["relation_orig_id"]
        }
        
        # Add lat/lng from lat_2 and lon_2
        if "lat_2" in row and pd.notna(row["lat_2"]):
            right["lat"] = float(row["lat_2"])
        if "lon_2" in row and pd.notna(row["lon_2"]):
            right["lng"] = float(row["lon_2"])
        
        # Check if this pair is liked
        is_liked = (uuid1, uuid2) in liked_pairs
        
        items.append({
            "id": f"{uuid1}|{uuid2}",
            "left": left,
            "right": right,
            "liked": is_liked,
            "distance": row.get("distance_meters", row.get("distance", None)),
            "city": city
        })
    
    next_cursor = end if end < total else None
    return {
        "items": items,
        "total": total,
        "nextCursor": next_cursor,
        "city": city
    }

@app.get("/image")
def get_image(uuid: str, city: str = None):
    """
    Serves an image from IMAGES_DIR by UUID.
    Searches city folders if city not specified.
    """
    if city:
        # Look in specific city folder
        city = city.lower()
        img_path = Path(IMAGES_DIR) / city / f"{uuid}.jpg"
        if img_path.exists():
            return FileResponse(img_path, media_type="image/jpeg")
        raise HTTPException(status_code=404, detail=f"Image {uuid}.jpg not found in {city}")
    
    # Search all city folders
    cities = ["berlin", "paris", "washington", "singapore"]
    for city_name in cities:
        img_path = Path(IMAGES_DIR) / city_name / f"{uuid}.jpg"
        if img_path.exists():
            return FileResponse(img_path, media_type="image/jpeg")
    
    # Fallback: check root IMAGES_DIR (for backward compatibility with old berlin images)
    img_path = Path(IMAGES_DIR) / f"{uuid}.jpg"
    if img_path.exists():
        return FileResponse(img_path, media_type="image/jpeg")
    
    raise HTTPException(status_code=404, detail=f"Image {uuid}.jpg not found in any folder")

@app.post("/like/")
def toggle_like(data: dict):
    """
    Toggles like status for a pair.
    Expected body: {
        "id": "uuid1|uuid2",
        "liked": true/false,
        "city": "berlin" (optional, will try to get from dataframe if not provided)
    }
    """
    pair_id = data.get("id")
    liked = data.get("liked", True)
    city = data.get("city")
    
    if not pair_id or "|" not in pair_id:
        raise HTTPException(status_code=400, detail="Invalid pair id format. Expected 'uuid1|uuid2'")
    
    uuid_1, uuid_2 = pair_id.split("|", 1)
    
    # Load the dataframe to get full pair details
    pkl = Path("latest_query.pkl")
    if not pkl.exists():
        raise HTTPException(status_code=400, detail="No latest_query.pkl found.")
    
    df = pd.read_pickle(pkl)
    
    # Get city from dataframe if not provided
    if not city:
        if hasattr(df, 'attrs') and 'city' in df.attrs:
            city = df.attrs['city']
        else:
            # Try to infer from query results or default to None
            city = None
    
    # Find the matching row
    row = df[(df["uuid"] == uuid_1) & (df["relation_uuid"] == uuid_2)]
    
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Pair {pair_id} not found in query results")
    
    row = row.iloc[0]
    orig_id_1 = row.get("orig_id")
    orig_id_2 = row.get("relation_orig_id")
    distance = row.get("distance_meters")
    lat_1 = row.get("lat_1")
    lon_1 = row.get("lon_1")
    lat_2 = row.get("lat_2")
    lon_2 = row.get("lon_2")
    
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    
    try:
        if liked:
            # Add like (avoid duplicates using ON CONFLICT)
            cur.execute("""
                INSERT INTO liked (uuid_1, uuid_2, orig_id_1, orig_id_2, distance, city, created_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(uuid_1, uuid_2) DO UPDATE SET
                    distance = excluded.distance,
                    city = excluded.city,
                    created_at = CURRENT_TIMESTAMP
            """, (uuid_1, uuid_2, orig_id_1, orig_id_2, distance, city))
        else:
            # Remove like
            cur.execute("""
                DELETE FROM liked WHERE uuid_1 = ? AND uuid_2 = ?
            """, (uuid_1, uuid_2))
        
        con.commit()
        return JSONResponse(content={"success": True, "liked": liked})
    except Exception as e:
        con.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()


@app.get("/liked/")
def get_liked(city: str = None):
    """
    Returns all liked pairs, optionally filtered by city.
    Query parameter: city (optional) - filter results by city name
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    
    # Check which columns exist
    cur.execute("PRAGMA table_info(liked)")
    columns = [row[1] for row in cur.fetchall()]
    has_created_at = "created_at" in columns
    has_city = "city" in columns
    
    # Build query based on available columns and filters
    select_cols = ["uuid_1", "uuid_2", "orig_id_1", "orig_id_2", "distance"]
    if has_city:
        select_cols.append("city")
    if has_created_at:
        select_cols.append("created_at")
    
    query = f"SELECT {', '.join(select_cols)} FROM liked"
    params = []
    
    if city and has_city:
        query += " WHERE city = ?"
        params.append(city.lower())
    
    if has_created_at:
        query += " ORDER BY created_at DESC"
    
    cur.execute(query, params)
    rows = cur.fetchall()
    con.close()
    
    items = []
    for row in rows:
        # Handle potential bytes values
        def safe_value(val):
            if isinstance(val, bytes):
                try:
                    return val.decode('utf-8')
                except:
                    return str(val)
            return val
        
        item = {
            "uuid_1": safe_value(row[0]),
            "uuid_2": safe_value(row[1]),
            "orig_id_1": safe_value(row[2]),
            "orig_id_2": safe_value(row[3]),
            "distance": row[4] if row[4] is not None else None
        }
        
        col_idx = 5
        if has_city and len(row) > col_idx:
            item["city"] = safe_value(row[col_idx])
            col_idx += 1
        
        if has_created_at and len(row) > col_idx:
            item["created_at"] = safe_value(row[col_idx])
        
        items.append(item)
    
    return {"items": items, "total": len(items)}