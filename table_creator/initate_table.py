"""
Mapillary Data Processing Script
Fetches and updates Mapillary metadata for a specified city from global_streetscapes table.
"""

import os
import requests
import pandas as pd
from getpass import getpass
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from tqdm import tqdm
from dotenv import load_dotenv
import re
from config import Config, Variables

# ============================================================================
# CONFIGURATION - Modify these values as needed
# ============================================================================

CITY_NAME = Variables.CITY  # City name (must match city_ascii in global_streetscapes)
TABLE_NAME = Variables.CITY_TABLE  # Table name (lowercase, underscores, no spaces)
WGS84_CRS = 4326  # WGS84 coordinate reference system
UTM_CRS = Variables.CITY_EPSG  # UTM zone for the city (e.g., 32734 for Cape Town UTM 34S)

# Database connection settings
DB_HOST = Config.DB_HOST
DB_NAME = Config.DB_NAME
DB_USER = Config.DB_USER
DB_PORT = Config.DB_PORT

# Data quality filters
MAX_HEADING_DIFF = 10

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_table_name(city_name: str) -> str:
    """
    Generate a valid PostgreSQL table name from city name.
    Converts to lowercase and replaces spaces/special chars with underscores.
    
    Examples:
        "Cape Town" -> "cape_town"
        "New York City" -> "new_york_city"
        "São Paulo" -> "sao_paulo"
    """
    # Convert to lowercase
    name = city_name.lower()
    # Replace spaces and special characters with underscores
    name = re.sub(r'[^\w]+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Replace multiple underscores with single underscore
    name = re.sub(r'_+', '_', name)
    return name

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def setup_database_connection():
    """Establish database connection with user credentials."""
    password = getpass("Enter your password: ")
    encoded_password = quote_plus(password)
    connection_string = f"postgresql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(connection_string)
    print("Database connection established")
    return engine

# ============================================================================
# MAPILLARY API FUNCTIONS
# ============================================================================

def fetch_mapillary_metadata(img_id: str, token: str, fields: str) -> dict | None:
    """
    Fetch metadata from Mapillary API for a given image ID.
    
    Args:
        img_id: Mapillary image ID
        token: Mapillary API access token
        fields: Comma-separated list of fields to retrieve
    
    Returns:
        Dictionary with requested fields or None if request fails
    """
    url = (
        f"https://graph.mapillary.com/{img_id}"
        f"?access_token={token}"
        f"&fields={fields}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching {img_id}: {e}")
        return None

# ============================================================================
# TABLE CREATION AND SETUP
# ============================================================================

def create_city_table(engine, table_name: str):
    """Create city-specific table from global_streetscapes."""
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        conn.execute(text(f"""
            CREATE TABLE {table_name} AS
            SELECT *
            FROM global_streetscapes
            WHERE city_ascii = :city
            AND ABS(heading - mly_computed_compass_angle) <= :heading_diff
            AND mly_quality_score >= {Variables.MLY_SCORE}
        """), {
            "city": CITY_NAME,
            "heading_diff": MAX_HEADING_DIFF
        })
    print(f"Created table '{table_name}' for {CITY_NAME}")

def setup_geometry_columns(engine, table_name: str):
    """Add and populate UTM geometry column."""
    geom_col = f"geometry_{UTM_CRS}"
    
    with engine.begin() as conn:
        # Add UTM geometry column
        conn.execute(text(f"""
            ALTER TABLE {table_name} 
            ADD COLUMN IF NOT EXISTS {geom_col} geometry(Geometry, {UTM_CRS})
        """))
        
        # Transform original geometry to UTM
        conn.execute(text(f"""
            UPDATE {table_name}
            SET {geom_col} = ST_Transform(geometry, {UTM_CRS})
        """))
        
        # Create spatial index
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_spgist_{geom_col} 
            ON {table_name} USING SPGIST ({geom_col})
        """))
    
    print(f"Added UTM geometry column and spatial index")

def add_metadata_columns(engine, table_name: str):
    """Add columns for Mapillary metadata."""
    with engine.begin() as conn:
        # Add computed coordinate columns
        conn.execute(text(f"""
            ALTER TABLE {table_name}
            ADD COLUMN IF NOT EXISTS comp_lat double precision,
            ADD COLUMN IF NOT EXISTS comp_lon double precision
        """))
        
        # Add is_pano column
        conn.execute(text(f"""
            ALTER TABLE {table_name}
            ADD COLUMN IF NOT EXISTS is_pano boolean DEFAULT false
        """))
        
        # Add sequence_id column
        conn.execute(text(f"""
            ALTER TABLE {table_name}
            ADD COLUMN IF NOT EXISTS sequence_id text
        """))
        
        # Add computed geometry columns
        conn.execute(text(f"""
            ALTER TABLE {table_name}
            ADD COLUMN IF NOT EXISTS geom_comp geometry(Point, {WGS84_CRS})
        """))
        
        conn.execute(text(f"""
            ALTER TABLE {table_name}
            ADD COLUMN IF NOT EXISTS geometry_comp_{UTM_CRS} geometry(Geometry, {UTM_CRS})
        """))
    
    print(f"Added metadata columns")

# ============================================================================
# DATA FETCHING AND UPDATING
# ============================================================================

def fetch_and_update_all_metadata(engine, table_name: str, token: str):
    """
    Fetch all metadata (computed coordinates, is_pano, sequence_id) from Mapillary 
    in a single pass and update database.
    """
    # Get IDs that need any metadata update
    with engine.connect() as conn:
        ids_to_update = conn.execute(text(f"""
            SELECT orig_id_x
            FROM {table_name}
            WHERE comp_lat IS NULL 
               OR comp_lon IS NULL
               OR is_pano IS false
               OR sequence_id IS NULL
        """)).scalars().all()
    
    print(f"Found {len(ids_to_update)} images needing metadata updates")
    
    if not ids_to_update:
        print("All images already have complete metadata")
        return
    
    # Fetch all metadata in one API call per image
    values = []
    for img_id in tqdm(ids_to_update, desc="Fetching Mapillary metadata"):
        data = fetch_mapillary_metadata(
            img_id, 
            token, 
            "id,computed_geometry,is_pano,sequence"
        )
        
        if data:
            # Extract computed geometry coordinates
            coords = data.get("computed_geometry", {}).get("coordinates")
            lon, lat = None, None
            if coords and len(coords) == 2:
                lon, lat = coords
            
            # Extract is_pano
            is_pano = data.get("is_pano", False)
            
            # Extract sequence_id
            sequence_id = data.get("sequence")
            
            values.append({
                "img_id": img_id,
                "lon": lon,
                "lat": lat,
                "is_pano": is_pano,
                "sequence_id": sequence_id
            })
    
    print(f"Successfully fetched metadata for {len(values)} images")
    
    # Bulk update all fields
    if values:
        with engine.begin() as conn:
            conn.execute(
                text(f"""
                    UPDATE {table_name}
                    SET comp_lon = :lon,
                        comp_lat = :lat,
                        is_pano = :is_pano,
                        sequence_id = :sequence_id
                    WHERE orig_id_x = :img_id
                """),
                values
            )
        print(f"Updated {len(values)} rows with all metadata")
        
        # Update computed geometry columns
        with engine.begin() as conn:
            conn.execute(text(f"""
                UPDATE {table_name} 
                SET geom_comp = ST_SetSRID(ST_MakePoint(comp_lon, comp_lat), {WGS84_CRS})
                WHERE comp_lon IS NOT NULL AND comp_lat IS NOT NULL AND geom_comp IS NULL
            """))
            
            conn.execute(text(f"""
                UPDATE {table_name}
                SET geometry_comp_{UTM_CRS} = ST_Transform(geom_comp, {UTM_CRS})
                WHERE geom_comp IS NOT NULL AND geometry_comp_{UTM_CRS} IS NULL
            """))
        
        print(f"Updated computed geometry columns")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    # Use provided table name or auto-generate from city name
    table_name = TABLE_NAME if TABLE_NAME else generate_table_name(CITY_NAME)
    
    print("=" * 70)
    print(f"Mapillary Data Processing for {CITY_NAME}")
    print(f"Table: {table_name}")
    print(f"UTM CRS: {UTM_CRS} | WGS84 CRS: {WGS84_CRS}")
    print("=" * 70)
    
    # Load Mapillary token from .env
    load_dotenv()
    TOKEN = os.getenv("MAPILLARY_TOKEN")
    if not TOKEN:
        raise RuntimeError("MAPILLARY_TOKEN not set. Put MAPILLARY_TOKEN=MLY|... in your env or .env")
    
    # Setup database connection
    engine = setup_database_connection()
    
    # Execute pipeline
    print("\n--- Phase 1: Table Creation ---")
    create_city_table(engine, table_name)
    
    print("\n--- Phase 2: Geometry Setup ---")
    setup_geometry_columns(engine, table_name)
    
    print("\n--- Phase 3: Metadata Columns ---")
    add_metadata_columns(engine, table_name)
    
    print("\n--- Phase 4: Fetch All Metadata ---")
    fetch_and_update_all_metadata(engine, table_name, TOKEN)
    
    print("\n" + "=" * 70)
    print(f"Processing complete for {CITY_NAME}")
    print("=" * 70)

if __name__ == "__main__":
    main()