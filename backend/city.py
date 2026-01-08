import csv
import re
from pathlib import Path
from sqlalchemy import text
from utils.db import get_db_connection

# City - EPSG lookup loaded from CSV
CITY_EPSG = {}
CITY_MLY_SCORE = {}
CITY_TABLES = {}

def load_cities_from_csv(csv_path: str = "available_tables_cities.csv"):
    """Load city configuration from CSV file"""
    global CITY_EPSG, CITY_MLY_SCORE, CITY_TABLES
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            table_name = row.get("table_name", "").strip()
            city = row.get("city", "").strip().lower()
            epsg = int(row.get("epsg", 0))
            mly_score = row.get("mly_score", "").strip()
            
            if city and epsg:
                CITY_EPSG[city] = epsg
                CITY_TABLES[city] = table_name
                
                # If mly_score is empty, try to extract from table_name
                if not mly_score:
                    mly_score = extract_mly_score(table_name)
                
                CITY_MLY_SCORE[city] = mly_score
    
    print(f"Loaded {len(CITY_EPSG)} cities from CSV")
    # Debug: print loaded scores
    for city, score in CITY_MLY_SCORE.items():
        print(f"  {city}: {score if score else '(empty)'}")

def extract_mly_score(table_name: str) -> str:
    """
    Extract mly score from table name and convert to decimal.
    Example: 'berlin_mly0_5_035' -> '0.5'
    """
    if not table_name:
        return ""
    
    # Look for pattern like mly0_5 (stops at underscore before numbers like 035)
    pattern = r'mly(\d_\d)(?:_|$)'
    match = re.search(pattern, table_name.lower())
    if match:
        # Extract the number part and replace underscore with decimal point
        score_raw = match.group(1)  # e.g., '0_5'
        score = score_raw.replace('_', '.')  # e.g., '0.5'
        print(f"  Extracted '{score}' from '{table_name}'")
        return score
    
    print(f"  WARNING: Could not extract mly score from '{table_name}'")
    return ""

def supported_cities():
    """Return list of supported cities from CSV"""
    if not CITY_EPSG:
        raise ValueError("No cities loaded. Ensure CSV file exists and is loaded.")
    return list(CITY_EPSG.keys())

def get_epsg(city: str) -> int:
    """Get EPSG code for a city"""
    city_lower = city.lower()
    if city_lower not in CITY_EPSG:
        raise ValueError(f"City '{city}' not found in configuration")
    return CITY_EPSG[city_lower]

def get_mly_score(city: str) -> str:
    """Get MLY score for a city"""
    city_lower = city.lower()
    score = CITY_MLY_SCORE.get(city_lower, "")
    
    if not score:
        print(f"WARNING: No MLY score found for city '{city}'. Using empty string.")
    
    return score

def is_valid_city(city: str) -> bool:
    """Check if city is supported"""
    return city.lower() in CITY_EPSG

def fetch_tables():
    """Fetch all tables from database"""
    query = text("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_type = 'BASE TABLE';
    """)
    engine = get_db_connection()
    with engine.connect() as conn:
        result = conn.execute(query)
        tables = [row[0] for row in result.fetchall()]
    return tables

def match_city(table_name: str):
    """Match table name to a city (more flexible)"""
    table_name = table_name.lower()
    for city in supported_cities():
        if city in table_name:
            return city
    return None

def get_table_name(city: str) -> str:
    """Get full table name for a city"""
    city_lower = city.lower()
    return CITY_TABLES.get(city_lower, "")

def main():
    """Generate available_tables_cities.csv from database tables"""
    CITY_EPSG_FALLBACK = {
        "berlin": 32633,
        "sao_paulo": 32723,
        "washington": 32610,
        "sydney": 32756,
        "cape_town": 32734,
        "taipei": 32651,
    }
    
    print("Fetching tables from database...")
    tables = fetch_tables()
    print(f"Found {len(tables)} tables")
    
    rows = []
    
    for table in tables:
        table_lower = table.lower()
        for city, epsg in CITY_EPSG_FALLBACK.items():
            if city in table_lower:
                # Extract mly_score from table name
                mly_score = extract_mly_score(table)
                
                rows.append({
                    "table_name": table,
                    "city": city,
                    "epsg": epsg,
                    "mly_score": mly_score,
                })
                break
    
    print(f"\nWriting {len(rows)} rows to CSV...")
    
    # Write CSV with new field
    output_file = "available_tables_cities.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["table_name", "city", "epsg", "mly_score"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"CSV written to {output_file}")

if __name__ == "__main__":
    main()
else:
    # Auto-load CSV when imported
    try:
        load_cities_from_csv()
    except FileNotFoundError:
        print("Warning: CSV file not found. Run 'python city.py' to generate it.")