#!/usr/bin/env python3
import sys
from pathlib import Path

# Get the parent directory (table_creator/)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import pickle
import lzma
from sqlalchemy import create_engine, text
from config import Config, Variables

def get_db_connection():
    url = f"postgresql+psycopg://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}"
    engine = create_engine(url)
    return engine

def export_tiles_database(output_file='my-tiles-database.pkl'):
    engine = get_db_connection()
    
    with engine.connect() as conn:
        # Adjust column names to match your schema
        result = conn.execute(text("""
            SELECT 
                orig_id_x::text as id,
                sequence_id::text,  
                mly_computed_compass_angle as angle,      
                comp_lat as latitude,
                comp_lon as longitude,
                is_pano
            FROM {Variables.CITY}
            WHERE is_pano IS NOT True
        """))
        
        db = {}
        for row in result:
            img_id = row.id
            db[img_id] = {
                'seqid': row.sequence_id,
                'angle': row.compass_angle if row.compass_angle is not None else 0.0,
                'lat': row.latitude,
                'lon': row.longitude,
                'is_pano': row.is_pano if row.is_pano is not None else False
            }
    
    # Save as compressed pickle
    with lzma.open(output_file, 'wb') as fp:
        pickle.dump(db, fp)
    
    print(f"Exported {len(db)} images to {output_file}")
    return len(db)

if __name__ == '__main__':
    city = Variables.CITY
    export_tiles_database(city)