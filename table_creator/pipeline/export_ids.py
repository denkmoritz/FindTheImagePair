#!/usr/bin/env python3
"""
Step 1: Export image IDs from PostgreSQL database for downloading
"""
import sys
from pathlib import Path
# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config, Variables, Directories
from sqlalchemy import create_engine, text
import pandas as pd

def get_db_connection():
    url = f"postgresql+psycopg2://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}"
    engine = create_engine(url)
    return engine

def create_materialized_view(city_table, city_epsg, inner, outer):
    view_name = f"{city_table}_slice"
    table_name = city_table
    index_name = f"{view_name}_gist"
    
    QUERY = f"""
    DROP MATERIALIZED VIEW IF EXISTS {view_name};
    CREATE MATERIALIZED VIEW {view_name} AS
    SELECT
        uuid,
        orig_id_x,
        geometry_comp_{city_epsg},
        comp_lat,
        mly_computed_compass_angle,
        comp_lon,
        ST_Difference(
            ST_Difference(
                ST_Buffer(geometry_comp_{city_epsg}, {outer}),
                ST_Buffer(geometry_comp_{city_epsg}, {inner})
            ),
            ST_Buffer(
                ST_MakeLine(
                    ST_SetSRID(ST_MakePoint(
                        ST_X(geometry_comp_{city_epsg})+{outer}*SIN(RADIANS(heading)),
                        ST_Y(geometry_comp_{city_epsg})+{outer}*COS(RADIANS(heading))
                    ), {city_epsg}),
                    ST_SetSRID(ST_MakePoint(
                        ST_X(geometry_comp_{city_epsg})-{outer}*SIN(RADIANS(heading)),
                        ST_Y(geometry_comp_{city_epsg})-{outer}*COS(RADIANS(heading))
                    ), {city_epsg})
                ),
                {inner}, 'endcap=square join=mitre')
        ) AS slice_geom
    FROM {table_name}
    WHERE mly_quality_score >= {Variables.MLY_SCORE};
    DROP INDEX IF EXISTS {index_name};
    CREATE INDEX {index_name} ON {view_name} USING gist (slice_geom);
    """
    
    engine = get_db_connection()
    with engine.connect() as conn:
        conn.execute(text(QUERY))
        conn.commit()

def export_image_ids(city_table, city_epsg, output_file=None):
    if output_file is None:
        output_file = str(Directories.IMGIDS_FILE)
    
    engine = get_db_connection()
    view_name = f"{city_table}_slice"
    table_name = city_table
    
    QUERY = f"""
    WITH high_q AS (
        SELECT *
        FROM {table_name}
        WHERE mly_quality_score >= {Variables.MLY_SCORE}
    )
    SELECT a.orig_id_x AS id, b.orig_id_x AS relation_id
    FROM {view_name} a
    JOIN high_q b
    ON b.uuid > a.uuid
    AND b.geometry_comp_{city_epsg} && a.slice_geom
    AND ST_Within(b.geometry_comp_{city_epsg}, a.slice_geom)
    AND LEAST(ABS(a.mly_computed_compass_angle - b.mly_computed_compass_angle), 360 - ABS(a.mly_computed_compass_angle - b.mly_computed_compass_angle)) <= 45
    AND b.is_pano IS NOT NULL;
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(QUERY, conn)
    
    image_ids = pd.unique(df[['id','relation_id']].values.ravel()).tolist()
    
    with open(output_file, "w") as f:
        f.write("\n".join(map(str, image_ids)))
    
    print(f"Exported {len(image_ids)} image IDs -> {output_file}")
    return len(image_ids)

if __name__ == '__main__':
    city_table = Variables.CITY_TABLE
    city_epsg = Variables.CITY_EPSG
    inner = Variables.INNER
    outer = Variables.OUTER
    output_file = str(Directories.IMGIDS_FILE)
    
    print(f"Generating spatial slice view for {Variables.CITY.upper()} (EPSG:{city_epsg})")
    create_materialized_view(city_table, city_epsg, inner, outer)
    print(f"Created materialized view: {city_table}_slice\n")
    
    print(f"Exporting image relations inside viewing slice...")
    count = export_image_ids(city_table, city_epsg, output_file)
    print(f"Done. {count} total unique images exported.\n")