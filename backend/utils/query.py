import pandas as pd
from utils.db import get_db_connection

def run_query(city, epsg):
    
    # Dynamic table and view names based on city
    view_name = f"{city}_slice"
    table_name = city
    
    QUERY = f"""
    SELECT
        a.uuid AS uuid,
        b.uuid AS relation_uuid,
        a.orig_id_x AS orig_id,
        b.orig_id_x AS relation_orig_id,
        a.mly_computed_compass_angle AS h_1,
        b.mly_computed_compass_angle AS h_2,
        a.comp_lon AS lon_1,
        b.comp_lon AS lon_2,
        a.comp_lat AS lat_1,
        b.comp_lat AS lat_2,
        a.geometry_comp_{epsg} <-> b.geometry_comp_{epsg} AS distance_meters,
        b.source_x AS source,
        CONCAT(a.uuid, '__', b.uuid) AS relation_id
    FROM {view_name} AS a
    JOIN {table_name} AS b
    ON  b.uuid > a.uuid
        AND b.geometry_comp_{epsg} && a.slice_geom
        AND ST_Within(b.geometry_comp_{epsg}, a.slice_geom)
        AND LEAST(
            ABS(a.mly_computed_compass_angle - b.mly_computed_compass_angle),
            360 - ABS(a.mly_computed_compass_angle - b.mly_computed_compass_angle)
            ) <= 45;"""
    
    engine = get_db_connection()
    with engine.connect() as conn:
        df = pd.read_sql_query(QUERY, conn)
    
    return df.shape[0], df