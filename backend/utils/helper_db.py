import sqlite3
from pathlib import Path

DB_PATH = "liked.db"

def init_db():
    """Initialize the database with a fresh schema including city support"""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    
    # Create table with all columns from the start
    cur.execute("""
        CREATE TABLE IF NOT EXISTS liked (
            uuid_1 TEXT NOT NULL,
            uuid_2 TEXT NOT NULL,
            orig_id_1 INTEGER,
            orig_id_2 INTEGER,
            distance FLOAT,
            city TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (uuid_1, uuid_2)
        )
    """)
    
    con.commit()
    con.close()
    print("Database initialized successfully")

# Initialize database when module is imported
init_db()