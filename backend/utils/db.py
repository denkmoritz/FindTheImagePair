from sqlalchemy import create_engine
from config import DB

def get_db_connection():
    url = f"postgresql+psycopg://{DB.DB_USER}:{DB.DB_PASSWORD}@{DB.DB_HOST}:{DB.DB_PORT}/{DB.DB_NAME}"
    engine = create_engine(url)
    return engine