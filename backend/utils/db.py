import time
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from config import DB

def get_db_connection(retries=30, delay=2):
    """Retry connecting to the database until it is ready."""
    url = (
        f"postgresql+psycopg://{DB.DB_USER}:"
        f"{DB.DB_PASSWORD}@{DB.DB_HOST}:"
        f"{DB.DB_PORT}/{DB.DB_NAME}"
    )

    for attempt in range(1, retries + 1):
        try:
            engine = create_engine(url)
            with engine.connect() as conn:
                pass
            print("Database connection established.")
            return engine
        except OperationalError:
            print(
                f"Database not ready yet (attempt {attempt}/{retries}), retrying in {delay}s..."
            )
            time.sleep(delay)

    raise RuntimeError("Could not connect to database after multiple attempts")