#!/usr/bin/env python3

import psycopg2
import logging
import argparse
import traceback
from time import sleep
from io import StringIO
from config import Config, Variables

# ========= Logging Setup ========= #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("delete.log", mode="w"),
        logging.StreamHandler()
    ]
)

# ========= DB Connection ========= #
def connect_with_retries(retries=3):
    for attempt in range(1, retries + 1):
        try:
            logging.info(f"Trying to connect to database (attempt {attempt})")
            conn = psycopg2.connect(
                host=Config.DB_HOST,
                port=Config.DB_PORT,
                database=Config.DB_NAME,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD,
                sslmode="disable",
                connect_timeout=10
            )
            logging.info("Connected successfully")
            return conn

        except Exception as e:
            logging.error(f"Connection failed: {e}")
            if attempt < retries:
                logging.info("Retrying in 5 seconds...")
                sleep(5)
            else:
                logging.critical("FATAL — cannot connect after retries")
                raise

# ========= Main Logic ========= #
def delete_rejected_images(c_value: float):
    city_table = Variables.CITY_TABLE

    # normalize C value for filenames and table suffix
    c_str = f"{c_value}"
    c_suffix = c_str.replace(".", "")

    source_table = city_table
    target_table = f"{city_table}_{c_suffix}"

    file_path = (
        f"{city_table}/filtered_test/"
        f"filtered-{city_table}-C{c_str}-H0.35-F0.8.txt"
    )

    logging.info(f"Accepted ID file: {file_path}")
    logging.info(f"Source table: {source_table}")
    logging.info(f"Target table: {target_table}")

    # ======== Load accepted IDs ======== #
    with open(file_path) as f:
        accepted = [int(x.strip()) for x in f if x.strip()]

    logging.info(f"Loaded {len(accepted)} accepted IDs")

    if not accepted:
        logging.warning("Accepted ID list is empty — aborting")
        return

    # ======== Connect ======== #
    conn = connect_with_retries()

    try:
        conn.autocommit = False
        cur = conn.cursor()

        # ======== SAFETY SETTINGS ======== #
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute("SET max_parallel_workers = 0")
        cur.execute("SET enable_parallel_append = off")
        cur.execute("SET enable_parallel_hash = off")
        cur.execute("SET work_mem = '8MB'")
        cur.execute("SET maintenance_work_mem = '32MB'")

        # ======== Confirmation ======== #
        confirm = input(
            f"Type DELETE to create '{target_table}' and modify it: "
        )
        if confirm != "DELETE":
            logging.info("Operation aborted by user")
            return

        # ======== Create cloned table ======== #
        logging.info(f"Creating table {target_table} (clone of {source_table})")

        cur.execute(f"""
            CREATE TABLE {target_table}
            (LIKE {source_table} INCLUDING ALL)
        """)

        logging.info(f"Inserting data into {target_table}")
        cur.execute(f"""
            INSERT INTO {target_table}
            SELECT * FROM {source_table}
        """)

        # ======== TEMP TABLE ======== #
        logging.info("Creating TEMP table accepted_ids")

        cur.execute("""
            CREATE TEMP TABLE accepted_ids (
                id BIGINT PRIMARY KEY
            ) ON COMMIT PRESERVE ROWS
        """)

        # ======== COPY accepted IDs ======== #
        logging.info("Bulk inserting accepted IDs")

        buf = StringIO()
        buf.write("\n".join(map(str, accepted)))
        buf.seek(0)

        cur.copy_from(buf, "accepted_ids", columns=("id",))

        # ======== DELETE ======== #
        logging.info(f"Deleting rejected rows from {target_table}")

        total_deleted = 0

        cur.execute(f"""
            DELETE FROM {target_table} b
            WHERE NOT EXISTS (
                SELECT 1
                FROM accepted_ids a
                WHERE a.id = b.orig_id_x
            )
        """)

        total_deleted = cur.rowcount
        logging.info(f"Deleted {total_deleted} rows")

        conn.commit()
        logging.info(
            f"DELETE committed — total rows removed from {target_table}: {total_deleted}"
        )

    except Exception as e:
        logging.error("ERROR OCCURRED")
        logging.error(e)
        logging.error(traceback.format_exc())

        try:
            conn.rollback()
            logging.warning("Rollback successful")
        except Exception:
            logging.warning("Rollback skipped — lost connection")

    finally:
        try:
            cur.close()
        except Exception:
            pass

        conn.close()
        logging.info("Connection closed")

# ========= Entry Point ========= #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clone table and delete rejected images based on accepted ID list"
    )

    parser.add_argument(
        "-C",
        "--c-value",
        type=float,
        required=True,
        help="C threshold value (e.g. 0.45)"
    )

    args = parser.parse_args()

    delete_rejected_images(args.c_value)