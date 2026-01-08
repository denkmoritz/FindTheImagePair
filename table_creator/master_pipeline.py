#!/usr/bin/env python3
"""
Master pipeline script using config.py for all settings.
Usage: python3 master_pipeline.py
  (All settings are defined in config.py - update config.py to change city/bbox/epsg)
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Import config
from config import Config, Variables, Directories


def run_command(cmd, description):
    """Execute a shell command and handle errors."""
    print(f"\n{'='*70}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(cmd, check=True, shell=isinstance(cmd, str))
        print(f"Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in {description}: {e}")
        sys.exit(1)


def create_config_json():
    """Create the bounding box config JSON file."""
    config = {
        "bounding_box": {
            "west": Variables.BBOX_WEST,
            "south": Variables.BBOX_SOUTH,
            "east": Variables.BBOX_EAST,
            "north": Variables.BBOX_NORTH
        },
        "tile_cache_dir": str(Directories.CACHE_DIR),
        "seqdir": str(Directories.IMG_DIR)
    }
    
    with open(Directories.CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created config file: {Directories.CONFIG_FILE}")


def main():
    print(f"\n{'#'*70}")
    print(f"# STARTING PIPELINE FOR: {Variables.CITY.upper()}")
    print(f"# EPSG: {Variables.CITY_EPSG}")
    print(f"# Bounding Box: W={Variables.BBOX_WEST} S={Variables.BBOX_SOUTH} E={Variables.BBOX_EAST} N={Variables.BBOX_NORTH}")
    print(f"# Working Directory: {Directories.CITY_DIR.absolute()}")
    print(f"{'#'*70}")
    
    steps = []
    
    # Step 1: Export IDs
    steps.append((1, "Export IDs", [
        "python3", "pipeline/export_ids.py",
        "--city", Variables.CITY_TABLE,
        "--epsg", Variables.CITY_EPSG,
        "--output-file", str(Directories.IMGIDS_FILE)
    ]))
    
    # Step 2: Create config JSON
    steps.append((2, "Create config JSON", create_config_json))
    
    # Step 3: Download Mapillary images
    steps.append((3, "Download Mapillary JPEGs", [
        "python3", "pipeline/mapillary_jpg_download.py",
        "-c", str(Directories.CONFIG_FILE),
        "--imgid-file", str(Directories.IMGIDS_FILE)
    ]))
    
    # Step 4: Make tiles database
    steps.append((4, "Create tiles database", [
        "python3", "pipeline/make_tiles_db.py",
        "-o", str(Directories.DB_FILE),
        "--seqs", str(Directories.IMG_DIR),
        str(Directories.CACHE_DIR)
    ]))
    
    # Step 5: Find all JPG files
    steps.append((5, "Create list of JPG files", 
        f"find {Directories.IMG_DIR} -name '*.jpg' > {Directories.JPGS_LIST_FILE}"
    ))
    
    # Step 6: Torch segmentation
    steps.append((6, "Process segmentation with torch", [
        "python3", "pipeline/torch_segm_images.py",
        "-v", "-F",
        "--output-filelist", str(Directories.NPZ_LIST_FILE),
        str(Directories.JPGS_LIST_FILE)
    ]))
    
    # Step 7: Process segmentation
    steps.append((7, "Torch process segmentation", [
        "python3", "pipeline/torch_process_segm.py",
        "-v", "--log",
        "-T", str(Directories.DB_FILE),
        "-C", Variables.CITY.title(),
        "-F", str(Directories.NPZ_LIST_FILE)
    ]))
    
    # Step 8: Find all .out files
    steps.append((8, "Create list of .out files",
        f"find {Directories.IMG_DIR} -name '*.out' > {Directories.OUT_LIST_FILE}"
    ))
    
    # Execute steps
    for step_num, description, cmd in steps:
        if callable(cmd):
            cmd()
        else:
            run_command(cmd, description)
    
    print(f"\n{'#'*70}")
    print(f"# PIPELINE COMPLETED SUCCESSFULLY FOR: {Variables.CITY.upper()}")
    print(f"# All files saved in: {Directories.CITY_DIR.absolute()}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()