#!/usr/bin/env python3
import os
from pathlib import Path
from config import Variables

# ========== CONFIG ==========
city = Variables.CITY_NORMAL
input_file = f"{city}/list-of-out-files-{city}.txt"
disable_road_check = True         # If True → ignore road centre count
os.makedirs(f"{city}/filtered_test", exist_ok=True)

# MULTI-PARAMETER SUPPORT
CONTRAST_THRESHOLDS     = [0.3, 0.35, 0.4, 0.45, 0.5]   # multiple values
TONE_THRESHOLDS         = [0.35]
TONE_MAPPING_FLOORS     = [0.8]
# ============================

# Matching tags inside `.out` files
contrast_tag = 'Skimage contrast: '
tone_mapping_tag = 'Tone-mapping score: '
centres_tag = 'Found road centres: ['
imgid_tag = 'Assuming imgid='
out_extensions = ['out']

# ----------------------------------------

def process_file(file, Cthr, Tthr, Tfloor, outfp):
    contrast = tone = imgid = None
    centres = []

    with open(file) as f:
        for line in f:
            if contrast_tag in line:
                contrast = float(line[len(contrast_tag):])
            elif tone_mapping_tag in line:
                tone = float(line[len(tone_mapping_tag):])
            elif imgid_tag in line:
                imgid = int(line[len(imgid_tag):])
            elif centres_tag in line:
                end = line.rfind("]")
                if end > -1:
                    centres = list(map(int, line[len(centres_tag):end].split()))

    # ACCEPT CHECK -----------------------------
    v = contrast + max(0, tone - Tfloor)
    accept = v > Cthr and (disable_road_check or len(centres) == 1)

    if accept and imgid is not None:
        outfp.write(f"{imgid}\n")


def run():
    files = [Path(x.strip()) for x in open(input_file).readlines()]

    # ---- generate ALL combinations ----
    for C in CONTRAST_THRESHOLDS:
        for H in TONE_THRESHOLDS:
            for F in TONE_MAPPING_FLOORS:

                output = f"{city}/filtered_test/filtered-{city}-C{C}-H{H}-F{F}.txt"
                print(f"\nRunning C={C} | H={H} | Floor={F} -> {output}")

                with open(output, "w") as outfp:
                    for p in files:
                        if p.is_file() and p.suffix[1:] in out_extensions:
                            process_file(p, C, H, F, outfp)

                print(f"Saved: {output}")


if __name__ == "__main__":
    run()