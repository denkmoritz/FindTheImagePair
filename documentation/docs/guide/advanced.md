# Advanced

This section explains how to generate your own tables for a city.

Follow these steps:

## 1. Choose a city / create the initial table

Configure `initiate_table.py`:

- Choose the **city name**, **table name**, and **WGS84/UTM CRS** for the city
- Make sure the **PostgreSQL credentials** are correct

Run the script:

```bash
python3 initiate_table.py
```

## 2. Run the master pipeline

```bash
python3 master_pipeline.py
```

By default it uses the following example. However, if the city should be changed, simply all the variables can be easily altered.

Example:
```bash
CITY="Cape Town" CITY_EPSG="32734" BBOX_WEST="18.3" BBOX_SOUTH="-34.1" BBOX_EAST="18.5" BBOX_NORTH="-33.8" INNER="5" OUTER="20" MLY_SCORE="0.9" python3 master_pipeline.py
```

## 3. Test different thresholds

By default, the original `filter_output.py` [script](https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/percept-vsvi-filter/tree/main) has different thresholds for:

| Option | Short | Description | Default |
|------|------|------------|--------|
| `--contrast-threshold` | `-C` | Minimum contrast score | `0.35` |
| `--tone-mapping-threshold` | `-H` | Minimum tone-mapping score | `0.35` |
| `--tone-mapping-floor` | — | Tone-mapping floor | `0.8` |

After testing, we only found that `-C` is useful. The modified script `filter_multiple.py` uses for `-C` five different values and the other two variables the default thresholds.

```bash
python3 filter_multiple.py
```

## 4. Decide which threshold to use

There is no such a thing as the perfect threshold, that is why the user should decide on their own which one to use. Depending on the size of the desired table a smaller or larger threshold can be chosen. The script `compare.py` provides assistance.

```bash
python3 compare.py
```

## 5. New table & delete entries

After deciding for value, the final script can be used. It creates a new table from the original city table and keeps only the accepeted ids'. The new table is then called `city_cvalue`, e.g., `cape_town_035`.

```bash
python3 delete_entries.py -C 0.35 # example value
```