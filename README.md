# Farm Monitor: Satellite-Based Cropland and Farm Verification

An end-to-end geospatial pipeline that uses free Sentinel-2 imagery and ESA WorldCover to (1) train a CNN that maps cropland, (2) compute NDVI on two dates, and (3) check individual farms against their claimed area and vegetation behaviour, flagging those that need manual review. The study area is a 10 km x 10 km box near Pimpalgaon Baswant, Nashik, Maharashtra.

This README documents the current script-based project and the target deployment (Docker + Django + plain HTML/JS frontend, replacing Streamlit). The thresholds used for flagging are **experimental prototype values, not validated banking or credit standards**.

---

## 1. What the project does

The idea in one paragraph: WorldCover 2021 gives a coarse, imperfect cropland label for every 10 m pixel. We use it as *weak supervision* to teach a MobileNetV2 CNN what cropland looks like in a January Sentinel-2 RGB image, so the model produces a fresh, up-to-date cropland map. We then compute NDVI for a January scene (crops near peak) and an April scene (after harvest or drier) and measure how vegetation changed. Finally, for each farm polygon (real, digitised in QGIS, or synthetic demo farms) we compare the cropland area the model sees against the area the farmer claims, look at the NDVI change, measure distance to roads and rivers, and raise REVIEW flags when something looks inconsistent.

Pipeline at a glance:

```
Planetary Computer (STAC)              Sentinel-2 L2A (2 dates) + WorldCover 2021
        |  00_download_data.py
        v
data/raw/sentinel2/<date>/*.tif, data/raw/worldcover/*.tif
        |  01_prepare_rasters.py        align everything to one 10 m UTM grid, cloud mask
        v
data/processed/s2_<date>.tif (5 bands), worldcover_aligned.tif
        |  02_make_dataset.py           8x8 px cells, 32x32 px context windows, spatial split
        v
data/processed/dataset.npz
        |  03_train_cnn.py              MobileNetV2 transfer learning + test metrics
        v
models/mobilenetv2_cropland.keras
        |  04_classify_map.py           classify the whole area
        v
data/processed/crop_pred.tif, crop_prob.tif
        |  04_2_make_polygons.py        (optional) cropland polygons
        |  05_ndvi.py                   NDVI date1, date2, difference (+ per-polygon stats)
        |  06_2_segment_fields.py       (optional) SLIC field segmentation + NDVI per field
        |  06_make_demo_farms.py        (optional) synthetic farms with fake claimed areas
        |  07_download_osm.py           (optional) roads and rivers
        v
        08_farm_analysis.py  ->  outputs/farm_results.csv + farm_results.geojson
```

---

## 2. Execution order

| Order | Script | Needs | Produces |
|---|---|---|---|
| 0 | `00_download_data.py` | internet | raw Sentinel-2 bands, WorldCover, updates `DATE1`/`DATE2` in `config.py` |
| 1 | `01_prepare_rasters.py` | step 0 | aligned stacks, WorldCover on the same grid, RGB previews |
| 2 | `02_make_dataset.py` | step 1 | `dataset.npz` (train/val/test) |
| 3 | `03_train_cnn.py` | step 2 | trained `.keras` model, `test_metrics.txt`, `confusion_matrix.png` |
| 4 | `04_classify_map.py` | step 3 | `crop_pred.tif`, `crop_prob.tif`, `classification_map.png` |
| 5 | `05_ndvi.py` | step 1, 4 | NDVI rasters, `ndvi_diff.tif`, `ndvi_maps.png` |
| 6 | `04_2_make_polygons.py cnn` | step 4 | `cropland_cnn.gpkg/.shp` (optional, enables per-polygon NDVI in step 5) |
| 7 | `06_2_segment_fields.py` | step 5 | `cropland_fields_ndvi.gpkg`, `field_segments.png` (optional) |
| 8 | `06_make_demo_farms.py` | step 4 | `data/farms.geojson` (skip if you digitise real farms) |
| 9 | `07_download_osm.py` | internet | `roads.geojson`, `rivers.geojson` (optional) |
| 10 | `08_farm_analysis.py` | steps 4, 5, 8 | `farm_results.csv`, `farm_results.geojson` |

Note: the file prefixes do not match the logical order exactly (two files start with `06`, and polygon creation is numbered `04_2` but is only needed by `05_ndvi.py` if you want per-polygon stats). The table above is the correct run order. If you run step 6 before step 5, `05_ndvi.py` will also write per-polygon NDVI.

---

## 3. File-by-file explanation

### `config.py`: central settings
Every script does `from config import *`. It defines all folders (`data/raw`, `data/processed`, `outputs`, `models`; created automatically), the study-area bounding box `BBOX = (west, south, east, north)` in WGS84, the working CRS `EPSG:32643` (UTM 43N, metres), the pixel size `RES = 10`, the two dates `DATE1` and `DATE2`, and the CNN parameters:

- `CELL = 8`: the map is classified in 8x8 pixel cells (80 m x 80 m).
- `WIN = 32`: the CNN sees a 32x32 pixel window (320 m) centred on each cell, so it has spatial context.
- `IMG_SIZE = 96`: the window is resized to 96x96 before entering MobileNetV2.
- `PURITY = 0.80`: a cell is used for training only if at least 80% of it is one class.
- `CROPLAND_CLASS = 40`: ESA WorldCover code for cropland.
- `RGB_MAX = 0.30`: reflectance value that maps to 255 when building 8-bit RGB images.

It also defines the file locations (`PRED_PATH`, `PROB_PATH`, `WC_PATH`, `FARMS_FILE`, `ROADS_FILE`, `RIVERS_FILE`) and two helpers, `stack_path(date)` and `ndvi_path(date)`. `MODEL_PATH` (a `.pt` file) is a leftover from an earlier PyTorch version and is not used by the TensorFlow scripts.

### `00_download_data.py`: data acquisition (no account needed)
Queries the Microsoft Planetary Computer STAC API. For each of two date windows (`WINDOW1` January to February, `WINDOW2` late March to April) it finds Sentinel-2 L2A scenes under 30% tile cloud that fully contain your `BBOX`, then checks the actual cloud/invalid fraction *inside your area* using the SCL band and picks the best scene. It reads only the study-area window of each cloud-optimised GeoTIFF (B02, B03, B04, B08 and SCL), converts digital numbers to reflectance with the post-2022 `-1000` offset, and saves per-band files under `data/raw/sentinel2/<date>/`. It also downloads ESA WorldCover 2021 for the box, and finally rewrites `DATE1` and `DATE2` in `config.py` with the chosen dates.

### `01_prepare_rasters.py`: alignment and cloud masking
Builds a single 10 m UTM reference grid covering `BBOX` and resamples everything onto it. For each date it creates a 5-band GeoTIFF `s2_<date>.tif` with bands B02, B03, B04, B08 and a VALID band (1 = usable pixel). VALID is false where any band is zero or where SCL marks nodata, saturation, cloud shadow, cloud, cirrus or snow. It saves an RGB preview PNG per date and aligns WorldCover to the same grid as `worldcover_aligned.tif`, printing the class shares.

### `02_make_dataset.py`: training data from weak labels
Slices the DATE1 image into 8x8 pixel cells, each paired with a 32x32 RGB window. Cells that are mostly cloud or nodata are skipped. A cell is labelled cropland (1) if at least 80% of its WorldCover pixels are class 40, non-cropland (0) if at most 20%, and skipped otherwise. To avoid spatial leakage the area is cut into a 5x5 grid of geographic blocks, and whole blocks are assigned to train (15), validation (5) or test (5), searching random seeds until every split has at least 100 cells of both classes. Output: `dataset.npz`.

### `03_train_cnn.py`: model training (TensorFlow/Keras)
Loads `dataset.npz`, applies flip and 90-degree rotation augmentation (satellite imagery has no "up"), computes class weights for imbalance, and builds MobileNetV2 (ImageNet weights) with a resize layer, a rescale to [-1, 1], global average pooling, dropout and a 2-class softmax head. Training is two-phase: 3 epochs training only the new head, then 8 epochs fine-tuning the whole network at a lower learning rate. A callback computes validation macro-F1 after each epoch and keeps the best model. The saved best model is evaluated once on the spatially separate test blocks, writing `test_metrics.txt` and `confusion_matrix.png`.

### `04_classify_map.py`: whole-area inference
Loads the saved model and slides over the entire DATE1 image in 8x8 cells with the 32x32 context window (reflect-padded at the edges), predicting cropland probability in batches of 256. The cell probabilities are expanded back to pixels, thresholded at 0.5 and written as `crop_pred.tif` (0, 1, or 255 for no data) and `crop_prob.tif`. It prints agreement with WorldCover (explicitly *not* a test score, because training areas are included) and saves a three-panel comparison figure.

### `04_2_make_polygons.py`: cropland polygons
Run as `python 04_2_make_polygons.py cnn` (or `wc` to use WorldCover directly). It removes isolated specks with a sieve (under 10 pixels), vectorises the mask, drops polygons under 0.5 ha, simplifies by 5 m to smooth stair-steps, and writes GeoPackage and Shapefile outputs (`cropland_cnn.*`).

### `05_ndvi.py`: vegetation index
Computes NDVI = (NIR - Red) / (NIR + Red) from the stacks for both dates, masked by VALID, and saves `ndvi_<date>.tif` and `ndvi_diff.tif` (date 2 minus date 1) with nodata `-9999`. It prints mean NDVI overall and over cropland only, saves a three-panel map figure, and, if `cropland_cnn.gpkg` exists, calculates mean NDVI per polygon with `rasterstats`.

### `06_2_segment_fields.py`: field-sized segments
Optional. Splits the cropland mask into field-sized pieces with SLIC superpixel segmentation on a 3-channel image (NDVI date 1, NDVI date 2, NDVI change). Tunable settings: `TARGET_PX` (average field size in pixels), `COMPACTNESS`, `MIN_AREA_M2`. It vectorises the segments, computes mean NDVI per segment for both dates and the change, and writes `cropland_fields_ndvi.gpkg` plus a boundary overlay PNG.

### `06_make_demo_farms.py`: synthetic farms
Optional shortcut. Picks 14 random cropland and 6 non-cropland locations from the CNN map, draws 120 to 250 m square farms, assigns random crop names, and sets a fake `claimed_area_ha` that is usually accurate but sometimes inflated or deflated. Saves `data/farms.geojson`. Skip it if you digitise real farms in QGIS (required columns: `farm_id`, `claimed_area_ha`; optional: `crop_type`).

### `07_download_osm.py`: roads and rivers
Optional. Uses `osmnx` (>= 2.0) to download major roads and rivers/streams/canals around the study area (with a 0.02 degree margin) and saves them as GeoJSON in `data/raw/osm/`.

### `08_farm_analysis.py`: the farm check
For each farm polygon it computes: the area of CNN-detected cropland inside the polygon; the **area consistency ratio** (detected cropland / claimed area); mean NDVI on both dates, the NDVI change, and the cloud-free coverage for each date; and centroid distance to the nearest road and river. A farm gets status `REVIEW` with plain-language reasons if any of these hold:

| Rule | Threshold |
|---|---|
| Area consistency outside the range | below 70% or above 130% |
| NDVI change | at or below -0.20 |
| Cloud-free coverage on either date | below 50% |

Results are written to `outputs/farm_results.csv` and `outputs/farm_results.geojson` (EPSG:4326), which are exactly what the web frontend will display.

### `requirements.txt`
Lists the Python dependencies. It currently still includes `torch`, `torchvision`, `streamlit`, `folium` and `streamlit-folium` from earlier versions, and is missing some packages the scripts need (see below). It will be replaced by a cleaned file in the Docker step.

### `_gitignore` (to be renamed `.gitignore`)
Keeps data, models, outputs and caches out of version control.

---

## 4. Known issues to fix before containerising

1. **Missing dependencies in `requirements.txt`:** `pystac-client`, `planetary-computer` (step 0), `scikit-image` (step 6_2), `rasterstats` (step 05), and `osmnx` is present but commented as optional. `torch`, `torchvision`, `streamlit`, `folium`, `streamlit-folium` are unnecessary for the Django version.
2. **Docstring numbering is stale:** several docstrings refer to `05_make_polygons.py` and `06_ndvi.py`, which are now `04_2_make_polygons.py` and `05_ndvi.py`. Behaviour is unaffected.
3. **Duplicate prefixes:** `06_2_segment_fields.py` and `06_make_demo_farms.py` both start with `06`; `04_classify_map.py` and `03_train_cnn.py` docstrings say "use this instead of..." although they are the only versions in the project.
4. **`config.py` leftovers:** `MODEL_PATH` points to a `.pt` file that is never used.
5. **GeoPandas version:** `08_farm_analysis.py` calls `union_all()` when available and falls back to `unary_union`, so it works on both old and new GeoPandas.
6. **Scripts run at import time:** most scripts execute their logic at module level instead of inside a `main()`. 

---

## 5. Target deployment: Docker + Django + HTML frontend

```
farm-monitor/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── manage.py
├── farmmonitor/          Django project (settings, urls, wsgi)
├── monitor/              Django app
│   ├── models.py         PipelineRun, Farm
│   ├── views.py          pages + JSON/GeoJSON API endpoints
│   ├── services.py       runs pipeline scripts as subprocesses, imports farm_results
│   ├── tasks.py          background execution and log capture
│   ├── templates/monitor/   index.html (map + table), run.html (pipeline control + logs)
│   └── static/monitor/      Leaflet map, JS, CSS
├── pipeline/             your existing scripts 00 to 08 + config.py, unchanged
└── data/ models/ outputs/   mounted as Docker volumes so results survive restarts
```

What Finished system will do:

- **Pipeline control page:** buttons to run each pipeline step (or the whole chain in order), with live log output and run status.
- **Farm upload:** upload a `farms.geojson` or generate demo farms.
- **Map page (Leaflet):** farm polygons coloured OK/REVIEW, popups with claimed vs detected area, NDVI values and flag reasons, toggleable layers for the CNN cropland map, NDVI change, roads and rivers.
- **Results table:** sortable and filterable, with CSV download.
- **Docker:** one image with Python, GDAL-capable geospatial wheels and TensorFlow (CPU), served by gunicorn, and `docker-compose up` brings everything up with persistent volumes.


## 6. Running the current scripts locally (before Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib rasterio geopandas shapely pyproj scikit-learn tensorflow \
            pystac-client planetary-computer scikit-image rasterstats osmnx
python 00_download_data.py
python 01_prepare_rasters.py
python 02_make_dataset.py
python 03_train_cnn.py
python 04_classify_map.py
python 04_2_make_polygons.py cnn
python 05_ndvi.py
python 06_make_demo_farms.py       
python 07_download_osm.py           
python 08_farm_analysis.py
```

## 7. Limitations

The cropland labels come from WorldCover 2021, which has its own errors, so the CNN inherits them and its test scores measure agreement with WorldCover rather than ground truth. Two dates cannot distinguish crop types or confirm a specific crop calendar. Area and NDVI thresholds are prototype values and must be calibrated against field-verified data before any real lending, insurance or compliance use.