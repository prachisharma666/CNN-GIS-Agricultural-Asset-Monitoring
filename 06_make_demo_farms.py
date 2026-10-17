"""OPTIONAL shortcut: generate synthetic square 'farms' + fake claimed areas.
Skip this if you digitize real polygons in QGIS (save them as data/farms.geojson)."""
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import xy
from shapely.geometry import box
from config import *

if not PRED_PATH.is_file():
    raise SystemExit(
        f"{PRED_PATH} not found. Run 04_classify_map.py first "
        "(after 02_make_dataset.py and 03_train_cnn.py)."
    )

rng = np.random.default_rng(7)
with rasterio.open(PRED_PATH) as src:
    pred, tr = src.read(1), src.transform
H, W = pred.shape
m = 40
inside = np.zeros_like(pred, dtype=bool); inside[m:H - m, m:W - m] = True
crop_idx = np.argwhere((pred == 1) & inside)
other_idx = np.argwhere((pred == 0) & inside)

picks = [crop_idx[i] for i in rng.choice(len(crop_idx), 14, replace=False)]
if len(other_idx) >= 6:
    picks += [other_idx[i] for i in rng.choice(len(other_idx), 6, replace=False)]

crops = ["Onion", "Grapes", "Wheat", "Sugarcane", "Tomato"]
recs = []
for k, (r, c) in enumerate(picks, 1):
    x, y = xy(tr, int(r), int(c))
    side = rng.uniform(120, 250)                       # metres
    poly = box(x - side / 2, y - side / 2, x + side / 2, y + side / 2)
    area_ha = poly.area / 1e4
    claimed = round(area_ha * rng.choice([1.0, 1.0, 1.0, 0.9, 1.1, 1.6]), 2)
    recs.append(dict(farm_id=f"F{k:03d}", crop_type=str(rng.choice(crops)),
                     claimed_area_ha=claimed, geometry=poly))

gdf = gpd.GeoDataFrame(recs, crs=CRS).to_crs("EPSG:4326")
FARMS_FILE.parent.mkdir(exist_ok=True)
gdf.to_file(FARMS_FILE, driver="GeoJSON")
print(f"Saved {len(gdf)} synthetic farms to {FARMS_FILE}")
