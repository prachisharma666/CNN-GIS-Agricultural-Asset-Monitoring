"""Steps 5, 7, 8: per-farm area check, NDVI summary, proximity, and monitoring flags."""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from pipeline.config import *

# ---- Experimental prototype thresholds (NOT validated banking standards)
AREA_LOW, AREA_HIGH = 70, 130     # area consistency ratio (%) outside this range -> review
NDVI_DROP = -0.20                 # NDVI change at or below this -> review
MIN_VALID = 0.50                  # minimum share of cloud-free pixels for NDVI comparison

if not FARMS_FILE.exists():
    raise SystemExit(f"{FARMS_FILE} not found. Digitize farms in QGIS or run 06_make_demo_farms.py")

farms = gpd.read_file(FARMS_FILE).to_crs(CRS)
bad = ~farms.geometry.is_valid
if bad.any():
    print(f"Fixing {bad.sum()} invalid geometries")
    farms.loc[bad, "geometry"] = farms.loc[bad, "geometry"].buffer(0)
for col in ("farm_id", "claimed_area_ha"):
    if col not in farms.columns:
        raise SystemExit(f"farms.geojson needs a '{col}' column")
if "crop_type" not in farms.columns:
    farms["crop_type"] = "n/a"


def read(path, nodata_to_nan=False):
    with rasterio.open(path) as src:
        a, t = src.read(1), src.transform
    if nodata_to_nan:
        a = np.where(a == -9999.0, np.nan, a)
    return a, t


pred, tr = read(PRED_PATH)
nd1, _ = read(ndvi_path(DATE1), True)
nd2, _ = read(ndvi_path(DATE2), True)


def union_of(path):
    if not path.exists():
        return None
    g = gpd.read_file(path).to_crs(CRS)
    if g.empty:
        return None
    return g.geometry.union_all() if hasattr(g.geometry, "union_all") else g.geometry.unary_union


roads, rivers = union_of(ROADS_FILE), union_of(RIVERS_FILE)

rows = []
for _, f in farms.iterrows():
    m = features.geometry_mask([f.geometry], out_shape=pred.shape, transform=tr, invert=True)
    n_pix = int(m.sum())
    poly_ha = f.geometry.area / 1e4
    claimed = float(f.claimed_area_ha)

    crop_ha = float((m & (pred == 1)).sum()) * RES * RES / 1e4
    ratio = crop_ha / claimed * 100 if claimed > 0 else np.nan

    v1, v2 = nd1[m], nd2[m]
    v1, v2 = v1[~np.isnan(v1)], v2[~np.isnan(v2)]
    cov1, cov2 = (len(v1) / n_pix, len(v2) / n_pix) if n_pix else (0, 0)
    mean1 = float(v1.mean()) if len(v1) else np.nan
    mean2 = float(v2.mean()) if len(v2) else np.nan
    change = mean2 - mean1

    c = f.geometry.centroid
    d_road = c.distance(roads) / 1000 if roads is not None else np.nan
    d_river = c.distance(rivers) / 1000 if rivers is not None else np.nan

    reasons = []
    if cov1 < MIN_VALID or cov2 < MIN_VALID:
        reasons.append("Too few cloud-free pixels for a reliable NDVI comparison")
    if not np.isnan(ratio) and ratio < AREA_LOW:
        reasons.append(f"Estimated cropland is only {ratio:.0f}% of claimed area")
    elif not np.isnan(ratio) and ratio > AREA_HIGH:
        reasons.append(f"Estimated cropland is {ratio:.0f}% of claimed area (above claim)")
    if not np.isnan(change) and change <= NDVI_DROP:
        reasons.append(f"NDVI declined by {abs(change):.2f} between dates")

    rows.append(dict(
        farm_id=f.farm_id, crop_type=f.crop_type, claimed_area_ha=round(claimed, 2),
        polygon_area_ha=round(poly_ha, 2), cropland_area_ha=round(crop_ha, 2),
        cropland_pct_of_polygon=round(crop_ha / poly_ha * 100, 1) if poly_ha else np.nan,
        area_consistency_pct=round(ratio, 1) if not np.isnan(ratio) else np.nan,
        ndvi_d1=round(mean1, 3), ndvi_d2=round(mean2, 3), ndvi_change=round(change, 3),
        valid_cov_d1=round(cov1 * 100, 1), valid_cov_d2=round(cov2 * 100, 1),
        dist_road_km=round(d_road, 2) if not np.isnan(d_road) else np.nan,
        dist_river_km=round(d_river, 2) if not np.isnan(d_river) else np.nan,
        status="REVIEW" if reasons else "OK",
        flags="; ".join(reasons) if reasons else "None",
    ))

res = pd.DataFrame(rows)
res.to_csv(OUT / "farm_results.csv", index=False)
out = gpd.GeoDataFrame(res, geometry=farms.geometry.values, crs=CRS).to_crs("EPSG:4326")
(OUT / "farm_results.geojson").unlink(missing_ok=True)
out.to_file(OUT / "farm_results.geojson", driver="GeoJSON")

pd.set_option("display.width", 200, "display.max_columns", 20)
print(res[["farm_id", "claimed_area_ha", "cropland_area_ha", "area_consistency_pct",
           "ndvi_d1", "ndvi_d2", "ndvi_change", "status"]].to_string(index=False))
print(f"\n{(res.status == 'REVIEW').sum()} of {len(res)} farms flagged for review.")
