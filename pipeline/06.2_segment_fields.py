"""Step 7: split the cropland mask into field-sized polygons (SLIC segmentation on NDVI)
and compute mean NDVI per field for both dates + the change.

Run AFTER 05_make_polygons.py and 06_ndvi.py.
Install first:  pip install scikit-image geopandas shapely
"""
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from pipeline.config import *

# ---------------- settings you can tune ----------------
POLY_SRC = "cnn"          # "cnn" (CNN map) or "wc" (WorldCover) - which cropland mask to split
TARGET_PX = 60         # average field size in pixels (60 px = 0.6 ha at 10 m). Smaller -> more, smaller fields
COMPACTNESS = 0.1         # lower = follow NDVI edges more; higher = more regular blobs
MIN_AREA_M2 = 1000        # drop segments smaller than 0.1 ha
NODATA = -9999.0
# --------------------------------------------------------

# cropland mask
if POLY_SRC == "cnn":
    with rasterio.open(PRED_PATH) as p:
        crop = p.read(1) == 1
else:
    with rasterio.open(WC_PATH) as p:
        crop = p.read(1) == CROPLAND_CLASS

# NDVI rasters written by 06_ndvi.py
def read_ndvi(date):
    with rasterio.open(ndvi_path(date)) as src:
        a = src.read(1)
        return np.where(a == NODATA, np.nan, a), src.transform, src.crs

n1, transform, crs = read_ndvi(DATE1)
n2, _, _ = read_ndvi(DATE2)
diff = n2 - n1
ok = ~np.isnan(n1) & ~np.isnan(n2)
crop = crop & ok

# image for segmentation: NDVI date1, NDVI date2, NDVI change, each scaled to 0-1
def scale(a, lo, hi):
    return np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0, 1)

img = np.dstack([scale(n1, -0.2, 0.9), scale(n2, -0.2, 0.9), scale(diff, -0.5, 0.5)]).astype("float32")

n_seg = max(10, int(crop.sum() // TARGET_PX))
print(f"Cropland pixels: {crop.sum():,}  -> asking SLIC for ~{n_seg:,} segments (this can take a minute)")
try:
    labels = slic(img, n_segments=n_seg, compactness=COMPACTNESS, mask=crop,
                  start_label=1, enforce_connectivity=True, channel_axis=-1)
except TypeError:                                   # older scikit-image
    labels = slic(img, n_segments=n_seg, compactness=COMPACTNESS, mask=crop,
                  start_label=1, enforce_connectivity=True, multichannel=True)
labels = np.where(crop, labels, 0).astype("int32")
print("Segments produced:", len(np.unique(labels)) - 1)

# mean NDVI per segment (fast, pure numpy)
nlab = labels.max() + 1
def seg_mean(arr):
    m = (labels > 0) & ~np.isnan(arr)
    s = np.bincount(labels[m], weights=arr[m], minlength=nlab)
    c = np.bincount(labels[m], minlength=nlab)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)

m1, m2 = seg_mean(n1), seg_mean(n2)
pix = np.bincount(labels[labels > 0], minlength=nlab)

# vectorise
geoms, ids = [], []
for g, v in features.shapes(labels, mask=labels > 0, transform=transform):
    geoms.append(shape(g)); ids.append(int(v))
gdf = gpd.GeoDataFrame({"seg_id": ids, "geometry": geoms}, crs=crs)
gdf = gdf.dissolve(by="seg_id", as_index=False)          # one polygon per segment

gdf["area_ha"] = (gdf.geometry.area / 10000).round(3)
gdf = gdf[gdf.geometry.area >= MIN_AREA_M2].copy()
gdf[f"ndvi_{DATE1}"] = m1[gdf["seg_id"]].round(4)
gdf[f"ndvi_{DATE2}"] = m2[gdf["seg_id"]].round(4)
gdf["ndvi_change"] = (gdf[f"ndvi_{DATE2}"] - gdf[f"ndvi_{DATE1}"]).round(4)
gdf = gdf.dropna(subset=["ndvi_change"])

out = OUT / "cropland_fields_ndvi.gpkg"
gdf.to_file(out, driver="GPKG")
print(f"\n{len(gdf)} field polygons, median size {gdf['area_ha'].median():.2f} ha")
print(gdf[[f"ndvi_{DATE1}", f"ndvi_{DATE2}", "ndvi_change"]].describe().round(3))
print("Saved", out)

# quick look: boundaries over the NDVI-change image
fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(mark_boundaries(img, labels, color=(1, 1, 0), mode="inner"))
ax.set_title("Field segments over NDVI stack (R=date1, G=date2, B=change)")
ax.axis("off")
plt.savefig(OUT / "field_segments.png", dpi=150, bbox_inches="tight")
print("Saved", OUT / "field_segments.png")