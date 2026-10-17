"""Step 6: NDVI for both dates and their difference, plus cropland-only means and per-polygon NDVI."""
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pipeline.config import *

NODATA = -9999.0

# ---- cropland mask: CNN map if it exists, otherwise WorldCover
POLY_SRC = "cnn"                      # "cnn" or "wc" (must match the polygons you made in step 5)
if POLY_SRC == "cnn":
    with rasterio.open(PRED_PATH) as p:
        crop = p.read(1) == 1
    POLY_PATH = OUT / "cropland_cnn.gpkg"
else:
    with rasterio.open(WC_PATH) as p:
        crop = p.read(1) == CROPLAND_CLASS
    POLY_PATH = OUT / "cropland_worldcover.gpkg"


def compute(date):
    with rasterio.open(stack_path(date)) as src:
        st, profile = src.read(), src.profile
    red, nir, valid = st[2], st[3], st[4] > 0
    ndvi = (nir - red) / (nir + red + 1e-10)
    ndvi = np.where(valid, ndvi, np.nan).astype("float32")
    ndvi_crop = np.where(crop, ndvi, np.nan)
    print(f"{date}: mean NDVI (all)={np.nanmean(ndvi):.3f}  "
          f"mean NDVI (cropland)={np.nanmean(ndvi_crop):.3f}  valid={valid.mean()*100:.1f}%")
    return ndvi, profile


def save(path, arr, profile):
    p = profile.copy()
    p.update(count=1, dtype="float32", nodata=NODATA, compress="deflate")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(np.where(np.isnan(arr), NODATA, arr).astype("float32"), 1)


n1, prof = compute(DATE1)
n2, _ = compute(DATE2)
diff = n2 - n1
print(f"Cropland-only mean NDVI change (date2 - date1): {np.nanmean(np.where(crop, diff, np.nan)):.3f}")
save(ndvi_path(DATE1), n1, prof)
save(ndvi_path(DATE2), n2, prof)
save(PROC / "ndvi_diff.tif", diff, prof)

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for a, arr, t in ((ax[0], n1, f"NDVI {DATE1}"), (ax[1], n2, f"NDVI {DATE2}")):
    im = a.imshow(arr, cmap="RdYlGn", vmin=-0.2, vmax=0.9); a.set_title(t); a.axis("off")
plt.colorbar(im, ax=ax[:2], shrink=0.7)
im = ax[2].imshow(diff, cmap="RdBu", vmin=-0.5, vmax=0.5); ax[2].set_title("NDVI change (date2 - date1)"); ax[2].axis("off")
plt.colorbar(im, ax=ax[2], shrink=0.7)
plt.savefig(OUT / "ndvi_maps.png", dpi=150, bbox_inches="tight")
print("Saved NDVI rasters and outputs/ndvi_maps.png")

# ---- mean NDVI per cropland polygon (needs: pip install rasterstats geopandas)
if POLY_PATH.is_file():
    import geopandas as gpd
    from rasterstats import zonal_stats

    gdf = gpd.read_file(POLY_PATH)
    for d in (DATE1, DATE2):
        z = zonal_stats(gdf, str(ndvi_path(d)), stats=["mean"], nodata=NODATA)
        gdf[f"ndvi_{d}"] = [s["mean"] for s in z]
    gdf["ndvi_change"] = gdf[f"ndvi_{DATE2}"] - gdf[f"ndvi_{DATE1}"]
    out_gpkg = OUT / "cropland_polygons_ndvi.gpkg"
    gdf.to_file(out_gpkg, driver="GPKG")
    print(gdf[[f"ndvi_{DATE1}", f"ndvi_{DATE2}", "ndvi_change"]].describe().round(3))
    print("Saved", out_gpkg)
else:
    print(f"{POLY_PATH.name} not found - run 05_make_polygons.py first to get per-polygon NDVI.")