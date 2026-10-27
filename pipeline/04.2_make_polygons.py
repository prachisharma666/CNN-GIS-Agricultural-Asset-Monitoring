"""Step 5: turn a raster (CNN prediction or WorldCover) into cropland polygons.

Usage:
    python 05_make_polygons.py cnn      # polygons from the CNN map (run step 4 first)
    python 05_make_polygons.py wc       # polygons straight from WorldCover (no model needed)

Install first:  pip install geopandas shapely
"""
import sys
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape
from config import *

MODE = sys.argv[1] if len(sys.argv) > 1 else "cnn"
MIN_AREA_M2 = 5000        # drop polygons smaller than 0.5 ha (50 pixels at 10 m)
SIEVE_PIXELS = 10         # remove isolated specks smaller than this many pixels first
SIMPLIFY_M = 5            # simplify tolerance in metres (0 = off); 5 m smooths stair-steps

if MODE == "cnn":
    src_path, value, name = PRED_PATH, 1, "cropland_cnn"
elif MODE == "wc":
    src_path, value, name = WC_PATH, CROPLAND_CLASS, "cropland_worldcover"
else:
    raise SystemExit("Use: python 05_make_polygons.py cnn   or   wc")

with rasterio.open(src_path) as src:
    arr = src.read(1)
    transform, crs = src.transform, src.crs

mask = (arr == value).astype("uint8")
# remove tiny isolated clumps before vectorising
mask = features.sieve(mask, size=SIEVE_PIXELS)

geoms = [shape(g) for g, v in features.shapes(mask, mask=mask.astype(bool), transform=transform) if v == 1]
gdf = gpd.GeoDataFrame({"geometry": geoms}, crs=crs)
gdf["area_m2"] = gdf.geometry.area
gdf = gdf[gdf["area_m2"] >= MIN_AREA_M2].copy()
if SIMPLIFY_M:
    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_M, preserve_topology=True)
gdf["area_ha"] = (gdf.geometry.area / 10000).round(3)
gdf["id"] = np.arange(1, len(gdf) + 1)
gdf = gdf[["id", "area_ha", "geometry"]]

gpkg = OUT / f"{name}.gpkg"
gdf.to_file(gpkg, driver="GPKG")
gdf.to_file(OUT / f"{name}.shp")
print(f"{len(gdf)} polygons, total {gdf['area_ha'].sum():.1f} ha")
print("Saved", gpkg, "and the .shp next to it")