"""Step 9: Streamlit dashboard.   Run:  streamlit run app.py"""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import streamlit as st
import folium
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from streamlit_folium import st_folium
from config import *

st.set_page_config(page_title="Nashik Farm Monitoring (Prototype)", layout="wide")
st.title("CNN-GIS Agricultural Asset Verification & Risk Monitoring - Nashik, Maharashtra")
st.caption("Academic research prototype. Flags support human review only; they are not proof of "
           "misreporting, crop damage, or a lending/insurance decision.")

RESULTS = OUT / "farm_results.geojson"
if not RESULTS.exists():
    st.error("Run 08_farm_analysis.py first.")
    st.stop()


@st.cache_data
def load_results():
    return gpd.read_file(RESULTS)


@st.cache_data
def load_raster(path):
    with rasterio.open(path) as src:
        return src.read(), src.transform


def crop(arr, transform, bounds, pad=150):
    """Cut arr (..., H, W) around bounds (UTM metres) + pad; return subarray and plot extent."""
    minx, miny, maxx, maxy = bounds
    c0, r1 = ~transform * (minx - pad, miny - pad)
    c1, r0 = ~transform * (maxx + pad, maxy + pad)
    H, W = arr.shape[-2:]
    r0, r1 = max(int(np.floor(r0)), 0), min(int(np.ceil(r1)), H)
    c0, c1 = max(int(np.floor(c0)), 0), min(int(np.ceil(c1)), W)
    sub = arr[..., r0:r1, c0:c1]
    left, top = transform.c + c0 * transform.a, transform.f + r0 * transform.e
    extent = (left, left + sub.shape[-1] * transform.a, top + sub.shape[-2] * transform.e, top)
    return sub, extent


def panel(img, extent, geom, title, cmap=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    for p in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
        x, y = p.exterior.xy
        ax.plot(x, y, color="red", lw=2)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    return fig


def fmt(v, nd=2):
    return "n/a" if v is None or pd.isna(v) else f"{v:.{nd}f}"


gdf = load_results()
farm_id = st.sidebar.selectbox("Select farm", gdf["farm_id"].tolist())
row = gdf[gdf["farm_id"] == farm_id].iloc[0]
geom_utm = gpd.GeoSeries([row.geometry], crs="EPSG:4326").to_crs(CRS).iloc[0]

st.subheader(f"Farm {row.farm_id}  -  crop (demo record): {row.crop_type}")
(st.warning if row.status == "REVIEW" else st.success)(f"Status: {row.status}   |   Flags: {row.flags}")

c = st.columns(4)
c[0].metric("Claimed area (ha)", fmt(row.claimed_area_ha))
c[1].metric("Polygon area (ha)", fmt(row.polygon_area_ha))
c[2].metric("Estimated cropland (ha)", fmt(row.cropland_area_ha))
c[3].metric("Area consistency (%)", fmt(row.area_consistency_pct, 1))
c = st.columns(4)
c[0].metric(f"Mean NDVI {DATE1}", fmt(row.ndvi_d1, 3))
c[1].metric(f"Mean NDVI {DATE2}", fmt(row.ndvi_d2, 3))
c[2].metric("NDVI change", fmt(row.ndvi_change, 3))
c[3].metric("Cloud-free pixels (d1 / d2 %)", f"{fmt(row.valid_cov_d1, 0)} / {fmt(row.valid_cov_d2, 0)}")
c = st.columns(2)
c[0].metric("Distance to nearest mapped road (km)", fmt(row.dist_road_km))
c[1].metric("Distance to nearest mapped river/stream (km)", fmt(row.dist_river_km))

# ---- Interactive map
st.markdown("#### Farm boundary")
ctr = row.geometry.centroid
m = folium.Map(location=[ctr.y, ctr.x], zoom_start=16, tiles=None)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery", name="Satellite").add_to(m)
for _, r in gdf.iterrows():
    sel = r.farm_id == farm_id
    colour = "red" if r.status == "REVIEW" else "lime"
    folium.GeoJson(r.geometry.__geo_interface__, tooltip=r.farm_id,
                   style_function=lambda _, col=colour, w=(4 if sel else 1): {
                       "color": col, "weight": w, "fillOpacity": 0.1}).add_to(m)
st_folium(m, height=420, width=900)

# ---- Raster panels
stack, tr = load_raster(str(stack_path(DATE1)))
pred, _ = load_raster(str(PRED_PATH))
n1, _ = load_raster(str(ndvi_path(DATE1)))
n2, _ = load_raster(str(ndvi_path(DATE2)))
b = geom_utm.bounds

sub, ext = crop(stack, tr, b)
rgb = np.clip(np.stack([sub[2], sub[1], sub[0]], -1) / RGB_MAX, 0, 1)
cls, _ = crop(pred[0], tr, b)
a1, _ = crop(np.where(n1[0] == -9999, np.nan, n1[0]), tr, b)
a2, _ = crop(np.where(n2[0] == -9999, np.nan, n2[0]), tr, b)

st.markdown("#### Satellite analysis")
c = st.columns(3)
c[0].pyplot(panel(rgb, ext, geom_utm, f"RGB {DATE1}"))
c[1].pyplot(panel(np.where(cls == 255, np.nan, cls), ext, geom_utm, "CNN: cropland (green) / other (tan)",
                  cmap=ListedColormap(["#d2b48c", "#2e8b3d"]), vmin=0, vmax=1))
c[2].pyplot(panel(a2 - a1, ext, geom_utm, "NDVI change", cmap="RdBu", vmin=-0.5, vmax=0.5))
c = st.columns(3)
c[0].pyplot(panel(a1, ext, geom_utm, f"NDVI {DATE1}", cmap="RdYlGn", vmin=-0.2, vmax=0.9))
c[1].pyplot(panel(a2, ext, geom_utm, f"NDVI {DATE2}", cmap="RdYlGn", vmin=-0.2, vmax=0.9))

# ---- Report download
report = f"""FARM MONITORING REPORT (research prototype)
Farm ID: {row.farm_id}   Crop (demo record): {row.crop_type}
Status: {row.status}
Flags: {row.flags}

Claimed area: {fmt(row.claimed_area_ha)} ha | Polygon area: {fmt(row.polygon_area_ha)} ha
CNN-estimated cropland: {fmt(row.cropland_area_ha)} ha ({fmt(row.cropland_pct_of_polygon, 1)}% of polygon)
Area consistency ratio: {fmt(row.area_consistency_pct, 1)}%
NDVI {DATE1}: {fmt(row.ndvi_d1, 3)} | NDVI {DATE2}: {fmt(row.ndvi_d2, 3)} | Change: {fmt(row.ndvi_change, 3)}
Cloud-free coverage: {fmt(row.valid_cov_d1, 0)}% / {fmt(row.valid_cov_d2, 0)}%
Nearest mapped road: {fmt(row.dist_road_km)} km | Nearest mapped river/stream: {fmt(row.dist_river_km)} km

NOTE: Thresholds are experimental. Flags are prompts for human review, not evidence of fraud,
crop damage or credit risk. Reference labels come from ESA WorldCover, not field surveys.
"""
d1, d2 = st.columns(2)
d1.download_button("Download report (.txt)", report, file_name=f"{row.farm_id}_report.txt")
d2.download_button("Download row (.csv)", pd.DataFrame([row.drop("geometry")]).to_csv(index=False),
                   file_name=f"{row.farm_id}_results.csv")
