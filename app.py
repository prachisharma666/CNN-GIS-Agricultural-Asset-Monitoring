"""Claim checker: draw a farm on the map, enter the claimed area, get a satellite-evidence verdict.
Run: streamlit run app.py
Needs the outputs of steps 01 (stacks), 04 (crop_pred.tif) and 05 (NDVI)."""
import numpy as np
import rasterio
import streamlit as st
import folium
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from folium.plugins import Draw
from rasterio import features
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from streamlit_folium import st_folium
from pipeline.config import *

st.set_page_config(page_title="Claim Checker", layout="wide")
st.title("Agricultural claim checker – Nashik study area")
st.caption("Does the satellite evidence support the claimed farm area? Research prototype: a decision-support "
           "aid for human reviewers, not proof of fraud, crop failure or loan eligibility.")

need = [PRED_PATH, stack_path(DATE1), ndvi_path(DATE1), ndvi_path(DATE2)]
if not all(p.exists() for p in need):
    st.error("Processed rasters are missing. Run steps 01, 04 and 05 first.")
    st.stop()


@st.cache_data
def load(path):
    with rasterio.open(path) as src:
        return src.read(), src.transform


def crop(arr, tr, bounds, pad=150):
    minx, miny, maxx, maxy = bounds
    c0, r1 = ~tr * (minx - pad, miny - pad)
    c1, r0 = ~tr * (maxx + pad, maxy + pad)
    H, W = arr.shape[-2:]
    r0, r1 = max(int(np.floor(r0)), 0), min(int(np.ceil(r1)), H)
    c0, c1 = max(int(np.floor(c0)), 0), min(int(np.ceil(c1)), W)
    sub = arr[..., r0:r1, c0:c1]
    left, top = tr.c + c0 * tr.a, tr.f + r0 * tr.e
    return sub, (left, left + sub.shape[-1] * tr.a, top + sub.shape[-2] * tr.e, top)


def panel(img, ext, geom, title, **kw):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, extent=ext, interpolation="nearest", **kw)
    for p in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
        x, y = p.exterior.xy
        ax.plot(x, y, color="red", lw=2)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return fig


pred = load(str(PRED_PATH))[0][0]
tr = load(str(PRED_PATH))[1]
stack = load(str(stack_path(DATE1)))[0]
n1 = np.where(load(str(ndvi_path(DATE1)))[0][0] == -9999, np.nan, load(str(ndvi_path(DATE1)))[0][0])
n2 = np.where(load(str(ndvi_path(DATE2)))[0][0] == -9999, np.nan, load(str(ndvi_path(DATE2)))[0][0])

# ---------------- inputs ----------------
with st.sidebar:
    st.header("1. Enter the claim")
    claimed = st.number_input("Claimed cultivated area (ha)", min_value=0.1, value=2.0, step=0.1)
    crop_type = st.text_input("Crop (optional)", "Onion")
    st.header("2. Draw the farm")
    st.write("Use the polygon or rectangle tool on the map (left toolbar) to outline the farm "
             "inside the yellow study-area box.")

W_, S_, E_, N_ = BBOX
m = folium.Map(location=[(S_ + N_) / 2, (W_ + E_) / 2], zoom_start=13, tiles=None)
folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                 attr="Esri World Imagery", name="Satellite").add_to(m)
folium.Rectangle([[S_, W_], [N_, E_]], color="yellow", fill=False, weight=2).add_to(m)
Draw(export=False, draw_options={"polygon": True, "rectangle": True, "polyline": False, "circle": False,
                                 "marker": False, "circlemarker": False}).add_to(m)
out = st_folium(m, height=480, width=900, key="map")

drawing = out.get("last_active_drawing") or (out.get("all_drawings") or [None])[-1]
if not drawing:
    st.info("Draw a farm boundary on the map to see the verdict.")
    st.stop()

# ---------------- analysis ----------------
to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform
geom = shp_transform(to_utm, shape(drawing["geometry"]))
if not geom.is_valid:
    geom = geom.buffer(0)
mask = features.geometry_mask([geom], out_shape=pred.shape, transform=tr, invert=True)
n_pix = int(mask.sum())
if n_pix < 30:
    st.error("The drawn area is outside the study area or smaller than about 0.3 ha. Draw a larger farm inside the yellow box.")
    st.stop()

poly_ha = geom.area / 1e4
cov = float((mask & (pred != 255)).sum()) / n_pix
crop_ha = float((mask & (pred == 1)).sum()) * RES * RES / 1e4
ratio = crop_ha / claimed * 100
v1, v2 = n1[mask], n2[mask]
v1, v2 = v1[~np.isnan(v1)], v2[~np.isnan(v2)]
m1 = float(v1.mean()) if len(v1) else np.nan
m2 = float(v2.mean()) if len(v2) else np.nan
chg = m2 - m1

if cov < 0.5:
    level, head = "grey", "CANNOT VERIFY – too few cloud-free satellite pixels over this farm"
elif 70 <= ratio <= 130:
    level, head = "green", "CLAIM SUPPORTED – satellite-estimated cropland is consistent with the claimed area"
elif 40 <= ratio < 70 or ratio > 130:
    level, head = "amber", "PARTIALLY SUPPORTED – claimed and estimated areas differ; human review suggested"
else:
    level, head = "red", "NOT SUPPORTED – little cropland detected for the claimed area; human review recommended"
{"green": st.success, "amber": st.warning, "red": st.error, "grey": st.info}[level](head)

c = st.columns(4)
c[0].metric("Claimed area (ha)", f"{claimed:.2f}")
c[1].metric("Drawn boundary (ha)", f"{poly_ha:.2f}")
c[2].metric("Satellite-estimated cropland (ha)", f"{crop_ha:.2f}")
c[3].metric("Area consistency", f"{ratio:.0f}%")
c = st.columns(3)
c[0].metric(f"Mean NDVI {DATE1}", "n/a" if np.isnan(m1) else f"{m1:.2f}")
c[1].metric(f"Mean NDVI {DATE2}", "n/a" if np.isnan(m2) else f"{m2:.2f}")
c[2].metric("Cloud-free pixels", f"{cov*100:.0f}%")

notes = []
if abs(poly_ha - claimed) / claimed > 0.3:
    notes.append(f"The boundary you drew ({poly_ha:.2f} ha) differs from the claimed area by more than 30%.")
if not np.isnan(m1) and m1 < 0.2:
    notes.append("Very little vegetation was visible on the first date; the land may be fallow, bare or not cultivated.")
if not np.isnan(chg) and chg <= -0.2:
    notes.append(f"NDVI fell by {abs(chg):.2f} between the dates. This can reflect harvest or season, not only crop damage.")
for n in notes:
    st.write("• " + n)

# ---------------- imagery ----------------
b = geom.bounds
sub, ext = crop(stack, tr, b)
rgb = np.clip(np.stack([sub[2], sub[1], sub[0]], -1) / RGB_MAX, 0, 1)
cls, _ = crop(pred, tr, b)
a1, _ = crop(n1, tr, b)
c = st.columns(3)
c[0].pyplot(panel(rgb, ext, geom, f"Satellite RGB {DATE1}"))
c[1].pyplot(panel(np.where(cls == 255, np.nan, cls), ext, geom, "CNN: cropland (green) / other (tan)",
                  cmap=ListedColormap(["#d2b48c", "#2e8b3d"]), vmin=0, vmax=1))
c[2].pyplot(panel(a1, ext, geom, f"NDVI {DATE1}", cmap="RdYlGn", vmin=-0.2, vmax=0.9))

report = (f"CLAIM CHECK (research prototype)\nCrop: {crop_type}\nVerdict: {head}\n"
          f"Claimed: {claimed:.2f} ha | Drawn boundary: {poly_ha:.2f} ha | Estimated cropland: {crop_ha:.2f} ha | "
          f"Consistency: {ratio:.0f}%\nNDVI {DATE1}: {m1:.2f} | NDVI {DATE2}: {m2:.2f} | Cloud-free: {cov*100:.0f}%\n"
          + "\n".join(notes) +
          "\n\nThis is preliminary evidence for human review. It does not establish ownership, fraud, crop failure "
          "or loan eligibility.\n")
st.download_button("Download verdict (.txt)", report, file_name="claim_check.txt")