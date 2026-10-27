"""FAST ALTERNATIVE to Copernicus - NO ACCOUNT NEEDED (Microsoft Planetary Computer).
Downloads, for your BBOX only (small files):
  * the least-cloudy Sentinel-2 L2A scene in each of two date windows (B02,B03,B04,B08,SCL)
  * ESA WorldCover 2021
Then it writes the chosen dates into config.py. Run once, then continue with 01_prepare_rasters.py.
Install first:  pip install pystac-client planetary-computer"""
import re
import numpy as np
import rasterio
from rasterio.windows import Window, transform as win_transform
from rasterio.warp import transform_bounds
import planetary_computer
import pystac_client
from shapely.geometry import box, shape
from config import *

WINDOW1 = "2026-01-01/2026-02-20"      # date 1: crops near peak growth
WINDOW2 = "2026-03-25/2026-04-30"      # date 2: after harvest / drier
MAX_TILE_CLOUD = 30                    # pre-filter on whole-tile cloud %
CANDIDATES = 8                         # how many scenes to inspect per window
BANDS = ["B02", "B03", "B04", "B08"]
BAD_SCL = [0, 1, 3, 8, 9, 10, 11]

catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)


def window_for(src, pad=0.003):
    w, s, e, n = BBOX
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", src.crs, w - pad, s - pad, e + pad, n + pad)
    r0, c0 = src.index(minx, maxy)
    r1, c1 = src.index(maxx, miny)
    r1, c1 = r1 + 1, c1 + 1
    if r0 < 0 or c0 < 0 or r1 > src.height or c1 > src.width:
        raise ValueError("this scene does not fully cover the study area")
    return Window(c0, r0, c1 - c0, r1 - r0)


def read_asset(href):
    """Read only the study-area window of a (cloud-optimised) GeoTIFF."""
    with rasterio.open(href) as src:
        win = window_for(src)
        return src.read(1, window=win), win_transform(win, src.transform), src.crs


def save(path, data, transform, crs):
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                       dtype=str(data.dtype), crs=crs, transform=transform, compress="deflate") as dst:
        dst.write(data, 1)


def bad_fraction(item):
    scl, _, _ = read_asset(item.assets["SCL"].href)
    return float(np.isin(scl, BAD_SCL).mean())


def pick_best(window):
    items = list(catalog.search(collections=["sentinel-2-l2a"], bbox=BBOX, datetime=window,
                                query={"eo:cloud_cover": {"lt": MAX_TILE_CLOUD}}).items())
    # keep only scenes whose footprint covers the WHOLE study box (a neighbouring tile that only
    # touches the box would give a mostly-empty image)
    aoi = box(*BBOX)
    items = [i for i in items if shape(i.geometry).contains(aoi)]
    items = sorted(items, key=lambda i: i.properties["eo:cloud_cover"])[:CANDIDATES]
    if not items:
        raise SystemExit(f"No scene fully covers your study box for {window}. Widen the window, "
                         "raise MAX_TILE_CLOUD, or shift BBOX in config.py.")
    best, best_bad = None, 2
    for it in items:
        try:
            frac = bad_fraction(it)
        except Exception as e:
            print(f"   skip {it.id}: {e}")
            continue
        print(f"   {it.datetime:%Y-%m-%d}  tile cloud={it.properties['eo:cloud_cover']:5.1f}%   "
              f"cloud/invalid in YOUR area={frac*100:5.1f}%")
        if frac < best_bad:
            best, best_bad = it, frac
    print(f"   -> chosen {best.datetime:%Y-%m-%d} ({best_bad*100:.1f}% invalid)")
    return best


def download_scene(item):
    date = item.datetime.strftime("%Y-%m-%d")
    folder = RAW_S2 / date
    folder.mkdir(parents=True, exist_ok=True)
    for b in BANDS + ["SCL"]:
        data, tr, crs = read_asset(item.assets[b].href)
        if b in BANDS:
            # raw DN -> reflectance. Scenes after Jan 2022 carry a +1000 offset. DN 0 = no data.
            refl = np.clip((data.astype("float32") - 1000.0) / 10000.0, 0.0001, 2.0)
            data = np.where(data == 0, 0.0, refl).astype("float32")
        else:
            data = data.astype("float32")
        save(folder / f"{date}_{b}.tif", data, tr, crs)
    print(f"   saved to {folder}")
    return date


def get_worldcover():
    items = list(catalog.search(collections=["esa-worldcover"], bbox=BBOX,
                                datetime="2021-01-01/2021-12-31").items())
    if not items:
        raise SystemExit("WorldCover 2021 not found on Planetary Computer.")
    data, tr, crs = read_asset(items[0].assets["map"].href)
    save(RAW_WC / "worldcover_2021_aoi.tif", data.astype("uint8"), tr, crs)
    print("WorldCover saved:", RAW_WC / "worldcover_2021_aoi.tif")


if __name__ == "__main__":
    print("Date 1 window:", WINDOW1)
    d1 = download_scene(pick_best(WINDOW1))
    print("Date 2 window:", WINDOW2)
    d2 = download_scene(pick_best(WINDOW2))
    print("WorldCover 2021:")
    get_worldcover()
    cfg = (ROOT / "config.py").read_text()
    cfg = re.sub(r'^DATE1 = ".*"', f'DATE1 = "{d1}"', cfg, flags=re.M)
    cfg = re.sub(r'^DATE2 = ".*"', f'DATE2 = "{d2}"', cfg, flags=re.M)
    (ROOT / "config.py").write_text(cfg)
    print(f"\nconfig.py updated: DATE1={d1}  DATE2={d2}\nNext: python 01_prepare_rasters.py")