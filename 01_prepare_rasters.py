"""Steps 1-2: build aligned Sentinel-2 stacks (B02,B03,B04,B08,VALID) for both dates,
and align ESA WorldCover to the same 10 m grid."""
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, Resampling
from config import *

BANDS = ["B02", "B03", "B04", "B08"]
BAD_SCL = [0, 1, 3, 8, 9, 10, 11]   # nodata, saturated, cloud shadow, clouds, cirrus, snow


def make_reference_grid():
    """A 10 m UTM grid covering BBOX. Everything is resampled onto it."""
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    x0, y0 = tr.transform(BBOX[0], BBOX[1])
    x1, y1 = tr.transform(BBOX[2], BBOX[3])
    west, south = np.floor(x0 / RES) * RES, np.floor(y0 / RES) * RES
    east, north = np.ceil(x1 / RES) * RES, np.ceil(y1 / RES) * RES
    return dict(transform=from_origin(west, north, RES, RES),
                width=int((east - west) / RES), height=int((north - south) / RES), crs=CRS)


REF = make_reference_grid()


def find_file(folder, key):
    for f in sorted(folder.iterdir()):
        if key in f.name and f.suffix.lower() in (".tif", ".tiff"):
            return f
    return None


def read_on_grid(path, resampling):
    out = np.zeros((REF["height"], REF["width"]), dtype="float32")
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=out,
                  dst_transform=REF["transform"], dst_crs=REF["crs"], resampling=resampling)
    return out


def build_stack(date):
    folder = RAW_S2 / date
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")
    layers = []
    for b in BANDS:
        f = find_file(folder, b)
        if f is None:
            raise SystemExit(f"No GeoTIFF containing '{b}' in {folder}")
        arr = np.nan_to_num(read_on_grid(f, Resampling.bilinear))
        if arr.max() > 2:            # digital numbers (0-10000) -> reflectance (0-1)
            arr = arr / 10000.0
        layers.append(arr.astype("float32"))

    valid = np.all([l > 0 for l in layers], axis=0)
    scl_file = find_file(folder, "SCL")
    if scl_file is not None:
        scl = np.rint(read_on_grid(scl_file, Resampling.nearest)).astype(int)
        valid &= ~np.isin(scl, BAD_SCL)
    else:
        print(f"  WARNING: no SCL file for {date}; clouds are NOT masked.")
    layers.append(valid.astype("float32"))

    path = stack_path(date)
    with rasterio.open(path, "w", driver="GTiff", height=REF["height"], width=REF["width"],
                       count=5, dtype="float32", crs=CRS, transform=REF["transform"],
                       compress="deflate") as dst:
        dst.write(np.stack(layers))
        dst.descriptions = tuple(BANDS + ["VALID"])

    rgb = np.stack([layers[2], layers[1], layers[0]], axis=-1)
    plt.imsave(OUT / f"rgb_{date}.png", np.clip(rgb / RGB_MAX, 0, 1))
    print(f"  {date}: saved {path.name}  size={REF['width']}x{REF['height']}  valid pixels={valid.mean()*100:.1f}%")


def align_worldcover():
    files = sorted(RAW_WC.glob("*.tif"))
    if not files:
        raise SystemExit(f"Put the WorldCover GeoTIFF in {RAW_WC}")
    with rasterio.open(files[0]) as src:
        with WarpedVRT(src, crs=CRS, transform=REF["transform"], width=REF["width"],
                       height=REF["height"], resampling=Resampling.nearest) as vrt:
            data = vrt.read(1).astype("uint8")
    with rasterio.open(WC_PATH, "w", driver="GTiff", height=REF["height"], width=REF["width"],
                       count=1, dtype="uint8", crs=CRS, transform=REF["transform"],
                       compress="deflate") as dst:
        dst.write(data, 1)
    vals, counts = np.unique(data, return_counts=True)
    print("  WorldCover class shares (%):",
          {int(v): round(c / data.size * 100, 1) for v, c in zip(vals, counts)})


if __name__ == "__main__":
    print("Sentinel-2 stacks:")
    for d in (DATE1, DATE2):
        build_stack(d)
    print("WorldCover:")
    align_worldcover()
    print("Done.")
