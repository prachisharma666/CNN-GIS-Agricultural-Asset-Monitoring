import json
import threading
from pathlib import Path

import numpy as np
import rasterio
from django.conf import settings
from pyproj import Transformer
from rasterio import features
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

AREA_LOW, AREA_HIGH = 70, 130
NDVI_DROP = -0.20
MIN_VALID = 0.50

FLAG_HELP = {
    "area_low": "The CNN finds noticeably less cropland inside the boundary than the farmer claims. The claimed area may be overstated, "
                "or part of the plot may be fallow, built-up or outside the drawn boundary.",
    "area_high": "The CNN finds more cropland inside the boundary than the farmer claims. The claim may be understated, "
                 "or the boundary may include neighbouring fields.",
    "ndvi_drop": "Vegetation greenness fell sharply between the two satellite dates. This can be normal harvest timing, "
                 "or it can point to crop stress or failure.",
    "coverage": "Too many pixels were hidden by cloud or shadow on one of the dates, so the vegetation comparison is not reliable.",
    "other": "",
}

_cache = {}
_lock = threading.Lock()


class AnalysisUnavailable(Exception):
    pass


class OutsideStudyArea(Exception):
    pass


def _cached(path, loader):
    p = Path(path)
    if not p.is_file():
        return None
    key = (str(p), p.stat().st_mtime_ns)
    with _lock:
        hit = _cache.get(key)
        if hit is None:
            for stale in [k for k in _cache if k[0] == key[0]]:
                del _cache[stale]
            hit = _cache[key] = loader(p)
    return hit


def _read_raster(p):
    with rasterio.open(p) as src:
        return src.read(1), src.transform


def _read_ndvi(p):
    arr, tr = _read_raster(p)
    return np.where(arr == -9999.0, np.nan, arr), tr


def _read_lines(p):
    data = json.loads(Path(p).read_text())
    geoms = [shape(f["geometry"]) for f in data.get("features", []) if f.get("geometry")]
    if not geoms:
        return None
    to_utm = Transformer.from_crs("EPSG:4326", settings.ANALYSIS_CRS, always_xy=True).transform
    return shp_transform(to_utm, unary_union(geoms))


def flag_code(text):
    if text.startswith("Too few cloud-free"):
        return "coverage"
    if "(above claim)" in text:
        return "area_high"
    if text.startswith("Estimated cropland is only"):
        return "area_low"
    if text.startswith("NDVI declined"):
        return "ndvi_drop"
    return "other"


def make_flags(ratio, change, cov1, cov2):
    texts = []
    if cov1 < MIN_VALID or cov2 < MIN_VALID:
        texts.append("Too few cloud-free pixels for a reliable NDVI comparison")
    if ratio is not None and ratio < AREA_LOW:
        texts.append(f"Estimated cropland is only {ratio:.0f}% of claimed area")
    elif ratio is not None and ratio > AREA_HIGH:
        texts.append(f"Estimated cropland is {ratio:.0f}% of claimed area (above claim)")
    if change is not None and change <= NDVI_DROP:
        texts.append(f"NDVI declined by {abs(change):.2f} between dates")
    return [{"code": flag_code(t), "text": t} for t in texts]


def priority(flags):
    codes = {f["code"] for f in flags}
    if not codes:
        return "none"
    return "high" if "area_low" in codes or len(codes) >= 2 else "medium"


def _num(v, digits):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), digits)


def study_bounds():
    hit = _cached(settings.ANALYSIS_PRED_PATH, _read_raster)
    if hit is None:
        return None
    pred, tr = hit
    h, w = pred.shape
    to_ll = Transformer.from_crs(settings.ANALYSIS_CRS, "EPSG:4326", always_xy=True).transform
    west, south = to_ll(tr.c, tr.f + h * tr.e)
    east, north = to_ll(tr.c + w * tr.a, tr.f)
    return [[south, west], [north, east]]


def analyse_geometry(geometry, claimed_ha):
    pred_hit = _cached(settings.ANALYSIS_PRED_PATH, _read_raster)
    if pred_hit is None:
        raise AnalysisUnavailable("The cropland map (crop_pred.tif) does not exist yet. Run the pipeline scripts first.")
    pred, tr = pred_hit

    geom = shape(geometry)
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("The boundary must be a polygon")
    g = shp_transform(Transformer.from_crs("EPSG:4326", settings.ANALYSIS_CRS, always_xy=True).transform, geom)

    h, w = pred.shape
    study = box(tr.c, tr.f + h * tr.e, tr.c + w * tr.a, tr.f)
    if not g.intersects(study):
        raise OutsideStudyArea("This boundary is outside the mapped study area. Draw it inside the dashed box.")

    warnings = []
    if g.intersection(study).area / g.area < 0.99:
        warnings.append("Part of this boundary lies outside the mapped study area; only the inside part was analysed.")

    mask = features.geometry_mask([g], out_shape=pred.shape, transform=tr, invert=True)
    n_pix = int(mask.sum())
    poly_ha = g.area / 1e4
    crop_ha = float((mask & (pred == 1)).sum()) * settings.ANALYSIS_PIXEL_SIZE**2 / 1e4
    claimed = float(claimed_ha)
    ratio = crop_ha / claimed * 100 if claimed > 0 else None
    if claimed > 0 and poly_ha > 3 * claimed:
        warnings.append(
            f"The drawn boundary ({poly_ha:.1f} ha) is {poly_ha / claimed:.0f} times larger than the claimed area ({claimed:.1f} ha). "
            "Redraw it around only the field being claimed; otherwise the cropland comparison is not meaningful."
        )
    elif claimed > 0 and poly_ha < claimed / 3:
        warnings.append(
            f"The drawn boundary ({poly_ha:.1f} ha) is much smaller than the claimed area ({claimed:.1f} ha). "
            "Check that the whole claimed field is inside the boundary.")
    n1 = _cached(_ndvi_path(settings.ANALYSIS_DATE1), _read_ndvi)
    n2 = _cached(_ndvi_path(settings.ANALYSIS_DATE2), _read_ndvi)
    mean1 = mean2 = change = None
    cov1 = cov2 = 0.0
    if n1 is not None and n2 is not None and n_pix:
        v1, v2 = n1[0][mask], n2[0][mask]
        v1, v2 = v1[~np.isnan(v1)], v2[~np.isnan(v2)]
        cov1, cov2 = len(v1) / n_pix, len(v2) / n_pix
        mean1 = float(v1.mean()) if len(v1) else None
        mean2 = float(v2.mean()) if len(v2) else None
        change = mean2 - mean1 if mean1 is not None and mean2 is not None else None
    else:
        warnings.append("NDVI rasters are missing, so the vegetation comparison was skipped.")

    centroid = g.centroid
    roads = _cached(settings.ANALYSIS_ROADS_FILE, _read_lines)
    rivers = _cached(settings.ANALYSIS_RIVERS_FILE, _read_lines)
    d_road = centroid.distance(roads) / 1000 if roads is not None else None
    d_river = centroid.distance(rivers) / 1000 if rivers is not None else None

    flags = make_flags(ratio, change, cov1, cov2)
    return {
        "polygon_area_ha": _num(poly_ha, 2),
        "cropland_area_ha": _num(crop_ha, 2),
        "cropland_pct_of_polygon": _num(crop_ha / poly_ha * 100, 1) if poly_ha else None,
        "area_consistency_pct": _num(ratio, 1),
        "ndvi_d1": _num(mean1, 3),
        "ndvi_d2": _num(mean2, 3),
        "ndvi_change": _num(change, 3),
        "valid_cov_d1": _num(cov1 * 100, 1),
        "valid_cov_d2": _num(cov2 * 100, 1),
        "dist_road_km": _num(d_road, 2),
        "dist_river_km": _num(d_river, 2),
        "flags": flags,
        "date1": settings.ANALYSIS_DATE1,
        "date2": settings.ANALYSIS_DATE2,
        "warnings": warnings,
    }


def thresholds():
    return {"area_low": AREA_LOW, "area_high": AREA_HIGH, "ndvi_drop": NDVI_DROP, "min_valid_pct": MIN_VALID * 100}


def _ndvi_path(date):
    return settings.BASE_DIR / "data" / "processed" / f"ndvi_{date}.tif"