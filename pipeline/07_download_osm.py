"""OPTIONAL: download roads and rivers/streams from OpenStreetMap (needs osmnx >= 2.0)."""
import osmnx as ox
from pipeline.config import *

west, south, east, north = BBOX
bbox = (west - 0.02, south - 0.02, east + 0.02, north + 0.02)   # (left, bottom, right, top)


def fetch(tags, out_file, label):
    try:
        g = ox.features_from_bbox(bbox=bbox, tags=tags)
    except Exception as e:
        print(f"{label}: nothing downloaded ({e})")
        return
    g = g[g.geometry.type.isin(["LineString", "MultiLineString"])][["geometry"]].reset_index(drop=True)
    g.to_file(out_file, driver="GeoJSON")
    print(f"{label}: saved {len(g)} features -> {out_file.name}")


fetch({"highway": ["motorway", "trunk", "primary", "secondary", "tertiary"]}, ROADS_FILE, "Roads")
fetch({"waterway": ["river", "stream", "canal"]}, RIVERS_FILE, "Rivers/streams")
