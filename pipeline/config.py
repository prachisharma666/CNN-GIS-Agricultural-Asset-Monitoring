"""Central settings. Edit this file first; every script imports it."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_S2 = ROOT / "data" / "raw" / "sentinel2"      # one sub-folder per date
RAW_WC = ROOT / "data" / "raw" / "worldcover"
RAW_OSM = ROOT / "data" / "raw" / "osm"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
for p in (RAW_S2, RAW_WC, RAW_OSM, PROC, OUT, MODELS):
    p.mkdir(parents=True, exist_ok=True)

# ---- Study area: ~10 km x 10 km near Pimpalgaon Baswant, Nashik (west, south, east, north)
BBOX = (73.95, 20.10, 74.05, 20.20)
CRS = "EPSG:32643"          # UTM zone 43N (metres) - correct for Nashik
RES = 10                    # Sentinel-2 pixel size in metres

# ---- Two Sentinel-2 dates (folder names inside data/raw/sentinel2/). CHANGE to your dates.
DATE1 = "2026-01-03"
DATE2 = "2026-04-13"

# ---- CNN settings
CELL = 8                    # classify 8x8 pixel cells (80 m x 80 m)
WIN = 32                    # CNN sees a 32x32 pixel window (320 m) centred on the cell
IMG_SIZE = 96               # window is resized to 96x96 before MobileNetV2
PURITY = 0.80               # a cell is used for training only if >=80% of it is one class
CROPLAND_CLASS = 40         # ESA WorldCover code for cropland
RGB_MAX = 0.30              # reflectance value that maps to 255 in RGB images

# ---- Files
FARMS_FILE = ROOT / "data" / "farms.geojson"
ROADS_FILE = RAW_OSM / "roads.geojson"
RIVERS_FILE = RAW_OSM / "rivers.geojson"
MODEL_PATH = MODELS / "mobilenetv2_cropland.pt"
PRED_PATH = PROC / "crop_pred.tif"
PROB_PATH = PROC / "crop_prob.tif"
WC_PATH = PROC / "worldcover_aligned.tif"


def stack_path(date):
    return PROC / f"s2_{date}.tif"


def ndvi_path(date):
    return PROC / f"ndvi_{date}.tif"
