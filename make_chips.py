from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

RGB_MAX, CELL, WIN, CROPLAND_CLASS, PURITY, PER_CLASS = 0.30, 8, 32, 40, 0.80, 3
pad = (WIN - CELL) // 2

proc = next((p for p in (Path("pipeline/data/processed"), Path("data/processed")) if list(p.glob("s2_*.tif"))), None)
if proc is None or not (proc / "worldcover_aligned.tif").is_file():
    raise SystemExit("Could not find s2_*.tif and worldcover_aligned.tif in pipeline/data/processed or data/processed")

with rasterio.open(sorted(proc.glob("s2_*.tif"))[0]) as src:
    st = src.read()
with rasterio.open(proc / "worldcover_aligned.tif") as src:
    wc = src.read(1)

rgb8 = (np.clip(np.stack([st[2], st[1], st[0]], axis=-1) / RGB_MAX, 0, 1) * 255).astype("uint8")
valid = st[4] > 0
is_crop = wc == CROPLAND_CLASS
H, W = wc.shape
out = Path("chips")
out.mkdir(exist_ok=True)

for name, want in (("cropland", True), ("noncropland", False)):
    cells = []
    for r in range(pad, H - CELL - pad, CELL):
        for c in range(pad, W - CELL - pad, CELL):
            if valid[r:r + CELL, c:c + CELL].mean() < 0.9 or (wc[r:r + CELL, c:c + CELL] == 0).mean() > 0.1:
                continue
            frac = is_crop[r:r + CELL, c:c + CELL].mean()
            if (want and frac >= PURITY) or (not want and frac <= 1 - PURITY):
                cells.append((r, c))
    if not cells:
        print(f"No {name} cells found")
        continue
    picks = np.random.default_rng(1).choice(len(cells), min(PER_CLASS, len(cells)), replace=False)
    for k, i in enumerate(picks, 1):
        r, c = cells[i]
        Image.fromarray(rgb8[r - pad:r + CELL + pad, c - pad:c + CELL + pad]).save(out / f"chip_{name}_{k}.png")
print("Saved:", ", ".join(sorted(p.name for p in out.glob("*.png"))))