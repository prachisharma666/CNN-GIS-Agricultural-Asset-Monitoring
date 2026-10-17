"""Step 3: cut the DATE1 image into 8x8-pixel cells (each seen through a 32x32 window),
label each cell from WorldCover, and split train/val/test by geographic blocks."""
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pipeline.config import *

with rasterio.open(stack_path(DATE1)) as src:
    st = src.read()
with rasterio.open(WC_PATH) as src:
    wc = src.read(1)

valid = st[4] > 0
rgb = np.stack([st[2], st[1], st[0]], axis=-1)                      # R, G, B = B04, B03, B02
rgb8 = (np.clip(rgb / RGB_MAX, 0, 1) * 255).astype("uint8")
H, W = wc.shape
pad = (WIN - CELL) // 2
rgb_p = np.pad(rgb8, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
is_crop = wc == CROPLAND_CLASS

X, y, rows, cols = [], [], [], []
for r in range(0, H - CELL + 1, CELL):
    for c in range(0, W - CELL + 1, CELL):
        if valid[r:r + CELL, c:c + CELL].mean() < 0.9:               # cloudy / no data
            continue
        if (wc[r:r + CELL, c:c + CELL] == 0).mean() > 0.1:
            continue
        frac = is_crop[r:r + CELL, c:c + CELL].mean()
        if frac >= PURITY:
            label = 1
        elif frac <= 1 - PURITY:
            label = 0
        else:
            continue                                                 # mixed cell -> skip
        X.append(rgb_p[r:r + WIN, c:c + WIN])
        y.append(label); rows.append(r); cols.append(c)

X, y = np.stack(X), np.array(y)
rows, cols = np.array(rows), np.array(cols)
print(f"Total labelled cells: {len(y)}   cropland={int(y.sum())}   non-cropland={int((y == 0).sum())}")

# ---- Spatial split: 5x5 geographic blocks -> 15 train, 5 val, 5 test
NB = 5
block = (rows * NB // H) * NB + (cols * NB // W)
for seed in range(500):
    perm = np.random.default_rng(seed).permutation(NB * NB)
    kind = np.empty(NB * NB, dtype=object)
    kind[perm[:15]], kind[perm[15:20]], kind[perm[20:]] = "train", "val", "test"
    split = kind[block]
    if all(((split == s) & (y == k)).sum() >= 100 for s in ("train", "val", "test") for k in (0, 1)):
        break
else:
    raise SystemExit("Could not make a split with both classes in every set. "
                     "Choose a study area with more cropland AND non-cropland.")

arrays = {}
for s in ("train", "val", "test"):
    m = split == s
    arrays[f"X_{s}"], arrays[f"y_{s}"] = X[m], y[m]
    print(f"{s:5s}: {m.sum():6d} cells  (cropland {int(y[m].sum())}, non-cropland {int((y[m] == 0).sum())})")
np.savez_compressed(PROC / "dataset.npz", **arrays)

plt.figure(figsize=(6, 6))
plt.imshow(is_crop, cmap="Greens"); plt.title("WorldCover cropland (class 40)")
plt.axis("off"); plt.savefig(OUT / "worldcover_cropland.png", dpi=150, bbox_inches="tight")
print("Saved data/processed/dataset.npz")
