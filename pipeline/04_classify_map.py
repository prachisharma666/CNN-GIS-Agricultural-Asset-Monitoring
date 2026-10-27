"""Step 4 (TensorFlow version): run the trained Keras model over the whole study area.
Use this INSTEAD of 04_classify_map.py."""
import numpy as np
import rasterio
from tensorflow import keras
import matplotlib.pyplot as plt
from config import *

MODEL_TF = MODELS / "mobilenetv2_cropland.keras"
if not MODEL_TF.is_file():
    raise SystemExit(
        f"{MODEL_TF} not found. Run 02_make_dataset.py and 03_train_cnn.py first."
    )

model = keras.models.load_model(MODEL_TF)

with rasterio.open(stack_path(DATE1)) as src:
    st, profile = src.read(), src.profile
valid = st[4] > 0
rgb = np.stack([st[2], st[1], st[0]], axis=-1)
rgb8 = (np.clip(rgb / RGB_MAX, 0, 1) * 255).astype("uint8")
H, W = valid.shape
pad = (WIN - CELL) // 2
rgb_p = np.pad(rgb8, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

nr, nc = H // CELL, W // CELL
coords = [(i, j) for i in range(nr) for j in range(nc)]
prob = np.zeros((nr, nc), dtype="float32")
BATCH = 256
for s in range(0, len(coords), BATCH):
    chunk = coords[s:s + BATCH]
    x = np.stack([rgb_p[i * CELL:i * CELL + WIN, j * CELL:j * CELL + WIN] for i, j in chunk]).astype("float32")
    p = model.predict(x, verbose=0)[:, 1]                    # probability of class 1 = cropland
    for (i, j), v in zip(chunk, p):
        prob[i, j] = v
    if (s // BATCH) % 10 == 0:
        print(f"  {s}/{len(coords)} cells")

prob_px = np.full((H, W), np.nan, dtype="float32")
prob_px[:nr * CELL, :nc * CELL] = np.kron(prob, np.ones((CELL, CELL), dtype="float32"))
pred = np.where(np.isnan(prob_px) | ~valid, 255, (prob_px >= 0.5).astype("uint8")).astype("uint8")

p1 = profile.copy(); p1.update(count=1, dtype="uint8", nodata=255, compress="deflate")
with rasterio.open(PRED_PATH, "w", **p1) as dst:
    dst.write(pred, 1)
p2 = profile.copy(); p2.update(count=1, dtype="float32", nodata=-1, compress="deflate")
with rasterio.open(PROB_PATH, "w", **p2) as dst:
    dst.write(np.where(np.isnan(prob_px), -1, prob_px).astype("float32"), 1)

# ---- Agreement with WorldCover (NOT a test score: training areas are included)
with rasterio.open(WC_PATH) as src:
    wc = src.read(1)
ok = (pred != 255) & (wc > 0)
agree = ((pred == 1) == (wc == CROPLAND_CLASS))[ok].mean() * 100
print(f"CNN map vs WorldCover agreement: {agree:.1f}%   cropland share (CNN) = {(pred[ok] == 1).mean()*100:.1f}%")

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(np.clip(rgb / RGB_MAX, 0, 1)); ax[0].set_title(f"Sentinel-2 RGB {DATE1}")
ax[1].imshow(np.where(pred == 255, np.nan, pred), cmap="YlGn", vmin=0, vmax=1); ax[1].set_title("CNN cropland map")
ax[2].imshow(np.where(wc == 0, np.nan, wc == CROPLAND_CLASS), cmap="YlGn", vmin=0, vmax=1); ax[2].set_title("WorldCover cropland")
for a in ax:
    a.axis("off")
plt.savefig(OUT / "classification_map.png", dpi=150, bbox_inches="tight")
print("Saved", PRED_PATH.name)