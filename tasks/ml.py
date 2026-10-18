import threading
from functools import lru_cache

import numpy as np
from django.conf import settings
from PIL import Image

WIN = 32
CLASSES = ["non-cropland", "cropland"]
_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_model():
    from tensorflow import keras

    path = settings.MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    return keras.models.load_model(path)


def predict_image(file_obj):
    img = Image.open(file_obj).convert("RGB").resize((WIN, WIN), Image.BILINEAR)
    x = np.asarray(img, dtype="float32")[None, ...]
    with _lock:
        probs = get_model().predict(x, verbose=0)[0]
    return {
        "label": CLASSES[int(probs.argmax())],
        "cropland_probability": float(probs[1]),
        "percent": round(float(probs[1]) * 100, 1),
        "probabilities": [float(p) for p in probs],
    }