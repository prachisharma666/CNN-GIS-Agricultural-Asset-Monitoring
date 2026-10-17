"""Step 4 (TensorFlow/Keras version): MobileNetV2 transfer learning + evaluation.
Use this INSTEAD of 03_train_cnn.py. Needs data/processed/dataset.npz from 02_make_dataset.py."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from pipeline.config import *

MODEL_TF = MODELS / "mobilenetv2_cropland.keras"
keras.utils.set_random_seed(42)
print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))

d = np.load(PROC / "dataset.npz")
BATCH = 64


def augment(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.rot90(x, k=tf.random.uniform([], 0, 4, dtype=tf.int32))   # satellite images have no "up"
    return x, y


def make_ds(X, y, train):
    ds = tf.data.Dataset.from_tensor_slices((X, y.astype("int32")))
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32), b), num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        ds = ds.shuffle(len(y), seed=42).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(d["X_train"], d["y_train"], True)
val_ds = make_ds(d["X_val"], d["y_val"], False)
test_ds = make_ds(d["X_test"], d["y_test"], False)

# class weights handle imbalance
counts = np.bincount(d["y_train"], minlength=2)
class_weight = {i: float(counts.sum() / (2.0 * counts[i])) for i in range(2)}
print("class weights:", class_weight)

# ---- Model: resize + scale + MobileNetV2 (ImageNet weights) + new 2-class layer
base = keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
                                      weights="imagenet")
inputs = keras.Input(shape=(WIN, WIN, 3))                    # raw RGB values 0-255
x = layers.Resizing(IMG_SIZE, IMG_SIZE)(inputs)
x = layers.Rescaling(1 / 127.5, offset=-1)(x)                # MobileNetV2 expects [-1, 1]
x = base(x, training=False)                                  # keeps BatchNorm in inference mode
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs, outputs)


class BestMacroF1(keras.callbacks.Callback):
    """After each epoch compute validation macro-F1 and save the best model."""
    def __init__(self):
        super().__init__()
        self.best = -1.0
        self.yv = d["y_val"]

    def on_epoch_end(self, epoch, logs=None):
        pred = np.argmax(self.model.predict(val_ds, verbose=0), axis=1)
        f1 = f1_score(self.yv, pred, average="macro")
        print(f"   val_macroF1={f1:.4f}")
        if f1 > self.best:
            self.best = f1
            self.model.save(MODEL_TF)


ckpt = BestMacroF1()


def compile_model(lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])


print("Phase 1: train only the new classifier layer")
base.trainable = False
compile_model(1e-3)
model.fit(train_ds, validation_data=val_ds, epochs=3, class_weight=class_weight, callbacks=[ckpt])

print("Phase 2: fine-tune the whole network")
base.trainable = True
compile_model(1e-4)
model.fit(train_ds, validation_data=val_ds, epochs=8, class_weight=class_weight, callbacks=[ckpt])

# ---- Final test on spatially separated blocks, using the best saved model
best_model = keras.models.load_model(MODEL_TF)
tp = np.argmax(best_model.predict(test_ds, verbose=0), axis=1)
tt = d["y_test"]
names = ["non-cropland", "cropland"]
report = classification_report(tt, tp, target_names=names, digits=3)
print("\nTEST RESULTS\n", report)
(OUT / "test_metrics.txt").write_text(report)

cm = confusion_matrix(tt, tp)
ConfusionMatrixDisplay(cm, display_labels=names).plot(cmap="Blues", values_format="d")
plt.title("Confusion matrix (test set)")
plt.savefig(OUT / "confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved model to", MODEL_TF)