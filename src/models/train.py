import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
  sys.path.insert(0, SRC_DIR)

from data.loader import load_data
from models.model_builder import build_model
import numpy as np
import tensorflow as tf
from utils.config import MODELS_DIR


class CleanHistoryCallback(tf.keras.callbacks.Callback):

  def __init__(self):
    super().__init__()
    self.history = {
        "accuracy": [],
        "val_accuracy": [],
        "loss": [],
        "val_loss": [],
    }

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    for k in self.history.keys():
      val = logs.get(k)
      if val is not None:
        self.history[k].append(float(val))


def train_architecture(architecture="mobilenet"):
  train_ds = load_data("train")
  val_ds = load_data("val")

  class_names = train_ds.class_names
  print(f"\nClases oficiales detectadas: {class_names}")

  class_weights = {0: 1.0, 1: 1.1, 2: 1.1, 3: 1.1, 4: 0.9}
  print(f"Entrenando arquitectura: {architecture.upper()}")

  model = build_model(architecture, is_training=True)

  print(
      f"\n=== ENTRENAMIENTO ESTABLE Y ROBUSTO ({architecture.upper()}) ==="
  )
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
      loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
      metrics=["accuracy"],
  )

  save_dir = os.path.join(MODELS_DIR, architecture)
  os.makedirs(save_dir, exist_ok=True)

  weights_path = os.path.join(save_dir, "weights.h5")
  config_path = os.path.join(save_dir, "model_config.json")

  cb_history = CleanHistoryCallback()

  callbacks = [
      tf.keras.callbacks.ModelCheckpoint(
          weights_path,
          save_best_only=True,
          monitor="val_accuracy",
          save_weights_only=True,
      ),
      tf.keras.callbacks.ReduceLROnPlateau(
          monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
      ),
      tf.keras.callbacks.EarlyStopping(
          patience=5, restore_best_weights=True, monitor="val_loss"
      ),
      cb_history,
  ]

  model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=25,
      class_weight=class_weights,
      callbacks=callbacks,
      verbose=1,
  )

  model.save_weights(weights_path)

  config_data = {
      "architecture": architecture,
      "class_names": class_names,
      "history": cb_history.history,
  }

  with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config_data, f, indent=4)

  print(f"\nEntrenamiento exitoso y estable. Guardado en: {save_dir}")


if __name__ == "__main__":
  train_architecture("mobilenet")