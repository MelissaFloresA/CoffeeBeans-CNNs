import argparse
import gc
import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
  sys.path.insert(0, SRC_DIR)

from data.loader import load_data
from data.preprocessing import set_seed
from models.model_builder import build_model
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from utils.config import (
    ARCHITECTURES,
    BATCH_SIZE,
    DATA_PROCESSED_DIR,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    MODELS_DIR,
    SEED,
)


# Guarda accuracy/loss por época en un dict simple, para el JSON final.
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


# Calcula pesos de clase balanceados a partir del conteo real de train.
def compute_balanced_class_weights(class_names):
  train_dir = os.path.join(DATA_PROCESSED_DIR, "train")
  counts = {
      cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in class_names
  }
  print(f"Conteo real de imágenes por clase (train): {counts}")

  empty_classes = [cls for cls, n in counts.items() if n == 0]
  if empty_classes:
    raise RuntimeError(
        f"Las siguientes clases no tienen imágenes en {train_dir}:"
        f" {empty_classes}. Vuelve a correr 'python data/prepare_dataset.py'"
        f" (limpia data/processed/ por completo antes de reconstruirlo)."
    )

  y_train = np.concatenate([
      np.full(counts[cls], idx) for idx, cls in enumerate(class_names)
  ])

  weights = compute_class_weight(
      class_weight="balanced",
      classes=np.arange(len(class_names)),
      y=y_train,
  )
  class_weights = {i: float(w) for i, w in enumerate(weights)}
  print(f"Pesos de clase calculados: {class_weights}")
  return class_weights


# Entrena una arquitectura con los hiperparámetros de utils/config.py y
# guarda pesos + config en models/<architecture>/.
def train_architecture(architecture="mobilenet"):
  set_seed(SEED)

  # Libera memoria de GPU entre arquitecturas (relevante al entrenar las
  # 4 en el mismo proceso).
  tf.keras.backend.clear_session()
  gc.collect()

  train_ds = load_data("train", batch_size=BATCH_SIZE)
  val_ds = load_data("val", batch_size=BATCH_SIZE)

  class_names = train_ds.class_names
  print(f"\nClases oficiales detectadas: {class_names}")

  class_weights = compute_balanced_class_weights(class_names)
  print(f"Entrenando arquitectura: {architecture.upper()}")

  model = build_model(architecture, is_training=True)

  print(f"\n=== ENTRENAMIENTO ({architecture.upper()}) ===")
  print(
      f"Hiperparámetros compartidos: lr={LEARNING_RATE}, epochs={EPOCHS},"
      f" batch_size={BATCH_SIZE}, label_smoothing={LABEL_SMOOTHING}"
  )
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
      loss=tf.keras.losses.CategoricalCrossentropy(
          label_smoothing=LABEL_SMOOTHING
      ),
      metrics=["accuracy"],
  )

  save_dir = os.path.join(MODELS_DIR, architecture)
  os.makedirs(save_dir, exist_ok=True)

  weights_path = os.path.join(save_dir, "weights.h5")
  config_path = os.path.join(save_dir, "model_config.json")

  cb_history = CleanHistoryCallback()

  # EarlyStopping (restaura mejores pesos según val_loss). Se
  # descartó ReduceLROnPlateau: no mejoraba val_loss en pruebas.
  callbacks = [
      tf.keras.callbacks.EarlyStopping(
          patience=EARLY_STOPPING_PATIENCE,
          restore_best_weights=True,
          monitor="val_loss",
      ),
      cb_history,
  ]

  model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=EPOCHS,
      class_weight=class_weights,
      callbacks=callbacks,
      verbose=1,
  )

  model.save_weights(weights_path)

  config_data = {
      "architecture": architecture,
      "class_names": class_names,
      "class_weights": class_weights,
      "hyperparameters": {
          "learning_rate": LEARNING_RATE,
          "epochs_max": EPOCHS,
          "batch_size": BATCH_SIZE,
          "label_smoothing": LABEL_SMOOTHING,
      },
      "history": cb_history.history,
  }

  with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config_data, f, indent=4)

  print(f"\nAprobado. Guardado en: {save_dir}")


# Entrena las 4 arquitecturas en secuencia con la misma config.
def train_all_architectures():
  for architecture in ARCHITECTURES:
    print("\n" + "#" * 70)
    print(f"# Entrenando: {architecture.upper()}")
    print("#" * 70)
    train_architecture(architecture)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Entrena una o todas las arquitecturas del proyecto."
  )
  parser.add_argument(
      "--architecture",
      "-a",
      choices=ARCHITECTURES,
      default="mobilenet",
      help="Arquitectura a entrenar (ignorado si se usa --all).",
  )
  parser.add_argument(
      "--all",
      action="store_true",
      help="Entrena las 4 arquitecturas (mobilenet, resnet, efficientnet, vgg)"
      " en secuencia, con los mismos hiperparámetros.",
  )
  args = parser.parse_args()

  if args.all:
    train_all_architectures()
  else:
    train_architecture(args.architecture)
