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
    REDUCE_LR_PATIENCE,
    SEED,
)


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


def compute_balanced_class_weights(class_names):
  """Calcula pesos de clase reales a partir del conteo de imágenes en
  data/processed/train/<clase> (fórmula 'balanced' de sklearn)."""
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


def train_architecture(architecture="mobilenet"):
  """Entrena una arquitectura usando SIEMPRE los mismos hiperparámetros
  compartidos definidos en utils/config.py (LEARNING_RATE, EPOCHS,
  LABEL_SMOOTHING, paciencias de callbacks, class weights calculados de la
  misma forma). Esto es lo que garantiza que comparar mobilenet vs resnet
  vs efficientnet vs vgg sea una comparación justa: la única variable que
  cambia entre corridas es la arquitectura en sí.
  """
  set_seed(SEED)

  # *** FIX: liberar memoria de GPU entre arquitecturas ***
  # Cuando train_all_architectures() entrena las 4 arquitecturas en el
  # mismo proceso, Keras mantiene un grafo/sesión global: cada modelo que
  # se construye (MobileNet, luego ResNet, luego EfficientNet...) deja
  # residuos en memoria de GPU que NO se liberan solos al terminar esa
  # función. Para cuando le toca a VGG (el más pesado en memoria, al ser
  # el 4to), ya queda mucha menos VRAM libre de la que debería, y termina
  # en ResourceExhaustedError aunque VGG solo, entrenado desde cero,
  # probablemente habría entrado sin problema. clear_session() resetea
  # ese estado global antes de construir el modelo nuevo.
  tf.keras.backend.clear_session()
  gc.collect()

  train_ds = load_data("train", batch_size=BATCH_SIZE)  # augment=True por defecto
  val_ds = load_data("val", batch_size=BATCH_SIZE)  # augment=False por defecto

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

  # Un solo criterio de guardado: EarlyStopping restaura los mejores
  # pesos según val_loss, y esos son los que se guardan al final.
  callbacks = [
      tf.keras.callbacks.ReduceLROnPlateau(
          monitor="val_loss",
          factor=0.5,
          patience=REDUCE_LR_PATIENCE,
          min_lr=1e-6,
          verbose=1,
      ),
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

  print(f"\nEntrenamiento exitoso. Guardado en: {save_dir}")


def train_all_architectures():
  """Entrena las 4 arquitecturas, una tras otra, con la MISMA config
  (viene de utils/config.py). Es el modo recomendado para producir una
  comparación homogénea con compare_architectures.py."""
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
