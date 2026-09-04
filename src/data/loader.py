import os

import cv2
import numpy as np
import tensorflow as tf

from data.preprocessing import get_data_augmentation, preprocess_image
from utils.config import BATCH_SIZE, DATA_PROCESSED_DIR, IMG_SIZE, SEED

AUTOTUNE = tf.data.AUTOTUNE


def _preprocess_tf(image, label):
  """Envuelve preprocess_image (OpenCV, no-TF) para usarla en tf.data.
  tf.py_function permite llamar código Python/NumPy arbitrario dentro
  del pipeline; se pierde el shape estático, por eso se restablece con
  set_shape al final."""

  def _run(img):
    img_bgr = cv2.cvtColor(img.numpy().astype("uint8"), cv2.COLOR_RGB2BGR)
    processed_bgr = preprocess_image(img_bgr)
    return cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB).astype("float32")

  processed = tf.py_function(_run, [image], tf.float32)
  processed.set_shape(IMG_SIZE + (3,))
  return processed, label


def load_data(subset="train", batch_size=BATCH_SIZE, augment=None):
  """Carga un subset (train/val/test), aplica preprocess_image a cada
  imagen (realce, ver preprocessing.py), cachea el resultado
  para no repetirlo en cada época, y recién ahí aplica el aumento de
  datos aleatorio (train) y arma los batches."""
  subset_dir = os.path.join(DATA_PROCESSED_DIR, subset)
  if not os.path.exists(subset_dir):
    raise FileNotFoundError(f"El directorio no existe: {subset_dir}")

  is_train = subset == "train"
  if augment is None:
    augment = is_train

  # batch_size=None: se preprocesa imagen por imagen, se batchea al final.
  dataset = tf.keras.utils.image_dataset_from_directory(
      subset_dir,
      labels="inferred",
      label_mode="categorical",
      image_size=IMG_SIZE,
      batch_size=None,
      shuffle=is_train,
      seed=SEED if is_train else None,
  )
  class_names = dataset.class_names

  dataset = dataset.map(_preprocess_tf, num_parallel_calls=AUTOTUNE)
  dataset = dataset.cache()  # el realce solo corre una vez, no cada época

  if augment:
    augmentation = get_data_augmentation()
    dataset = dataset.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )

  dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
  dataset.class_names = class_names
  return dataset
