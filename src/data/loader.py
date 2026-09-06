import os

import cv2
import numpy as np
import tensorflow as tf

from data.preprocessing import get_data_augmentation, preprocess_image
from utils.config import BATCH_SIZE, DATA_PROCESSED_DIR, IMG_SIZE, SEED

AUTOTUNE = tf.data.AUTOTUNE


# Envuelve preprocess_image para usarla dentro de tf.data.
def _preprocess_tf(image, label):

  def _run(img):
    img_bgr = cv2.cvtColor(img.numpy().astype("uint8"), cv2.COLOR_RGB2BGR)
    processed_bgr = preprocess_image(img_bgr)
    return cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB).astype("float32")

  processed = tf.py_function(_run, [image], tf.float32)
  processed.set_shape(IMG_SIZE + (3,))
  return processed, label


# Carga un subset (train/val/test), preprocesa, cachea y arma batches.
# Usada en train.py, evaluate.py e hyperparam_search.py.
def load_data(subset="train", batch_size=BATCH_SIZE, augment=None):
  subset_dir = os.path.join(DATA_PROCESSED_DIR, subset)
  if not os.path.exists(subset_dir):
    raise FileNotFoundError(f"El directorio no existe: {subset_dir}")

  is_train = subset == "train"
  if augment is None:
    augment = is_train

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
  dataset = dataset.cache()

  if augment:
    augmentation = get_data_augmentation()
    dataset = dataset.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )

  dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
  dataset.class_names = class_names
  return dataset
