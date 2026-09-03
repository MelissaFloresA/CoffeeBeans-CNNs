import os
import tensorflow as tf
from utils.config import DATA_PROCESSED_DIR


def load_data(subset="train", batch_size=16):
  subset_dir = os.path.join(DATA_PROCESSED_DIR, subset)

  if not os.path.exists(subset_dir):
    raise FileNotFoundError(f"El directorio no existe: {subset_dir}")

  dataset = tf.keras.utils.image_dataset_from_directory(
      subset_dir,
      labels="inferred",
      label_mode="categorical",
      image_size=(224, 224),
      batch_size=batch_size,
      shuffle=(subset == "train"),
  )

  return dataset