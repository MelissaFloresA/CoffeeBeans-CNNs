import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import tensorflow as tf
from utils.config import BATCH_SIZE, DATA_PROCESSED_DIR, IMG_SIZE


def load_data(split='train'):
    split_dir = os.path.join(DATA_PROCESSED_DIR, split)

    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"No se encontró el directorio: {split_dir}")

    shuffle = True if split == 'train' else False

    dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=shuffle,
    )

    # Cast obligatorio a float32 para asegurar compatibilidad con capas Keras
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), y))
    
    # Cache y prefetch para acelera el pipeline en GPU
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)