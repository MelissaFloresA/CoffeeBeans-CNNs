import os
import random
import shutil
import sys
import tensorflow as tf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
  sys.path.insert(0, SRC_DIR)

from utils.config import DATA_DIR, DATA_PROCESSED_DIR


def prepare_dataset(source_data_dir, train_ratio=0.8, val_ratio=0.1):
  """Divide un directorio de clases crudas en carpetas train, val y test

  dentro de data/processed/, redimensionando las imágenes a 224x224.
  """
  if not os.path.exists(source_data_dir):
    raise FileNotFoundError(
        f"No se encontró el directorio de origen: {source_data_dir}"
    )

  subsets = ["train", "val", "test"]
  classes = sorted([
      d
      for d in os.listdir(source_data_dir)
      if os.path.isdir(os.path.join(source_data_dir, d))
  ])

  for subset in subsets:
    for cls in classes:
      os.makedirs(os.path.join(DATA_PROCESSED_DIR, subset, cls), exist_ok=True)

  print(f"Procesando clases encontradas: {classes}")

  for cls in classes:
    cls_src_path = os.path.join(source_data_dir, cls)
    images = [
        f
        for f in os.listdir(cls_src_path)
        if f.lower().endswith(("jpg", "jpeg", "png", "bmp", "webp"))
    ]

    random.shuffle(images)
    total_images = len(images)

    train_idx = int(total_images * train_ratio)
    val_idx = int(total_images * (train_ratio + val_ratio))

    train_imgs = images[:train_idx]
    val_imgs = images[train_idx:val_idx]
    test_imgs = images[val_idx:]

    splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

    for subset, img_list in splits.items():
      dest_folder = os.path.join(DATA_PROCESSED_DIR, subset, cls)
      for img_name in img_list:
        src_img_path = os.path.join(cls_src_path, img_name)
        dest_img_path = os.path.join(dest_folder, img_name)

        try:
          img = tf.keras.utils.load_img(src_img_path, target_size=(224, 224))
          img.save(dest_img_path)
        except Exception as e:
          print(f"Error procesando {src_img_path}: {e}")

    print(
        f"Clase '{cls}': {len(train_imgs)} train, {len(val_imgs)} val,"
        f" {len(test_imgs)} test."
    )

  print(f"\n[OK] Dataset preparado exitosamente en: {DATA_PROCESSED_DIR}")


if __name__ == "__main__":
  raw_dataset_path = os.path.join(DATA_DIR, "raw")

  if not os.path.exists(raw_dataset_path):
    raw_dataset_path = r"C:\Users\Melissa\Desktop\Vision\CoffeeBeans-CNNs\data\raw\coffee_union"

  prepare_dataset(raw_dataset_path)