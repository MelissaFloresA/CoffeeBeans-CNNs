import os
import random
import shutil
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
  sys.path.insert(0, SRC_DIR)

from data.preprocessing import set_seed
from utils.config import DATA_DIR, DATA_PROCESSED_DIR, SEED

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _dir_has_images(dir_path):
  """Una carpeta solo cuenta como clase si tiene al menos una imagen."""
  return any(
      f.lower().endswith(IMAGE_EXTENSIONS)
      for f in os.listdir(dir_path)
      if os.path.isfile(os.path.join(dir_path, f))
  )


def prepare_dataset(source_data_dir, train_ratio=0.70, val_ratio=0.15):
  """Separa las imágenes crudas en train/val/test (copia de archivos,
  sin procesarlas). El preprocesamiento (GrabCut, realce) lo hace
  preprocessing.py al cargar los datos, no aquí.

  train_ratio/val_ratio definen la proporción de cada split (el resto va
  a test). Con 0.70/0.15/0.15, train sigue siendo el más grande (más
  datos para aprender) pero val queda con más imágenes que antes (0.10),
  lo que da una estimación de val_accuracy menos ruidosa durante el
  entrenamiento.
  """
  if not os.path.exists(source_data_dir):
    raise FileNotFoundError(f"No se encontró el directorio de origen: {source_data_dir}")

  set_seed(SEED)

  # Limpieza total: si se corre más de una vez, no deben quedar imágenes
  # de una corrida anterior mezcladas con las nuevas (fuga de datos).
  if os.path.exists(DATA_PROCESSED_DIR):
    print(f"Limpiando partición previa en: {DATA_PROCESSED_DIR}")
    shutil.rmtree(DATA_PROCESSED_DIR)

  subsets = ["train", "val", "test"]
  classes = sorted([
      d
      for d in os.listdir(source_data_dir)
      if os.path.isdir(os.path.join(source_data_dir, d))
      and _dir_has_images(os.path.join(source_data_dir, d))
  ])

  if not classes:
    raise ValueError(
        f"No se encontraron subcarpetas de clases con imágenes dentro de"
        f" {source_data_dir}. Revisa que la ruta no esté anidada dos veces."
    )

  for subset in subsets:
    for cls in classes:
      os.makedirs(os.path.join(DATA_PROCESSED_DIR, subset, cls), exist_ok=True)

  print(f"Clases encontradas: {classes}")

  for cls in classes:
    cls_src_path = os.path.join(source_data_dir, cls)
    images = [f for f in os.listdir(cls_src_path) if f.lower().endswith(IMAGE_EXTENSIONS)]
    random.shuffle(images)

    train_idx = int(len(images) * train_ratio)
    val_idx = int(len(images) * (train_ratio + val_ratio))
    splits = {
        "train": images[:train_idx],
        "val": images[train_idx:val_idx],
        "test": images[val_idx:],
    }

    for subset, img_list in splits.items():
      dest_folder = os.path.join(DATA_PROCESSED_DIR, subset, cls)
      for img_name in img_list:
        shutil.copy2(os.path.join(cls_src_path, img_name), os.path.join(dest_folder, img_name))

    print(
        f"Clase '{cls}': {len(splits['train'])} train, {len(splits['val'])} val,"
        f" {len(splits['test'])} test."
    )

  print(f"\n[OK] Dataset separado en: {DATA_PROCESSED_DIR}")


if __name__ == "__main__":
  raw_dataset_path = os.path.join(DATA_DIR, "raw", "coffee_union")
  prepare_dataset(raw_dataset_path)
