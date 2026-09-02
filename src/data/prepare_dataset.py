import os
import sys
import shutil
import zipfile
import tempfile

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import splitfolders
from utils.config import DATA_RAW_DIR, DATA_PROCESSED_DIR

def process_zip_to_processed(zip_name="coffee_union.zip"):
    zip_path = os.path.join(DATA_RAW_DIR, zip_name)

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"No se encontró el archivo '{zip_name}' en: {DATA_RAW_DIR}")

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"1. Descomprimiendo '{zip_name}'...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        extracted_items = [item for item in os.listdir(temp_dir) if not item.startswith('__MACOSX')]
        source_dir = temp_dir

        if len(extracted_items) == 1 and os.path.isdir(os.path.join(temp_dir, extracted_items[0])):
            source_dir = os.path.join(temp_dir, extracted_items[0])

        print(f"Clases detectadas: {sorted(os.listdir(source_dir))}")

        if os.path.exists(DATA_PROCESSED_DIR):
            print(f"Limpiando directorio previo: {DATA_PROCESSED_DIR}")
            shutil.rmtree(DATA_PROCESSED_DIR)

        print("2. Dividiendo dataset (80% Train, 10% Val, 10% Test)...")
        splitfolders.ratio(
            source_dir,
            output=DATA_PROCESSED_DIR,
            seed=42,
            ratio=(0.8, 0.1, 0.1),
            group_prefix=None,
            move=False
        )

    print("\n" + "="*50)
    print(" ¡PROCESAMIENTO Y DIVISIÓN COMPLETADOS CON ÉXITO!")
    print("="*50)

    for split in ["train", "val", "test"]:
        split_path = os.path.join(DATA_PROCESSED_DIR, split)
        print(f"\n--- {split.upper()} ---")
        if os.path.exists(split_path):
            for cls in sorted(os.listdir(split_path)):
                cls_path = os.path.join(split_path, cls)
                if os.path.isdir(cls_path):
                    print(f"  {cls}: {len(os.listdir(cls_path))} imágenes")

if __name__ == "__main__":
    process_zip_to_processed("coffee_union.zip")