import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
  sys.path.insert(0, SRC_DIR)

from data.loader import load_data
from data.preprocessing import CoffeeFeatureEnhancer, get_data_augmentation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from utils.config import DATA_PROCESSED_DIR, RESULTS_DIR


def generate_pipeline_reports(architecture="mobilenet"):
  fig_dir = os.path.join(RESULTS_DIR, "figures")
  os.makedirs(fig_dir, exist_ok=True)

  train_path = os.path.join(DATA_PROCESSED_DIR, "train")
  class_names = sorted(os.listdir(train_path))
  counts = [
      len(os.listdir(os.path.join(train_path, c))) for c in class_names
  ]

  plt.figure(figsize=(10, 5))
  bars = plt.bar(
      class_names, counts, color=sns.color_palette("Blues_d", len(class_names))
  )
  plt.title("Distribucion de Clases - Train", fontsize=14, fontweight="bold")
  plt.xlabel("Clase", fontsize=12)
  plt.ylabel("Cantidad", fontsize=12)
  plt.grid(axis="y", linestyle="--", alpha=0.6)

  for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 2,
        int(yval),
        ha="center",
        va="bottom",
        fontweight="bold",
    )

  balance_path = os.path.join(fig_dir, "balance_clases_train.png")
  plt.savefig(balance_path, dpi=300, bbox_inches="tight")
  plt.close()

  train_ds = load_data("train")
  for images, _ in train_ds.take(1):
    sample_images = images.numpy()
    break

  augmentation = get_data_augmentation()
  single_img = tf.expand_dims(sample_images[0], 0)

  plt.figure(figsize=(12, 9))
  plt.suptitle(
      "Demostracion de Data Augmentation", fontsize=14, fontweight="bold"
  )

  for i in range(12):
    augmented = augmentation(single_img, training=True)
    plt.subplot(3, 4, i + 1)
    plt.imshow(np.array(augmented[0]).astype("uint8"))
    plt.title(f"Aumento {i+1}", fontsize=10)
    plt.axis("off")

  aug_path = os.path.join(fig_dir, "demostracion_data_augmentation.png")
  plt.savefig(aug_path, dpi=300, bbox_inches="tight")
  plt.close()

  enhancer = CoffeeFeatureEnhancer()

  plt.figure(figsize=(15, 6))
  plt.suptitle(
      f"Comparativa Original vs CLAHE + Nitidez ({architecture.upper()})",
      fontsize=14,
      fontweight="bold",
  )

  for i in range(5):
    orig = sample_images[i]
    prep_tensor = enhancer(tf.expand_dims(orig, 0))[0].numpy()
    prep_disp = prep_tensor.astype("uint8")

    plt.subplot(2, 5, i + 1)
    plt.imshow(orig.astype("uint8"))
    plt.title(f"Original {i+1}", fontsize=10)
    plt.axis("off")

    plt.subplot(2, 5, i + 6)
    plt.imshow(prep_disp)
    plt.title(f"Procesada {i+1}", fontsize=10)
    plt.axis("off")

  prep_path = os.path.join(
      fig_dir, f"comparativa_preprocesamiento_{architecture}.png"
  )
  plt.savefig(prep_path, dpi=300, bbox_inches="tight")
  plt.close()
  print(f"Reportes visuales generados con exito en {fig_dir}")


if __name__ == "__main__":
  generate_pipeline_reports("mobilenet")