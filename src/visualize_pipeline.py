import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data.loader import load_data
from data.preprocessing import get_data_augmentation, preprocess_input_by_architecture
from utils.config import DATA_PROCESSED_DIR, FIGURES_DIR


def generate_pipeline_reports(architecture="mobilenet"):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Distribución Total del Dataset
    splits = ["train", "val", "test"]
    class_names = sorted(os.listdir(os.path.join(DATA_PROCESSED_DIR, "train")))
    total_counts = {c: 0 for c in class_names}

    for split in splits:
        split_path = os.path.join(DATA_PROCESSED_DIR, split)
        if os.path.exists(split_path):
            for c in class_names:
                c_dir = os.path.join(split_path, c)
                if os.path.exists(c_dir):
                    total_counts[c] += len(os.listdir(c_dir))

    counts = [total_counts[c] for c in class_names]

    # Colores generados con el colormap de matplotlib (sin depender de seaborn)
    bar_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(class_names)))

    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_names, counts, color=bar_colors)
    plt.title("Distribución Total del Dataset (Train + Val + Test)", fontsize=14, fontweight="bold")
    plt.xlabel("Clase", fontsize=12)
    plt.ylabel("Cantidad Total de Imágenes", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 5,
            int(yval),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    balance_path = os.path.join(FIGURES_DIR, "distribucion_total_dataset.png")
    plt.savefig(balance_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Selección de 1 Muestra por Clase (augment=False: imágenes ya
    # aisladas/realzadas tal como quedaron guardadas por prepare_dataset.py,
    # sin aumento de datos adicional)
    train_ds = load_data("train", augment=False)
    samples_per_class = {}

    for images, labels in train_ds:
        labels_idx = np.argmax(labels.numpy(), axis=1)
        for img, lbl in zip(images.numpy(), labels_idx):
            c_name = class_names[lbl]
            if c_name not in samples_per_class:
                samples_per_class[c_name] = img
            if len(samples_per_class) == len(class_names):
                break
        if len(samples_per_class) == len(class_names):
            break

    # 3. Demostración de Augmentation
    augmentation = get_data_augmentation()
    first_sample = list(samples_per_class.values())[0]
    single_img = tf.expand_dims(first_sample, 0)

    plt.figure(figsize=(12, 9))
    plt.suptitle("Demostración de Data Augmentation", fontsize=14, fontweight="bold")

    for i in range(12):
        augmented = augmentation(single_img, training=True)
        plt.subplot(3, 4, i + 1)
        plt.imshow(np.array(augmented[0]).astype("uint8"))
        plt.title(f"Aumento {i+1}", fontsize=10)
        plt.axis("off")

    aug_path = os.path.join(FIGURES_DIR, "demostracion_data_augmentation.png")
    plt.savefig(aug_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Comparativa Aislado/Realzado (guardado en disco) vs Normalizado
    # final por arquitectura (paso obligatorio dentro del modelo)
    plt.figure(figsize=(16, 7))
    plt.suptitle(
        f"Comparativa por Clase: Aislado/Realzado vs Normalizado ({architecture.upper()})",
        fontsize=14,
        fontweight="bold",
    )

    for i, c_name in enumerate(class_names):
        orig = samples_per_class[c_name]
        prep_tensor = preprocess_input_by_architecture(tf.expand_dims(orig, 0), architecture)[0].numpy()

        if prep_tensor.min() < 0:
            prep_disp = ((prep_tensor - prep_tensor.min()) / (prep_tensor.max() - prep_tensor.min()) * 255.0)
        else:
            prep_disp = prep_tensor
        prep_disp = np.clip(prep_disp, 0, 255).astype("uint8")

        plt.subplot(2, 5, i + 1)
        plt.imshow(orig.astype("uint8"))
        plt.title(f"Aislado/Realzado: {c_name}", fontsize=10, fontweight="bold")
        plt.axis("off")

        plt.subplot(2, 5, i + 6)
        plt.imshow(prep_disp)
        plt.title(f"Normalizado: {c_name}", fontsize=10, fontweight="bold")
        plt.axis("off")

    prep_path = os.path.join(FIGURES_DIR, f"comparativa_preprocesamiento_{architecture}.png")
    plt.savefig(prep_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Reportes visuales generados con éxito en: {FIGURES_DIR}")


if __name__ == "__main__":
    generate_pipeline_reports("mobilenet")
