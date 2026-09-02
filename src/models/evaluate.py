import os
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from data.loader import load_data
from models.model_builder import build_model
from utils.config import DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR

def evaluate(architecture="mobilenet"):
    test_path = os.path.join(DATA_PROCESSED_DIR, "test")
    class_names = sorted(os.listdir(test_path))
    test_ds = load_data("test")

    model_dir = os.path.join(MODELS_DIR, architecture)
    weights_path = os.path.join(model_dir, "weights.h5")
    model_path_h5 = os.path.join(model_dir, "model.h5")
    config_path = os.path.join(model_dir, "model_config.json")

    model = build_model(architecture)

    if os.path.exists(weights_path):
        print(f"Cargando pesos desde: {weights_path}")
        model.load_weights(weights_path)
    elif os.path.exists(model_path_h5):
        print(f"Cargando modelo completo desde: {model_path_h5}")
        model = tf.keras.models.load_model(model_path_h5)
    else:
        raise FileNotFoundError(f"No se encontraron pesos ni modelo en: {model_dir}")

    y_true, y_pred = [], []
    print(f"Generando predicciones para {architecture.upper()}...")

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print("\n" + "=" * 60)
    print(f"       MÉTRICAS DE CLASIFICACIÓN POR CLASE ({architecture.upper()})")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    fig_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Matriz de Confusión - {architecture.upper()}", fontsize=14, fontweight="bold")
    plt.xlabel("Predicción", fontsize=12)
    plt.ylabel("Clase Real", fontsize=12)
    plt.tight_layout()

    cm_path = os.path.join(fig_dir, f"cm_{architecture}.png")
    plt.savefig(cm_path, dpi=300)
    plt.show()

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            history = config_data.get("history", {})
            if history and "accuracy" in history:
                epochs_range = range(1, len(history["accuracy"]) + 1)

                plt.figure(figsize=(14, 5))

                plt.subplot(1, 2, 1)
                plt.plot(epochs_range, history["accuracy"], label="Entrenamiento", linewidth=2)
                plt.plot(epochs_range, history["val_accuracy"], label="Validación", linewidth=2)
                plt.title("Exactitud (Accuracy)", fontsize=12, fontweight="bold")
                plt.xlabel("Épocas")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.6)

                plt.subplot(1, 2, 2)
                plt.plot(epochs_range, history["loss"], label="Entrenamiento", linewidth=2)
                plt.plot(epochs_range, history["val_loss"], label="Validación", linewidth=2)
                plt.title("Pérdida (Loss)", fontsize=12, fontweight="bold")
                plt.xlabel("Épocas")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.6)

                plt.suptitle(f"Curvas de Aprendizaje - {architecture.upper()}", fontsize=14, fontweight="bold")
                plt.tight_layout()

                curves_path = os.path.join(fig_dir, f"learning_curves_{architecture}.png")
                plt.savefig(curves_path, dpi=300)
                plt.show()
        except Exception as e:
            print(f"Error al graficar curvas: {e}")

if __name__ == "__main__":
    # Cambia el string para evaluar cada arquitectura: "mobilenet", "efficientnet", "resnet", "vgg"
    evaluate("mobilenet")