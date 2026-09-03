import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
  sys.path.insert(0, SRC_DIR)

from data.loader import load_data
import matplotlib.pyplot as plt
from models.model_builder import build_model
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from utils.config import MODELS_DIR, RESULTS_DIR


def evaluate(architecture="mobilenet"):
  test_ds = load_data("test")

  model_dir = os.path.join(MODELS_DIR, architecture)
  weights_path = os.path.join(model_dir, "weights.h5")
  config_path = os.path.join(model_dir, "model_config.json")

  # Cargar configuración, clases e historial desde el JSON
  if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
      config_data = json.load(f)
      class_names = config_data.get(
          "class_names",
          ["Black", "Broken", "Immature", "Insect Damage", "Premium"],
      )
      history = config_data.get("history", None)
  else:
    class_names = ["Black", "Broken", "Immature", "Insect Damage", "Premium"]
    history = None

  model = build_model(architecture, is_training=False)

  if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print(f"Pesos cargados exitosamente desde: {weights_path}")
  else:
    raise FileNotFoundError(f"No se encontraron pesos en: {weights_path}")

  y_true, y_pred = [], []
  print(
      f"Generando predicciones sobre Test para la arquitectura:"
      f" {architecture.upper()}..."
  )

  for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

  # 1. Reporte en texto
  report = classification_report(
      y_true, y_pred, target_names=class_names, zero_division=0
  )
  print("\n" + "=" * 60)
  print(f"       MÉTRICAS DE CLASIFICACIÓN ({architecture.upper()})")
  print("=" * 60)
  print(report)

  # Directorio para guardar gráficos
  fig_dir = os.path.join(RESULTS_DIR, "figures")
  os.makedirs(fig_dir, exist_ok=True)

  # 2. Generar y guardar la Matriz de Confusión
  cm = confusion_matrix(y_true, y_pred)
  plt.figure(figsize=(8, 6))
  sns.heatmap(
      cm,
      annot=True,
      fmt="d",
      cmap="Blues",
      xticklabels=class_names,
      yticklabels=class_names,
  )
  plt.title(
      f"Matriz de Confusión - {architecture.upper()}",
      fontsize=14,
      fontweight="bold",
  )
  plt.xlabel("Predicción", fontsize=12)
  plt.ylabel("Clase Real", fontsize=12)
  plt.tight_layout()

  cm_path = os.path.join(fig_dir, f"cm_{architecture}.png")
  plt.savefig(cm_path, dpi=300)
  plt.close()
  print(f"Matriz de confusión guardada en: {cm_path}")

  # 3. Generar y guardar las curvas de Loss y Accuracy (si el historial existe en el JSON)
  if history and "accuracy" in history and "val_accuracy" in history:
    plt.figure(figsize=(12, 5))

    # Gráfica de Precisión (Accuracy)
    plt.subplot(1, 2, 1)
    plt.plot(
        history["accuracy"],
        label="Train Accuracy",
        linewidth=2,
        color="royalblue",
    )
    plt.plot(
        history["val_accuracy"],
        label="Val Accuracy",
        linewidth=2,
        color="darkorange",
    )
    plt.title(
        f"Precisión del Modelo - {architecture.upper()}",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Épocas", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Gráfica de Pérdida (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Train Loss", linewidth=2, color="royalblue")
    plt.plot(
        history["val_loss"], label="Val Loss", linewidth=2, color="darkorange"
    )
    plt.title(
        f"Pérdida del Modelo - {architecture.upper()}",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Épocas", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    history_path = os.path.join(fig_dir, f"history_{architecture}.png")
    plt.savefig(history_path, dpi=300)
    plt.close()
    print(f"Gráfico de entrenamiento (Loss/Accuracy) guardado en: {history_path}")
  else:
    print(
        "Aviso: No se encontró historial detallado en el JSON para graficar"
        " Loss/Accuracy."
    )


if __name__ == "__main__":
  evaluate("mobilenet")