import argparse
from datetime import datetime
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
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
import tensorflow as tf
from utils.config import ARCHITECTURES, BATCH_SIZE, FIGURES_DIR, MODELS_DIR, REPORTS_DIR


def _load_config(architecture):
  model_dir = os.path.join(MODELS_DIR, architecture)
  weights_path = os.path.join(model_dir, "weights.h5")
  config_path = os.path.join(model_dir, "model_config.json")

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

  return weights_path, class_names, history


def _plot_confusion_matrix(cm, class_names, architecture):
  """Matriz de confusión con matplotlib puro (sin seaborn)."""
  fig, ax = plt.subplots(figsize=(8, 6))
  im = ax.imshow(cm, cmap="Blues")
  fig.colorbar(im, ax=ax)

  ax.set_xticks(np.arange(len(class_names)))
  ax.set_yticks(np.arange(len(class_names)))
  ax.set_xticklabels(class_names, rotation=45, ha="right")
  ax.set_yticklabels(class_names)
  ax.set_xlabel("Predicción", fontsize=12)
  ax.set_ylabel("Clase Real", fontsize=12)
  ax.set_title(
      f"Matriz de Confusión - {architecture.upper()}",
      fontsize=14,
      fontweight="bold",
  )

  threshold = cm.max() / 2.0 if cm.max() > 0 else 0
  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
      ax.text(
          j,
          i,
          format(cm[i, j], "d"),
          ha="center",
          va="center",
          color="white" if cm[i, j] > threshold else "black",
          fontsize=11,
      )

  fig.tight_layout()
  os.makedirs(FIGURES_DIR, exist_ok=True)
  cm_path = os.path.join(FIGURES_DIR, f"cm_{architecture}.png")
  fig.savefig(cm_path, dpi=300)
  plt.close(fig)
  return cm_path


def _plot_training_history(history, architecture):
  if not (history and "accuracy" in history and "val_accuracy" in history):
    print(
        "Aviso: no se encontró historial detallado en el JSON para graficar"
        " Loss/Accuracy."
    )
    return None

  fig = plt.figure(figsize=(12, 5))

  plt.subplot(1, 2, 1)
  plt.plot(history["accuracy"], label="Train Accuracy", linewidth=2, color="royalblue")
  plt.plot(history["val_accuracy"], label="Val Accuracy", linewidth=2, color="darkorange")
  plt.title(f"Precisión del Modelo - {architecture.upper()}", fontsize=12, fontweight="bold")
  plt.xlabel("Épocas", fontsize=10)
  plt.ylabel("Accuracy", fontsize=10)
  plt.legend()
  plt.grid(True, linestyle="--", alpha=0.6)

  plt.subplot(1, 2, 2)
  plt.plot(history["loss"], label="Train Loss", linewidth=2, color="royalblue")
  plt.plot(history["val_loss"], label="Val Loss", linewidth=2, color="darkorange")
  plt.title(f"Pérdida del Modelo - {architecture.upper()}", fontsize=12, fontweight="bold")
  plt.xlabel("Épocas", fontsize=10)
  plt.ylabel("Loss", fontsize=10)
  plt.legend()
  plt.grid(True, linestyle="--", alpha=0.6)

  plt.tight_layout()
  os.makedirs(FIGURES_DIR, exist_ok=True)
  history_path = os.path.join(FIGURES_DIR, f"history_{architecture}.png")
  plt.savefig(history_path, dpi=300)
  plt.close(fig)
  return history_path


def evaluate(architecture="mobilenet"):
  """Evalúa una arquitectura sobre el set de test y devuelve un dict con
  todas las métricas (para que compare_architectures.py pueda reutilizarlo
  sin volver a parsear texto)."""
  test_ds = load_data("test", batch_size=BATCH_SIZE, augment=False)

  # Libera memoria de GPU de cualquier modelo construido antes en este
  # mismo proceso (relevante cuando se evalúan varias arquitecturas
  # seguidas con --all o desde compare_architectures.py).
  tf.keras.backend.clear_session()

  weights_path, class_names, history = _load_config(architecture)

  model = build_model(architecture, is_training=False)

  if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print(f"Pesos cargados exitosamente desde: {weights_path}")
  else:
    raise FileNotFoundError(f"No se encontraron pesos en: {weights_path}")

  y_true, y_pred = [], []
  print(f"Generando predicciones sobre Test para: {architecture.upper()}...")

  for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

  y_true = np.array(y_true)
  y_pred = np.array(y_pred)

  # --- Métricas ---
  accuracy = accuracy_score(y_true, y_pred)

  precision_per_class, recall_per_class, f1_per_class, support_per_class = (
      precision_recall_fscore_support(
          y_true, y_pred, labels=range(len(class_names)), zero_division=0
      )
  )
  precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
      y_true, y_pred, average="macro", zero_division=0
  )

  per_class_df = pd.DataFrame(
      {
          "precision": precision_per_class,
          "recall": recall_per_class,
          "f1_score": f1_per_class,
          "support": support_per_class,
      },
      index=class_names,
  ).round(4)
  per_class_df.loc["MACRO AVG"] = [
      round(precision_macro, 4),
      round(recall_macro, 4),
      round(f1_macro, 4),
      int(support_per_class.sum()),
  ]

  cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
  cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

  # --- Gráficos (matplotlib puro) ---
  cm_path = _plot_confusion_matrix(cm, class_names, architecture)
  history_path = _plot_training_history(history, architecture)

  # --- Reporte de texto (incluye la matriz de confusión "simulada" como
  # tabla, gracias a pandas.DataFrame.to_string) ---
  os.makedirs(REPORTS_DIR, exist_ok=True)
  report_path = os.path.join(REPORTS_DIR, f"reporte_{architecture}.txt")

  lines = []
  lines.append("=" * 70)
  lines.append(f"REPORTE DE EVALUACIÓN - {architecture.upper()}")
  lines.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
  lines.append("=" * 70)
  lines.append("")
  lines.append(f"Accuracy general (test): {accuracy:.4f} ({accuracy*100:.2f}%)")
  lines.append(f"Precision macro:         {precision_macro:.4f}")
  lines.append(f"Recall macro:            {recall_macro:.4f}")
  lines.append(f"F1-score macro:          {f1_macro:.4f}")
  lines.append(f"Total de imágenes test:  {int(support_per_class.sum())}")
  lines.append("")
  lines.append("-" * 70)
  lines.append("MÉTRICAS POR CLASE")
  lines.append("-" * 70)
  lines.append(per_class_df.to_string())
  lines.append("")
  lines.append("-" * 70)
  lines.append("MATRIZ DE CONFUSIÓN (filas = clase real, columnas = predicción)")
  lines.append("-" * 70)
  lines.append(cm_df.to_string())
  lines.append("")
  report_text = "\n".join(lines)

  with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

  per_class_df.to_csv(os.path.join(REPORTS_DIR, f"metricas_{architecture}.csv"))
  cm_df.to_csv(os.path.join(REPORTS_DIR, f"matriz_confusion_{architecture}.csv"))

  print("\n" + report_text)
  print(f"\nReporte guardado en: {report_path}")
  print(f"Matriz de confusión (imagen) guardada en: {cm_path}")
  if history_path:
    print(f"Gráfico de entrenamiento guardado en: {history_path}")

  return {
      "architecture": architecture,
      "accuracy": accuracy,
      "precision_macro": precision_macro,
      "recall_macro": recall_macro,
      "f1_macro": f1_macro,
      "per_class": per_class_df,
      "confusion_matrix": cm_df,
      "class_names": class_names,
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Evalúa una o todas las arquitecturas sobre el set de test."
  )
  parser.add_argument(
      "--architecture", "-a", choices=ARCHITECTURES, default="mobilenet"
  )
  parser.add_argument(
      "--all",
      action="store_true",
      help="Evalúa las 4 arquitecturas en secuencia (requiere que ya estén"
      " entrenadas).",
  )
  args = parser.parse_args()

  if args.all:
    for arch in ARCHITECTURES:
      print("\n" + "#" * 70)
      print(f"# Evaluando: {arch.upper()}")
      print("#" * 70)
      evaluate(arch)
  else:
    evaluate(args.architecture)
