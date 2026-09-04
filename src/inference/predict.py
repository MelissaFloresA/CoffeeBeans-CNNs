import argparse
import json
import os
import sys
import tkinter as tk
from tkinter import filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
  sys.path.insert(0, SRC_DIR)

from data.preprocessing import preprocess_image
from models.model_builder import build_model
from utils.config import ARCHITECTURES, FIGURES_DIR, IMG_SIZE, MODELS_DIR


def select_image_via_file_dialog():
  """Única forma de elegir imagen: ventana del explorador de archivos.
  No se acepta una ruta escrita a mano por línea de comandos."""
  root = tk.Tk()
  root.withdraw()
  root.attributes("-topmost", True)
  file_path = filedialog.askopenfilename(
      title="Selecciona una imagen de grano de café",
      filetypes=[("Archivos de Imagen", "*.jpg *.jpeg *.png *.bmp *.webp")],
  )
  root.destroy()
  return file_path


def _load_original_and_preprocessed(image_path):
  """original: la foto redimensionada, sin tocar, solo para mostrar.
  preprocessed: pasada por preprocess_image (mismo preprocessing.py que
  usa prepare_dataset.py) — es la que se le da al modelo."""
  img_bgr = cv2.imread(image_path)
  if img_bgr is None:
    raise ValueError(f"No se pudo leer la imagen: {image_path}")

  original_rgb = cv2.cvtColor(
      cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_LINEAR),
      cv2.COLOR_BGR2RGB,
  )

  preprocessed_bgr = preprocess_image(img_bgr)
  preprocessed_rgb = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)

  return original_rgb, preprocessed_rgb


def _save_annotated_image(original_rgb, predicted_label, confidence, out_path):
  """Dibuja la predicción sobre la imagen ORIGINAL (no la preprocesada)
  con OpenCV, para que el archivo guardado sea fácil de reconocer."""
  annotated = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR).copy()
  w = annotated.shape[1]

  label_text = f"{predicted_label} ({confidence:.1f}%)"
  (text_w, text_h), _ = cv2.getTextSize(
      label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
  )

  cv2.rectangle(annotated, (0, 0), (min(w, text_w + 16), text_h + 16), (40, 40, 40), -1)
  cv2.putText(
      annotated,
      label_text,
      (8, text_h + 8),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (255, 255, 255),
      2,
      cv2.LINE_AA,
  )

  cv2.imwrite(out_path, annotated)
  return out_path


def predict_single_image(image_path, architecture="mobilenet"):
  if not image_path or not os.path.exists(image_path):
    print("No se seleccionó ninguna imagen.")
    return None, None

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
  else:
    class_names = ["Black", "Broken", "Immature", "Insect Damage", "Premium"]

  print(f"Clases oficiales cargadas ({architecture.upper()}): {class_names}")

  model = build_model(architecture, is_training=False)

  if os.path.exists(weights_path):
    model.load_weights(weights_path)
  else:
    raise FileNotFoundError(
        f"No se encontraron pesos en: {weights_path}. Ejecuta train.py primero."
    )

  original_rgb, preprocessed_rgb = _load_original_and_preprocessed(image_path)

  # El modelo se alimenta con la versión AISLADA/REALZADA, no con la
  # original cruda — así es como se entrenó (ver prepare_dataset.py).
  img_batch = tf.expand_dims(preprocessed_rgb.astype(np.float32), 0)

  predictions = model.predict(img_batch, verbose=0)[0]
  predicted_class_idx = np.argmax(predictions)
  confidence = predictions[predicted_class_idx] * 100
  predicted_label = class_names[predicted_class_idx]

  print("\n" + "=" * 55)
  print(f"   DISTRIBUCIÓN DE PROBABILIDADES ({architecture.upper()})")
  print("=" * 55)
  for name, prob in zip(class_names, predictions):
    bar = "#" * int(prob * 20)
    print(f"  {name:<15}: {prob*100:6.2f}% {bar}")
  print("=" * 55)
  print(f"Predicción Final: {predicted_label} ({confidence:.2f}%)")

  fig, axes = plt.subplots(1, 3, figsize=(16, 5))

  axes[0].imshow(original_rgb)
  axes[0].set_title("Imagen Original", fontweight="bold")
  axes[0].axis("off")

  axes[1].imshow(preprocessed_rgb)
  axes[1].set_title("Preprocesada (contraste + nitidez)", fontweight="bold")
  axes[1].axis("off")

  y_pos = np.arange(len(class_names))
  plot_classes = class_names[::-1]
  plot_preds = (predictions * 100)[::-1]
  plot_colors = ["gray"] * len(class_names)

  axes[2].barh(y_pos, plot_preds, color=plot_colors)
  axes[2].set_yticks(y_pos)
  axes[2].set_yticklabels(plot_classes)
  axes[2].set_xlabel("Confianza (%)")
  axes[2].set_title("Desglose por Clase", fontweight="bold")
  axes[2].set_xlim(0, 100)

  for i, v in enumerate(plot_preds):
    axes[2].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)

  plt.suptitle(
      f"Predicción Final ({architecture.upper()}): {predicted_label} ({confidence:.2f}%)",
      fontsize=14,
      fontweight="bold",
      color="darkblue",
  )
  plt.tight_layout()

  os.makedirs(FIGURES_DIR, exist_ok=True)
  out_file = os.path.join(FIGURES_DIR, f"resultado_inferencia_{architecture}.png")
  plt.savefig(out_file, dpi=300)
  plt.show()

  annotated_path = os.path.join(
      FIGURES_DIR, f"anotada_{architecture}_{os.path.basename(image_path)}"
  )
  _save_annotated_image(original_rgb, predicted_label, confidence, annotated_path)
  print(f"Imagen anotada guardada en: {annotated_path}")

  return predicted_label, confidence


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Predice la clase de un grano de café.")
  parser.add_argument("--architecture", "-a", choices=ARCHITECTURES, default="mobilenet")
  args = parser.parse_args()

  print("Abriendo explorador de archivos...")
  selected_file = select_image_via_file_dialog()

  if selected_file:
    print(f"Imagen seleccionada: {selected_file}")
    predict_single_image(selected_file, architecture=args.architecture)
  else:
    print("No se seleccionó ninguna imagen. Cancelado.")
