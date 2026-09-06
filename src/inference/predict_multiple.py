# Clasifica varios granos de café en una sola foto (fondo simple).
# Reutiliza foreground_mask, preprocess_image y load_trained_model de
# predict.py/preprocessing.py.

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
  sys.path.insert(0, SRC_DIR)

from data.preprocessing import foreground_mask, preprocess_image
from inference.predict import load_trained_model, select_image_via_file_dialog
from utils.config import ARCHITECTURES, FIGURES_DIR

MIN_BEAN_AREA_FRACTION = 0.005  # descarta motas/ruido chico


# Encuentra el rectángulo (x, y, w, h) de cada grano en la foto. Granos
# que se tocan entre sí se fusionan en un solo contorno (límite conocido
# de umbral + contornos, sin watershed).
def find_bean_boxes(image_bgr):
  mask = foreground_mask(image_bgr)
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  image_h, image_w = image_bgr.shape[:2]
  min_area = image_h * image_w * MIN_BEAN_AREA_FRACTION

  return [
      cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area
  ]


# Preprocesa y clasifica un recorte de grano.
def classify_crop(crop_bgr, model, architecture):
  preprocessed_bgr = preprocess_image(crop_bgr)
  preprocessed_rgb = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)

  img_batch = tf.expand_dims(preprocessed_rgb.astype(np.float32), 0)
  predictions = model.predict(img_batch, verbose=0)[0]
  predicted_class_idx = np.argmax(predictions)
  confidence = predictions[predicted_class_idx] * 100
  return predicted_class_idx, confidence


# Detecta y clasifica cada grano de la foto; muestra y guarda la imagen
# anotada con cajas y etiquetas.
def predict_multiple_beans(image_path, architecture="mobilenet"):
  if not image_path or not os.path.exists(image_path):
    print("No se seleccionó ninguna imagen.")
    return None

  image_bgr = cv2.imread(image_path)
  if image_bgr is None:
    raise ValueError(f"No se pudo leer la imagen: {image_path}")

  model, class_names = load_trained_model(architecture)

  boxes = find_bean_boxes(image_bgr)
  if not boxes:
    print("No se encontraron granos en la imagen.")
    return None

  print(f"Granos detectados: {len(boxes)}")

  annotated = image_bgr.copy()
  results = []

  for (x, y, w, h) in boxes:
    crop = image_bgr[y : y + h, x : x + w]
    predicted_class_idx, confidence = classify_crop(crop, model, architecture)
    predicted_label = class_names[predicted_class_idx]
    results.append((predicted_label, confidence))

    label_text = f"{predicted_label} ({confidence:.0f}%)"
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(annotated, (x, y - text_h - 8), (x + text_w + 4, y), (0, 255, 0), -1)
    cv2.putText(
        annotated,
        label_text,
        (x + 2, y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    print(f"  Grano en ({x}, {y}): {predicted_label} ({confidence:.2f}%)")

  os.makedirs(FIGURES_DIR, exist_ok=True)
  out_file = os.path.join(
      FIGURES_DIR, f"resultado_multiple_{architecture}_{os.path.basename(image_path)}"
  )
  cv2.imwrite(out_file, annotated)
  print(f"\nImagen anotada guardada en: {out_file}")

  annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
  plt.figure(figsize=(10, 8))
  plt.imshow(annotated_rgb)
  plt.axis("off")
  plt.title(
      f"Granos detectados: {len(boxes)} ({architecture.upper()})",
      fontsize=14,
      fontweight="bold",
      color="darkblue",
  )
  plt.tight_layout()
  plt.show()

  return results


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Predice la clase de varios granos de café en una foto.")
  parser.add_argument("--architecture", "-a", choices=ARCHITECTURES, default="mobilenet")
  args = parser.parse_args()

  print("Abriendo explorador de archivos...")
  selected_file = select_image_via_file_dialog()

  if selected_file:
    print(f"Imagen seleccionada: {selected_file}")
    predict_multiple_beans(selected_file, architecture=args.architecture)
  else:
    print("No se seleccionó ninguna imagen. Cancelado.")
