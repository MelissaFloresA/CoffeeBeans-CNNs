import json
import os
import sys
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
  sys.path.insert(0, SRC_DIR)

from models.model_builder import build_model
from utils.config import MODELS_DIR, RESULTS_DIR


def select_image_via_file_dialog():
  root = tk.Tk()
  root.withdraw()
  root.attributes("-topmost", True)
  file_path = filedialog.askopenfilename(
      title="Selecciona una imagen de grano de café",
      filetypes=[("Archivos de Imagen", "*.jpg *.jpeg *.png *.bmp *.webp")],
  )
  root.destroy()
  return file_path


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

  raw_img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
  img_array = tf.keras.utils.img_to_array(raw_img)
  img_batch = tf.expand_dims(img_array, 0)

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

  fig, axes = plt.subplots(1, 2, figsize=(12, 5))

  axes[0].imshow(img_array.astype("uint8"))
  axes[0].set_title("Imagen Seleccionada", fontweight="bold")
  axes[0].axis("off")

  y_pos = np.arange(len(class_names))
  plot_classes = class_names[::-1]
  plot_preds = (predictions * 100)[::-1]

  # Todas las barras en gris como lo solicitaste
  plot_colors = ["gray"] * len(class_names)

  axes[1].barh(y_pos, plot_preds, color=plot_colors)
  axes[1].set_yticks(y_pos)
  axes[1].set_yticklabels(plot_classes)
  axes[1].set_xlabel("Confianza (%)")
  axes[1].set_title("Desglose por Clase", fontweight="bold")
  axes[1].set_xlim(0, 100)

  for i, v in enumerate(plot_preds):
    axes[1].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)

  # Título superior fijo en color azul
  plt.suptitle(
      f"Predicción Final: {predicted_label} ({confidence:.2f}%)",
      fontsize=14,
      fontweight="bold",
      color="darkblue",
  )
  plt.tight_layout()

  fig_dir = os.path.join(RESULTS_DIR, "figures")
  os.makedirs(fig_dir, exist_ok=True)
  out_file = os.path.join(fig_dir, f"resultado_inferencia_{architecture}.png")
  plt.savefig(out_file, dpi=300)
  plt.show()

  return predicted_label, confidence


if __name__ == "__main__":
  print("Abriendo explorador de archivos...")
  selected_file = select_image_via_file_dialog()

  if selected_file:
    print(f"Imagen seleccionada: {selected_file}")
    predict_single_image(selected_file, architecture="mobilenet")