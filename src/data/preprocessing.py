"""Preprocesamiento de imágenes de granos de café.

preprocess_image() realza contraste/nitidez de cada foto. La llama
data/loader.py al cargar cada imagen (train/val/test), y predict.py para
una imagen suelta — mismo código en ambos casos.

NOTA sobre el aislamiento de fondo: se probaron varias versiones
(comparar color de esquinas, GrabCut con rectángulo, GrabCut guiado por
bordes) y ninguna resultó confiable en fotos reales con fondos variados
— fallaban cortando mal el grano o dejando pedazos de fondo pegados, y
el resultado se veía peor que no tocar el fondo. Se sacó esa parte:
ahora el tratamiento es solo realce, sin intentar recortar nada.
"""

import os
import random

import cv2
import numpy as np
import tensorflow as tf

from utils.config import IMG_SIZE


def set_seed(seed=42):
  """Fija la semilla en Python, NumPy y TensorFlow."""
  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)


def preprocess_image(image_bgr, img_size=IMG_SIZE):
  """Redimensiona a 224x224 y realza contraste/nitidez.

  Orden importa: primero se quita ruido, recién después se sube
  contraste y nitidez. Si se afila antes de quitar ruido, el ruido se
  amplifica junto con el detalle real — eso es lo que hacía ver "sucias"
  las fotos de baja calidad. Ningún filtro clásico puede inventar detalle
  que la foto no tiene (eso requeriría un modelo de super-resolución
  aparte); esto sí mejora contraste y nitidez de forma consistente.

  1) Denoise (fastNlMeansDenoisingColored): limpia ruido sin borrar bordes.
  2) CLAHE sobre luminancia: contraste local, resalta manchas y hoyos.
  3) Unsharp mask suave: marca mejor bordes de grietas.
  """
  img = cv2.resize(image_bgr, img_size, interpolation=cv2.INTER_LINEAR)

  denoised = cv2.fastNlMeansDenoisingColored(
      img, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21
  )

  lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
  l, a, b = cv2.split(lab)
  l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
  enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

  blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
  sharpened = cv2.addWeighted(enhanced, 1.15, blur, -0.15, 0)

  return sharpened


def get_data_augmentation():
  """Aumento liviano: flip/rotación/zoom (el grano no tiene orientación
  fija) + contraste/brillo leve (simula variación de luz). Solo train."""
  return tf.keras.Sequential([
      tf.keras.layers.RandomFlip("horizontal_and_vertical"),
      tf.keras.layers.RandomRotation(factor=0.15),
      tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
      tf.keras.layers.RandomContrast(factor=0.1),
      tf.keras.layers.RandomBrightness(factor=0.1, value_range=(0, 255)),
  ], name="data_augmentation")


def preprocess_input_by_architecture(x, architecture="mobilenet"):
  """Normalización final obligatoria por arquitectura (así se entrenaron
  los pesos de ImageNet, no es elección nuestra)."""
  arch = architecture.lower()
  if arch == "mobilenet":
    return tf.keras.applications.mobilenet_v2.preprocess_input(x)
  elif arch == "resnet":
    return tf.keras.applications.resnet50.preprocess_input(x)
  elif arch == "efficientnet":
    return tf.keras.applications.efficientnet.preprocess_input(x)
  elif arch == "vgg":
    return tf.keras.applications.vgg16.preprocess_input(x)
  raise ValueError(f"Arquitectura '{architecture}' no soportada.")
