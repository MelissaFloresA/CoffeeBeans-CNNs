# Preprocesamiento de imágenes: quitar fondo + realce. Usado por
# loader.py (train/val/test) y predict.py/predict_multiple.py (inferencia).

import os
import random

import cv2
import numpy as np
import tensorflow as tf

from utils.config import IMG_SIZE


# Fija la semilla en Python, NumPy y TensorFlow.
def set_seed(seed=42):
  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)


# Umbral de Otsu; detecta solo si el fondo es claro u oscuro.
def otsu_foreground_mask(gray):
  _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  inverted = cv2.bitwise_not(thresh)

  border = np.zeros_like(gray, dtype=bool)
  border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True

  thresh_touches_border = thresh[border].mean()
  inverted_touches_border = inverted[border].mean()
  return thresh if thresh_touches_border < inverted_touches_border else inverted


# Máscara de grano candidato a partir del brillo.
def foreground_mask(image_bgr):
  gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
  return otsu_foreground_mask(gray)


# Máscara final del grano: une todos los contornos relevantes (asume un
# grano por foto) y aplica convex hull + dilatación leve.
def get_bean_mask(image_bgr):
  mask = foreground_mask(image_bgr)

  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if not contours:
    return None

  significant_contours = [c for c in contours if cv2.contourArea(c) > 50]
  if not significant_contours:
    return None

  all_points = np.vstack(significant_contours)
  hull = cv2.convexHull(all_points)

  mask = np.zeros_like(mask)
  cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
  return cv2.dilate(mask, kernel, iterations=1)


# Pone en negro todo lo que no sea el grano.
def remove_background(image_bgr):
  mask = get_bean_mask(image_bgr)
  if mask is None:
    return image_bgr

  mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
  return cv2.bitwise_and(image_bgr, mask_3ch)


# Quita fondo, redimensiona y realza contraste/nitidez.
def preprocess_image(image_bgr, img_size=IMG_SIZE):
  no_bg = remove_background(image_bgr)
  img = cv2.resize(no_bg, img_size, interpolation=cv2.INTER_LINEAR)

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


# Aumento de datos (flip/rotación/zoom/contraste/brillo), solo train.
def get_data_augmentation():
  return tf.keras.Sequential([
      tf.keras.layers.RandomFlip("horizontal_and_vertical"),
      tf.keras.layers.RandomRotation(factor=0.15),
      tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
      tf.keras.layers.RandomContrast(factor=0.1),
      tf.keras.layers.RandomBrightness(factor=0.1, value_range=(0, 255)),
  ], name="data_augmentation")


# Normalización final requerida por cada arquitectura de ImageNet.
def preprocess_input_by_architecture(x, architecture="mobilenet"):
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
