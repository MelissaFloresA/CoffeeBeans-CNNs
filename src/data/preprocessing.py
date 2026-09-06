"""Preprocesamiento de imágenes de granos de café.

preprocess_image() quita el fondo de cada foto y realza contraste/
nitidez. La llama data/loader.py al cargar cada imagen (train/val/test),
y predict.py/predict_multiple.py para inferencia — mismo código en todos
los casos, para que el modelo vea siempre el mismo tipo de imagen tanto
en entrenamiento como en uso real.

NOTA sobre el aislamiento de fondo: una versión anterior (comparar color
de esquinas, GrabCut) no era confiable — cortaba mal el grano o dejaba
pedazos de fondo pegados. La versión actual (umbral de Otsu + contorno
más grande + convex hull + dilatación leve) da un contorno más completo
y estable, pero sigue asumiendo fondo simple y relativamente uniforme
(como las fotos del dataset), no cualquier fondo real complejo.
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


def otsu_foreground_mask(gray):
  """Umbral de Otsu que elige automáticamente qué lado es el fondo.

  THRESH_BINARY_INV por sí solo asume que el fondo siempre es más claro
  que el grano (cierto con fondo blanco, pero se invierte con fondo
  negro: ahí terminaría marcando el fondo entero como "grano"). Como el
  fondo casi siempre toca los bordes de la foto y el grano casi nunca,
  comparamos qué lado del umbral toca menos los bordes y ese es el
  grano — funciona con fondo claro u oscuro sin necesitar saberlo de
  antemano.
  """
  _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  inverted = cv2.bitwise_not(thresh)

  border = np.zeros_like(gray, dtype=bool)
  border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True

  thresh_touches_border = thresh[border].mean()
  inverted_touches_border = inverted[border].mean()
  return thresh if thresh_touches_border < inverted_touches_border else inverted


def foreground_mask(image_bgr):
  """Máscara de grano candidato: umbral de Otsu sobre brillo
  (otsu_foreground_mask).

  Se probó sumar un filtro de tono verde para descartar fondos tipo
  pasto/plástico verde, pero el dataset es de café CRUDO (green coffee):
  los granos mismos son verdes/oliva, así que ese filtro borraba grano
  real, no solo fondo. Se sacó — separar por color no es viable en este
  dataset, solo por brillo.
  """
  gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
  return otsu_foreground_mask(gray)


def get_bean_mask(image_bgr):
  """Separa el grano del fondo con un umbral de Otsu (foreground_mask) y
  se queda con el contorno más grande como máscara del grano.

  Convex hull + dilatación leve sobre el contorno crudo: sin esto, un
  grano con textura muy irregular (algún parche casi tan claro como el
  fondo) queda con muescas y se ve "quebrado" en vez de una silueta
  completa.
  """
  mask = foreground_mask(image_bgr)

  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if not contours:
    return None

  # Se asume UN grano por foto (así son las fotos de entrenamiento y
  # las de predict.py). Con esa garantía, cualquier contorno relevante
  # es parte del mismo grano: si una mancha o hendidura hace que Otsu
  # lo parta en 2+ pedazos separados, unimos todos antes del hull en
  # vez de quedarnos solo con el más grande — si no, se pierde el
  # pedazo más chico (ej. la punta oscura de un grano "Partial Black").
  # No aplica a find_bean_boxes() en predict_multiple.py, donde sí hay
  # varios granos reales y cada contorno debe seguir siendo distinto.
  significant_contours = [c for c in contours if cv2.contourArea(c) > 50]
  if not significant_contours:
    return None

  all_points = np.vstack(significant_contours)
  hull = cv2.convexHull(all_points)

  mask = np.zeros_like(mask)
  cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
  return cv2.dilate(mask, kernel, iterations=1)


def remove_background(image_bgr):
  """Pone en negro (cero señal) todo lo que no sea el grano, para que el
  modelo aprenda solo del grano y no del fondo de la foto. Si no se
  puede aislar el grano, devuelve la imagen sin tocar en vez de fallar."""
  mask = get_bean_mask(image_bgr)
  if mask is None:
    return image_bgr

  mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
  return cv2.bitwise_and(image_bgr, mask_3ch)


def preprocess_image(image_bgr, img_size=IMG_SIZE):
  """Quita el fondo, redimensiona a 224x224 y realza contraste/nitidez.

  Orden importa: primero se quita ruido, recién después se sube
  contraste y nitidez. Si se afila antes de quitar ruido, el ruido se
  amplifica junto con el detalle real — eso es lo que hacía ver "sucias"
  las fotos de baja calidad. Ningún filtro clásico puede inventar detalle
  que la foto no tiene (eso requeriría un modelo de super-resolución
  aparte); esto sí mejora contraste y nitidez de forma consistente.

  0) Quitar fondo (get_bean_mask + remove_background): el modelo nunca ve
     el fondo real de la foto, ni en entrenamiento ni en inferencia.
  1) Denoise (fastNlMeansDenoisingColored): limpia ruido sin borrar bordes.
  2) CLAHE sobre luminancia: contraste local, resalta manchas y hoyos.
  3) Unsharp mask suave: marca mejor bordes de grietas.
  """
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
