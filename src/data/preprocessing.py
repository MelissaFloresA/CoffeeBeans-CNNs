import tensorflow as tf


def get_data_augmentation():
  """Genera las capas de aumento de datos espacial y fotométrico balanceado."""
  return tf.keras.Sequential(
      [
          tf.keras.layers.RandomFlip("horizontal_and_vertical"),
          tf.keras.layers.RandomRotation(factor=0.5),
          tf.keras.layers.RandomTranslation(
              height_factor=0.05, width_factor=0.05
          ),
          tf.keras.layers.RandomContrast(factor=0.10),
      ],
      name="data_augmentation",
  )


def preprocess_input_by_architecture(x, architecture="mobilenet"):
  """Aplica la normalización de tensor específica requerida por cada backbone."""
  x_float = tf.cast(x, tf.float32)

  if architecture == "mobilenet":
    return tf.keras.applications.mobilenet_v2.preprocess_input(x_float)
  elif architecture == "resnet":
    return tf.keras.applications.resnet50.preprocess_input(x_float)
  elif architecture == "efficientnet":
    return tf.keras.applications.efficientnet.preprocess_input(x_float)
  elif architecture == "vgg":
    return tf.keras.applications.vgg16.preprocess_input(x_float)
  else:
    raise ValueError(f"Arquitectura '{architecture}' no soportada.")