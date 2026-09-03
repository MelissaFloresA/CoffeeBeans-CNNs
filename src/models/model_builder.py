import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from data.preprocessing import preprocess_input_by_architecture
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import EfficientNetB3, MobileNetV2, ResNet50, VGG16
from tensorflow.keras.layers import BatchNormalization, Dropout, GlobalAveragePooling2D
from utils.config import IMG_SIZE, NUM_CLASSES


def build_model(architecture="mobilenet", is_training=False):
  inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

  arch_lower = architecture.lower()
  if arch_lower == "mobilenet":
    base = MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
  elif arch_lower == "resnet":
    base = ResNet50(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
  elif arch_lower == "efficientnet":
    base = EfficientNetB3(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
  elif arch_lower == "vgg":
    base = VGG16(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
  else:
    raise ValueError(f"Arquitectura '{architecture}' no soportada.")

  base.trainable = False

  x = preprocess_input_by_architecture(inputs, architecture)
  x = base(x, training=is_training)
  x = GlobalAveragePooling2D()(x)
  x = BatchNormalization()(x)
  x = Dropout(0.4)(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

  return Model(inputs=inputs, outputs=outputs)