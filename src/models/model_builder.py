import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from data.preprocessing import preprocess_input_by_architecture
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import EfficientNetB3, MobileNetV2, ResNet50, VGG16
from tensorflow.keras.layers import BatchNormalization, Dropout, GlobalAveragePooling2D
from utils.config import DROPOUT_RATE, IMG_SIZE, L2_REG, NUM_CLASSES

ARCHITECTURE_BUILDERS = {
    "mobilenet": MobileNetV2,
    "resnet": ResNet50,
    "efficientnet": EfficientNetB3,
    "vgg": VGG16,
}


# Activa precisión mixta (fp16) si hay GPU, para no quedarse sin VRAM
# con VGG16. Se ejecuta una sola vez al importar el módulo.
def _configure_mixed_precision():
  if tf.config.list_physical_devices("GPU"):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")


_configure_mixed_precision()


# Construye el modelo de transfer learning (backbone de ImageNet
# congelado + cabecita nueva). Usada en train.py, evaluate.py, predict.py
# e hyperparam_search.py.
def build_model(architecture="mobilenet", is_training=False, dropout_rate=None):
  if dropout_rate is None:
    dropout_rate = DROPOUT_RATE

  arch_lower = architecture.lower()
  if arch_lower not in ARCHITECTURE_BUILDERS:
    raise ValueError(
        f"Arquitectura '{architecture}' no soportada. Usa una de:"
        f" {list(ARCHITECTURE_BUILDERS.keys())}"
    )

  inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

  base_cls = ARCHITECTURE_BUILDERS[arch_lower]
  base = base_cls(
      include_top=False,
      weights="imagenet",
      input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
  )
  base.trainable = False

  x = preprocess_input_by_architecture(inputs, arch_lower)

  # training=False fijo: el backbone congelado necesita las estadísticas
  # de BatchNorm de ImageNet, no las del mini-batch actual.
  x = base(x, training=False)

  x = GlobalAveragePooling2D()(x)
  x = BatchNormalization()(x)
  x = Dropout(dropout_rate)(x)
  # float32 explícito: softmax + label_smoothing en float16 puede dar NaN.
  outputs = tf.keras.layers.Dense(
      NUM_CLASSES,
      activation="softmax",
      kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
      dtype="float32",
  )(x)

  return Model(inputs=inputs, outputs=outputs, name=f"coffee_{arch_lower}")
