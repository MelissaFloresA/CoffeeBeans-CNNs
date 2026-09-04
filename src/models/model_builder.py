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


def _configure_mixed_precision():
  """Activa precisión mixta (fp16) SOLO si hay GPU disponible.

  Por qué: VGG16 agotaba la memoria de GPU (~1.65GB visibles en este
  equipo) incluso corriendo solo, con batch_size=32, porque sus primeras
  capas mantienen resolución 224x224 con muchos canales sin reducir tan
  rápido como MobileNet/ResNet/EfficientNet. Bajar batch_size arreglaría
  el OOM, pero cambiaría la dinámica de entrenamiento (y las estadísticas
  del BatchNormalization de la cabeza) de forma distinta entre
  arquitecturas si solo se le baja a VGG, rompiendo la comparación
  homogénea. La precisión mixta reduce a la mitad la memoria que ocupan
  las activaciones intermedias (el cuello de botella real) sin tocar
  batch_size, learning_rate ni ningún otro hiperparámetro: se aplica por
  igual a las 4 arquitecturas. Los pesos (tf.Variable) se siguen
  almacenando en float32; solo el cómputo intermedio usa float16.
  """
  if tf.config.list_physical_devices("GPU"):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")


_configure_mixed_precision()


def build_model(architecture="mobilenet", is_training=False, dropout_rate=None):
  """Construye el modelo de transfer learning para la arquitectura pedida.

  is_training: solo se conserva por compatibilidad con el código anterior;
  Keras ya maneja automáticamente el modo train/inference de Dropout/BN de
  la cabeza según se llame model.fit()/model.predict()/model.evaluate().

  dropout_rate: permite sobreescribir DROPOUT_RATE de config.py para
  experimentos de búsqueda de hiperparámetros (ver hyperparam_search.py).
  En entrenamiento normal de las 4 arquitecturas se deja en None para que
  las 4 usen el mismo valor por defecto (DROPOUT_RATE), garantizando una
  comparación homogénea.
  """
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

  # Backbone congelado (base.trainable=False) => SIEMPRE training=False,
  # incluso durante el entrenamiento del resto del modelo, para que las
  # capas BatchNormalization del backbone usen las estadísticas de
  # ImageNet ya aprendidas y no las del mini-batch actual (ver historial
  # del proyecto: este era el bug que causaba val_accuracy inestable).
  x = base(x, training=False)

  x = GlobalAveragePooling2D()(x)
  x = BatchNormalization()(x)
  x = Dropout(dropout_rate)(x)
  # dtype="float32" explícito: con precisión mixta activa, la capa de
  # salida y el cálculo de la pérdida deben quedar en float32 por
  # estabilidad numérica (softmax + label_smoothing en float16 puede
  # producir NaNs). Esto es lo que recomienda la guía oficial de Keras
  # para mixed precision.
  outputs = tf.keras.layers.Dense(
      NUM_CLASSES,
      activation="softmax",
      kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
      dtype="float32",
  )(x)

  return Model(inputs=inputs, outputs=outputs, name=f"coffee_{arch_lower}")
