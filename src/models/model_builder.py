import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB3, VGG16

from data.preprocessing import get_data_augmentation, preprocess_input_by_architecture
from utils.config import IMG_SIZE, NUM_CLASSES


def build_model(architecture="mobilenet", is_training=True):
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # 1. Aumento de datos sólo en entrenamiento
    if is_training:
        augmentation = get_data_augmentation()
        x = augmentation(inputs)
    else:
        x = inputs

    # 2. Normalización específica según la red
    x = preprocess_input_by_architecture(x, architecture)

    # 3. Backbone
    if architecture == "mobilenet":
        base = MobileNetV2(include_top=False, weights="imagenet", input_tensor=x)
    elif architecture == "resnet":
        base = ResNet50(include_top=False, weights="imagenet", input_tensor=x)
    elif architecture == "efficientnet":
        base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=x)
    elif architecture == "vgg":
        base = VGG16(include_top=False, weights="imagenet", input_tensor=x)
    else:
        raise ValueError(f"Arquitectura '{architecture}' no reconocida.")

    base.trainable = False

    # 4. Clasificador
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs)