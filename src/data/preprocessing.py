import tensorflow as tf

def get_data_augmentation():
    """Genera las capas de aumento de datos espacial y fotométrico para granos de café."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(factor=1.0),  # Rotación 360°
            tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
            tf.keras.layers.RandomContrast(factor=0.10),
        ],
        name="data_augmentation",
    )

def preprocess_input_by_architecture(x, architecture="mobilenet"):
    """Aplica la normalización de tensor específica requerida por cada backbone."""
    if architecture == "mobilenet":
        return tf.keras.applications.mobilenet_v2.preprocess_input(x)
    elif architecture == "resnet":
        return tf.keras.applications.resnet50.preprocess_input(x)
    elif architecture == "efficientnet":
        return tf.keras.applications.efficientnet.preprocess_input(x)
    elif architecture == "vgg":
        return tf.keras.applications.vgg16.preprocess_input(x)
    else:
        raise ValueError(f"Arquitectura '{architecture}' no soportada para preprocesamiento.")