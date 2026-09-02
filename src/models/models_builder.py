import ssl
import tensorflow as tf

ssl._create_default_https_context = ssl._create_unverified_context

ARQUITECTURAS = {
    "resnet": tf.keras.applications.ResNet50,
    "mobilenet": tf.keras.applications.MobileNetV2,
    "efficientnet": tf.keras.applications.EfficientNetB0,
    "vgg": tf.keras.applications.VGG16
}

def construir_modelo(nombre_modelo="resnet", num_clases=4, tamano_entrada=(224, 224, 3)):
    nombre = nombre_modelo.lower()
    
    if nombre not in ARQUITECTURAS:
        raise ValueError(f"Modelo '{nombre}' no soportado: {list(ARQUITECTURAS.keys())}")

    ModeloBase = ARQUITECTURAS[nombre]
    base_model = ModeloBase(
        weights="imagenet",
        include_top=False,
        input_shape=tamano_entrada
    )

    # Para ResNet50, descongelamos los últimos 20 bloques convolucionales para Fine-Tuning
    if nombre == "resnet":
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
    else:
        base_model.trainable = False

    # Cabeza de clasificación mejorada
    inputs = tf.keras.Input(shape=tamano_entrada)
    x = base_model(inputs, training=True if nombre == "resnet" else False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(num_clases, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"Base_{nombre}")
    return model

if __name__ == "__main__":
    m = construir_modelo("resnet")
    print("ResNet50 ajustado con capas entrenables para Fine-Tuning.")