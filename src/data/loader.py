import tensorflow as tf
from pathlib import Path

DIRECTORIO_BASE = Path(__file__).resolve().parents[2]
CARPETA_PROCESSED = DIRECTORIO_BASE / "data" / "processed"

TAMANO_IMAGEN = (224, 224)
TAMANO_LOTE = 32

def cargar_datasets(tamano_imagen=TAMANO_IMAGEN, tamano_lote=TAMANO_LOTE):
    # Cargar entrenamiento
    train_ds = tf.keras.utils.image_dataset_from_directory(
        CARPETA_PROCESSED / "train",
        image_size=tamano_imagen,
        batch_size=tamano_lote,
        label_mode="categorical",
        shuffle=True,
        seed=42
    )

    # Cargar validación
    val_ds = tf.keras.utils.image_dataset_from_directory(
        CARPETA_PROCESSED / "val",
        image_size=tamano_imagen,
        batch_size=tamano_lote,
        label_mode="categorical",
        shuffle=False
    )

    # Cargar prueba (test)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        CARPETA_PROCESSED / "test",
        image_size=tamano_imagen,
        batch_size=tamano_lote,
        label_mode="categorical",
        shuffle=False
    )

    # Optimización de pipeline en memoria (Prefetch)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    print("Probando la carga de datasets con entorno dvc-tf...")
    train_ds, val_ds, test_ds = cargar_datasets()
    
    for imagenes, etiquetas in train_ds.take(1):
        print(f"\nForma del lote de imágenes: {imagenes.shape}")
        print(f"Forma del lote de etiquetas: {etiquetas.shape}")
        print("¡Carga exitosa!")