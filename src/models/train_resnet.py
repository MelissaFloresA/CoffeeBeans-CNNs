import sys
from pathlib import Path

# Agregar la raíz del proyecto (CoffeeBeans-CNNs) al path de Python
DIRECTORIO_RAIZ = Path(__file__).resolve().parents[2]
if str(DIRECTORIO_RAIZ) not in sys.path:
    sys.path.append(str(DIRECTORIO_RAIZ))

# Importaciones normales del proyecto
import tensorflow as tf
from src.utils import config
from src.data.loader import cargar_datasets
from src.data.preprocessing import obtener_capas_augmentation, obtener_escalado_modelo
from src.models.models_builder import construir_modelo
from src.utils.recorder import guardar_historial_csv
NOMBRE_MODELO = "resnet"

def entrenar():
    print(f"\n==================================================")
    print(f"  ENTRENANDO MODELO INDEPENDIENTE: RESNET50")
    print(f"==================================================")

    train_ds, val_ds, _ = cargar_datasets(
        tamano_imagen=config.TAMANO_IMAGEN,
        tamano_lote=config.TAMANO_LOTE
    )

    base = construir_modelo(
        nombre_modelo=NOMBRE_MODELO,
        num_clases=config.NUM_CLASES,
        tamano_entrada=(*config.TAMANO_IMAGEN, config.CANALES_COLOR)
    )

    augmentation = obtener_capas_augmentation()
    escalado = obtener_escalado_modelo(NOMBRE_MODELO)

    inputs = tf.keras.Input(shape=(*config.TAMANO_IMAGEN, config.CANALES_COLOR))
    x = augmentation(inputs)
    x = escalado(x)
    outputs = base(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"Coffee_{NOMBRE_MODELO}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    ruta_guardado = config.CARPETA_MODELOS / f"modelo_{NOMBRE_MODELO}.h5"
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ruta_guardado),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.PATIENCE_EARLY_STOPPING,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=config.PATIENCE_REDUCE_LR,
            verbose=1
        )
    ]

    historial = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks
    )

    guardar_historial_csv(historial, NOMBRE_MODELO)

if __name__ == "__main__":
    entrenar()