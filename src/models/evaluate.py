import sys
from pathlib import Path

# Registrar la raíz del proyecto
DIRECTORIO_RAIZ = Path(__file__).resolve().parents[2]
if str(DIRECTORIO_RAIZ) not in sys.path:
    sys.path.append(str(DIRECTORIO_RAIZ))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.utils import config
from src.data.loader import cargar_datasets

# Crear carpeta de resultados si no existe
CARPETA_RESULTADOS = DIRECTORIO_RAIZ / "results"
CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)


def graficar_curvas_entrenamiento(nombre_modelo):
    """Genera y guarda las gráficas de Loss y Accuracy desde el CSV de logs."""
    ruta_csv = config.CARPETA_LOGS / f"historial_{nombre_modelo}.csv"
    if not ruta_csv.exists():
        print(f"No se encontró el historial para {nombre_modelo} en {ruta_csv}")
        return

    df = pd.read_csv(ruta_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico de Pérdida (Loss)
    axes[0].plot(df['epoch'], df['loss'], label='Train Loss', color='navy', linewidth=2)
    axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss', color='crimson', linestyle='--', linewidth=2)
    axes[0].set_title(f'Pérdida (Loss) - {nombre_modelo.upper()}')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Gráfico de Exactitud (Accuracy)
    axes[1].plot(df['epoch'], df['accuracy'], label='Train Accuracy', color='navy', linewidth=2)
    axes[1].plot(df['epoch'], df['val_accuracy'], label='Val Accuracy', color='crimson', linestyle='--', linewidth=2)
    axes[1].set_title(f'Exactitud (Accuracy) - {nombre_modelo.upper()}')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    ruta_salida = CARPETA_RESULTADOS / f"curvas_entrenamiento_{nombre_modelo}.png"
    plt.savefig(ruta_salida, dpi=300)
    plt.close()
    print(f" -> Curvas de entrenamiento guardadas en: {ruta_salida}")


def evaluar_modelo(nombre_modelo):
    """Evalúa el modelo guardado en el conjunto de TEST y genera la matriz de confusión y métricas."""
    ruta_h5 = config.CARPETA_MODELOS / f"modelo_{nombre_modelo}.h5"
    if not ruta_h5.exists():
        print(f"No se encontró el archivo del modelo en {ruta_h5}")
        return

    print(f"\n==================================================")
    print(f"  EVALUANDO {nombre_modelo.upper()} EN CONJUNTO TEST")
    print(f"==================================================")

    # 1. Cargar modelo y conjunto de prueba
    model = tf.keras.models.load_model(str(ruta_h5))
    _, _, test_ds = cargar_datasets(tamano_imagen=config.TAMANO_IMAGEN, tamano_lote=config.TAMANO_LOTE)

    # 2. Obtener predicciones e etiquetas reales
    y_true = []
    y_pred_probs = []

    for imagenes, etiquetas in test_ds:
        predicciones = model.predict(imagenes, verbose=0)
        y_true.extend(np.argmax(etiquetas.numpy(), axis=1))
        y_pred_probs.extend(predicciones)

    y_true = np.array(y_true)
    y_pred = np.argmax(np.array(y_pred_probs), axis=1)

    # Nombres limpios de las clases para los gráficos
    nombres_clases = [c.replace("01_", "").replace("02_", "").replace("03_", "").replace("04_", "") for c in config.CLASES]

    # 3. Generar y guardar Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=nombres_clases,
                yticklabels=nombres_clases)
    plt.title(f'Matriz de Confusión (Test) - {nombre_modelo.upper()}')
    plt.ylabel('Clase Real')
    plt.xlabel('Clase Predicha')
    plt.tight_layout()
    
    ruta_cm = CARPETA_RESULTADOS / f"matriz_confusion_{nombre_modelo}.png"
    plt.savefig(ruta_cm, dpi=300)
    plt.close()
    print(f" -> Matriz de confusión guardada en: {ruta_cm}")

    # 4. Generar y guardar Reporte de Métricas en CSV
    reporte_dict = classification_report(y_true, y_pred, target_names=nombres_clases, output_dict=True)
    df_reporte = pd.DataFrame(reporte_dict).transpose()
    
    ruta_reporte = CARPETA_RESULTADOS / f"reporte_metricas_{nombre_modelo}.csv"
    df_reporte.to_csv(ruta_reporte)
    print(f" -> Reporte de métricas guardado en: {ruta_reporte}")

    # 5. Generar gráficas de pérdida y precisión de entrenamiento
    graficar_curvas_entrenamiento(nombre_modelo)


if __name__ == "__main__":
    # Evaluar ResNet50 (o cambiar por cualquier modelo ya entrenado)
    evaluar_modelo("resnet")