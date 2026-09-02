import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

def aplicar_filtro_cafe_avanzado(imagen_rgb):
    """
    PASO 4 DEL DIAGRAMA: Preprocesamiento de imágenes.
    Resalta hoyos de broca, grietas y manchas de fermentación mediante
    filtrado Bilateral + CLAHE adaptativo en canal L (LAB) + Unsharp Masking.
    """
    imagen_uint8 = imagen_rgb.astype(np.uint8)

    # 1. Reducción de ruido que conserva bordes e imperfecciones
    denoised = cv2.bilateralFilter(imagen_uint8, d=7, sigmaColor=35, sigmaSpace=35)

    # 2. Conversión a LAB y CLAHE en Luminancia (resalta profundidad de hoyos)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    lab_clahe = cv2.merge((l_clahe, a, b))
    imagen_lab = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    # 3. Unsharp Masking focalizado para afilar bordes
    suave = cv2.GaussianBlur(imagen_lab, (0, 0), 3)
    imagen_final = cv2.addWeighted(imagen_lab, 1.4, suave, -0.4, 0)

    return imagen_final


def obtener_capas_augmentation():
    """
    PASO 3 DEL DIAGRAMA: Aumento de datos (Solo en entrenamiento).
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(factor=1.0),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1)),
    ], name="aumento_de_datos")


def obtener_escalado_modelo(modelo_nombre="resnet"):
    """
    PASO 4 DEL DIAGRAMA: Escalado de píxeles según la CNN.
    """
    nombre = modelo_nombre.lower()

    if "efficientnet" in nombre:
        return tf.keras.layers.Layer(name="normalizacion_passthrough")
    elif "mobilenet" in nombre:
        return tf.keras.layers.Rescaling(scale=1./127.5, offset=-1, name="normalizacion_mobilenet")
    else:
        return tf.keras.layers.Rescaling(scale=1./255, name="normalizacion_estandar")


if __name__ == "__main__":
    #  PRUEBA VISUAL
    CLASE_A_PROBAR = "03_dano_biologico"  # Prueba con daño por broca
    INDICE_IMAGEN = 17                       # 1, 2, 3... para ver otras fotos
    # =========================================================================

    directorio_base = Path(__file__).resolve().parents[2]
    carpeta_clase = directorio_base / "data" / "processed" / "train" / CLASE_A_PROBAR

    # Buscar imágenes en la carpeta seleccionada
    imagenes_clase = sorted(list(carpeta_clase.glob("*.jpg")) + list(carpeta_clase.glob("*.png")))

    if imagenes_clase:
        # Validar índice dentro del rango de imágenes disponibles
        idx = min(INDICE_IMAGEN, len(imagenes_clase) - 1)
        ruta_imagen = imagenes_clase[idx]
        
        print(f"\nProbando preprocesamiento avanzado:")
        print(f"-> Clase: {CLASE_A_PROBAR}")
        print(f"-> Imagen ({idx + 1}/{len(imagenes_clase)}): {ruta_imagen.name}")

        # Leer y redimensionar
        imagen_bgr = cv2.imread(str(ruta_imagen))
        imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
        imagen_rgb = cv2.resize(imagen_rgb, (224, 224))

        # Aplicar el filtro especializado
        resultado = aplicar_filtro_cafe_avanzado(imagen_rgb)

        # Visualización comparativa
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(imagen_rgb)
        axes[0].set_title(f"Original ({CLASE_A_PROBAR})")
        axes[0].axis("off")

        axes[1].imshow(resultado)
        axes[1].set_title("Filtro Hoyos / Manchas / Color")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
    else:
        print(f"No se encontraron imágenes en la ruta: {carpeta_clase}")