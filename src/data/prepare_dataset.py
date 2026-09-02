import os
import shutil
import zipfile
import random
from pathlib import Path

# Rutas del proyecto
DIRECTORIO_BASE = Path(__file__).resolve().parents[2]
CARPETA_PROCESSED = DIRECTORIO_BASE / "data" / "processed"
ARCHIVO_ZIP = DIRECTORIO_BASE / "data" / "raw" / "coffee_union.zip"

# Renombrar clases
MAPEO_CLASES = {
    "Premium": "01_premium",
    "Daño Mecanico": "02_dano_mecanico",
    "Daño Biologico": "03_dano_biologico",
    "Defecto Fermentacion": "04_defecto_fermentacion"
}

ETAPAS = ["train", "val", "test"]
PROPORCION_TRAIN = 0.70
PROPORCION_VAL = 0.15

def preparar_dataset():
    random.seed(42)

    # Extraer el zip en una carpeta temporal
    carpeta_temporal = CARPETA_PROCESSED / "_temp"
    if carpeta_temporal.exists():
        shutil.rmtree(carpeta_temporal)

    with zipfile.ZipFile(ARCHIVO_ZIP, 'r') as zip_ref:
        zip_ref.extractall(carpeta_temporal)

    # Limpiar ejecuciones anteriores
    for elemento in CARPETA_PROCESSED.iterdir():
        if elemento.name != "_temp":
            shutil.rmtree(elemento) if elemento.is_dir() else elemento.unlink()

    # Crear carpetas de destino (train, val, test)
    for etapa in ETAPAS:
        for clase in MAPEO_CLASES.values():
            (CARPETA_PROCESSED / etapa / clase).mkdir(parents=True, exist_ok=True)

    # Diccionario para almacenar los conteos finales
    conteo_final = {etapa: {clase: 0 for clase in MAPEO_CLASES.values()} for etapa in ETAPAS}

    # Procesar carpetas de imagenes
    for ruta_actual, _, archivos in os.walk(carpeta_temporal):
        nombre_carpeta = Path(ruta_actual).name
        
        if nombre_carpeta in MAPEO_CLASES:
            clase = MAPEO_CLASES[nombre_carpeta]
            imagenes = archivos
            
            # Mezcla aleatoria
            random.shuffle(imagenes)

            # Limite de 300 para clase Premium
            if clase == "01_premium":
                imagenes = imagenes[:300]

            total = len(imagenes)
            corte_train = int(total * PROPORCION_TRAIN)
            corte_val = corte_train + int(total * PROPORCION_VAL)

            grupos = {
                "train": imagenes[:corte_train],
                "val": imagenes[corte_train:corte_val],
                "test": imagenes[corte_val:]
            }

            # Contador secuencial global para mantener números únicos en la clase
            contador_secuencial = 1

            # Mover y renombrar (001, 002...)
            for etapa, lista in grupos.items():
                destino = CARPETA_PROCESSED / etapa / clase
                for archivo in lista:
                    extension = Path(archivo).suffix
                    shutil.move(os.path.join(ruta_actual, archivo), destino / f"{contador_secuencial:03d}{extension}")
                    contador_secuencial += 1
                
                conteo_final[etapa][clase] = len(lista)

    shutil.rmtree(carpeta_temporal)
    
    # Imprimir resumen de conteos por carpeta
    print("\n--- RESUMEN DEL DATASET ---")
    for etapa in ETAPAS:
        print(f"\n[{etapa.upper()}]")
        for clase, cantidad in conteo_final[etapa].items():
            print(f"  {clase}: {cantidad} imágenes")

if __name__ == "__main__":
    preparar_dataset()