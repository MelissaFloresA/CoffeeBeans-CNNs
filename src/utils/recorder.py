import pandas as pd
from src.utils import config

def guardar_historial_csv(historial, nombre_modelo):
    """
    Guarda las métricas por época (loss, accuracy, val_loss, val_accuracy, etc.)
    en un solo archivo CSV por modelo dentro de la carpeta /logs.
    """
    config.CARPETA_LOGS.mkdir(parents=True, exist_ok=True)
    
    # Convertir el historial de Keras a un DataFrame
    df = pd.DataFrame(historial.history)
    df.insert(0, 'epoch', range(1, len(df) + 1))
    
    # Guardar en CSV
    ruta_csv = config.CARPETA_LOGS / f"historial_{nombre_modelo}.csv"
    df.to_csv(ruta_csv, index=False)
    
    print(f"\n Historial guardado con éxito en: {ruta_csv}")