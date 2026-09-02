from pathlib import Path

# 1. Rutas absolutas
DIRECTORIO_BASE = Path(__file__).resolve().parents[2]
DATOS_PROCESADOS = DIRECTORIO_BASE / "data" / "processed"
CARPETA_MODELOS = DIRECTORIO_BASE / "models"
CARPETA_LOGS = DIRECTORIO_BASE / "logs"

CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
CARPETA_LOGS.mkdir(parents=True, exist_ok=True)

# 2. Configuración de entrada e imágenes
TAMANO_IMAGEN = (224, 224)
TAMANO_LOTE = 32
CANALES_COLOR = 3
NUM_CLASES = 4
SEED = 42

CLASES = [
    "01_premium",
    "02_dano_mecanico",
    "03_dano_biologico",
    "04_defecto_fermentacion"
]

# 4. Hiperparámetros Estándar
EPOCHS = 30
LEARNING_RATE = 1e-3
PATIENCE_EARLY_STOPPING = 7
PATIENCE_REDUCE_LR = 3