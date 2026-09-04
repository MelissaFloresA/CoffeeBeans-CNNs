import os

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Constantes globales del proyecto
IMG_SIZE = (224, 224)
NUM_CLASSES = 5
BATCH_SIZE = 32
SEED = 42

# Arquitecturas soportadas. Único lugar donde se define la lista: si
# quieres agregar/quitar una arquitectura del pipeline (train, evaluate,
# compare_architectures), se cambia aquí y todos los scripts lo heredan.
ARCHITECTURES = ["mobilenet", "resnet", "efficientnet", "vgg"]

# *** HIPERPARÁMETROS COMPARTIDOS ***
# Estas constantes son LA MISMA configuración para las 4 arquitecturas.
# Esto es intencional: si cada arquitectura tuviera su propio learning
# rate/dropout/epochs, la comparación entre modelos dejaría de ser
# homogénea (no sabrías si un modelo ganó por ser mejor arquitectura o por
# tener mejor tuning). La forma correcta de "buscar hiperparámetros" sin
# romper esa homogeneidad es: buscar UNA vez sobre un modelo proxy
# (ver models/hyperparam_search.py, que usa solo mobilenet + val set) y
# actualizar estas constantes con el resultado ganador. A partir de ahí,
# todas las arquitecturas se entrenan con exactamente los mismos valores.
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.4
L2_REG = 1e-4
EPOCHS = 25
LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
