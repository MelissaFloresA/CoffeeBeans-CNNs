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

# Arquitecturas soportadas. Único lugar donde se define la lista.
ARCHITECTURES = ["mobilenet", "resnet", "efficientnet", "vgg"]

# Hiperparámetros compartidos por las 4 arquitecturas (comparación
# homogénea). Buscados sobre un modelo proxy en hyperparam_search.py.
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.4
L2_REG = 1e-4
EPOCHS = 25
LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 5
