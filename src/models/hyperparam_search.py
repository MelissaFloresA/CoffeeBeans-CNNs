"""Búsqueda de hiperparámetros compartidos.

IMPORTANTE - por qué este script existe:
Comparar 4 arquitecturas de forma justa requiere que todas usen los MISMOS
hiperparámetros (ver utils/config.py). Pero entonces, ¿cómo se eligen esos
hiperparámetros sin favorecer a una arquitectura en particular?

La respuesta estándar: se hace una búsqueda pequeña sobre UN SOLO modelo
"proxy" (aquí, MobileNet, por ser el más liviano/rápido de entrenar),
evaluando siempre sobre el set de VALIDACIÓN (nunca sobre test, para no
contaminar la evaluación final). El resultado ganador se copia manualmente
a utils/config.py como LEARNING_RATE / DROPOUT_RATE, y a partir de ahí se
entrena a las 4 arquitecturas con exactamente esos mismos valores
(models/train.py --all). Esto mantiene la comparación final homogénea:
la búsqueda influye en la elección de la config compartida, pero no le da
ventaja a ninguna arquitectura sobre otra.

Uso:
    python models/hyperparam_search.py
"""

import os
import sys
import gc
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
  sys.path.insert(0, SRC_DIR)

from data.loader import load_data
from data.preprocessing import set_seed
from models.model_builder import build_model
import pandas as pd
import tensorflow as tf
from utils.config import LABEL_SMOOTHING, REPORTS_DIR, SEED

# Grilla pequeña a propósito: cada combinación entrena un modelo desde
# cero. Amplía esta grilla si tienes tiempo/GPU de sobra.
LEARNING_RATES = [1e-3, 5e-4, 1e-4]
DROPOUT_RATES = [0.3, 0.4, 0.5]
SEARCH_EPOCHS = 12  # menos épocas que el entrenamiento final: solo se
                    # necesita comparar tendencias entre combinaciones.
PROXY_ARCHITECTURE = "mobilenet"


def run_search():
  set_seed(SEED)

  train_ds = load_data("train")  # augment=True por defecto
  val_ds = load_data("val")  # augment=False por defecto

  results = []

  for lr in LEARNING_RATES:
    for dropout_rate in DROPOUT_RATES:
      print("\n" + "-" * 60)
      print(f"Probando lr={lr}, dropout_rate={dropout_rate}")
      print("-" * 60)

      set_seed(SEED)  # misma inicialización para cada combinación
      tf.keras.backend.clear_session()  # liberar memoria del modelo anterior
      gc.collect()
      model = build_model(
          PROXY_ARCHITECTURE, is_training=True, dropout_rate=dropout_rate
      )
      model.compile(
          optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
          loss=tf.keras.losses.CategoricalCrossentropy(
              label_smoothing=LABEL_SMOOTHING
          ),
          metrics=["accuracy"],
      )

      early_stop = tf.keras.callbacks.EarlyStopping(
          monitor="val_loss", patience=4, restore_best_weights=True
      )

      history = model.fit(
          train_ds,
          validation_data=val_ds,
          epochs=SEARCH_EPOCHS,
          callbacks=[early_stop],
          verbose=0,
      )

      best_val_loss = min(history.history["val_loss"])
      best_val_accuracy = max(history.history["val_accuracy"])

      results.append({
          "learning_rate": lr,
          "dropout_rate": dropout_rate,
          "best_val_loss": round(best_val_loss, 4),
          "best_val_accuracy": round(best_val_accuracy, 4),
          "epochs_corridas": len(history.history["val_loss"]),
      })
      print(
          f"  -> best_val_loss={best_val_loss:.4f},"
          f" best_val_accuracy={best_val_accuracy:.4f}"
      )

  results_df = pd.DataFrame(results).sort_values("best_val_loss")

  lines = []
  lines.append("=" * 70)
  lines.append(f"BÚSQUEDA DE HIPERPARÁMETROS (proxy: {PROXY_ARCHITECTURE.upper()})")
  lines.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
  lines.append("Evaluado siempre sobre el set de VALIDACIÓN (no test).")
  lines.append("=" * 70)
  lines.append("")
  lines.append(results_df.to_string(index=False))
  lines.append("")

  best = results_df.iloc[0]
  lines.append(
      f"Mejor combinación (menor val_loss): learning_rate={best['learning_rate']},"
      f" dropout_rate={best['dropout_rate']}"
  )
  lines.append("")
  lines.append(
      "Siguiente paso: copia estos valores a utils/config.py "
      "(LEARNING_RATE, DROPOUT_RATE) y entrena las 4 arquitecturas con "
      "'python models/train.py --all' para que TODAS usen esta misma "
      "configuración."
  )
  report_text = "\n".join(lines)

  os.makedirs(REPORTS_DIR, exist_ok=True)
  report_path = os.path.join(REPORTS_DIR, "busqueda_hiperparametros.txt")
  with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
  results_df.to_csv(
      os.path.join(REPORTS_DIR, "busqueda_hiperparametros.csv"), index=False
  )

  print("\n" + report_text)
  print(f"\nResultados guardados en: {report_path}")

  return results_df


if __name__ == "__main__":
  run_search()
