# Búsqueda de hiperparámetros compartidos, sobre un solo modelo proxy
# (mobilenet) y evaluado siempre en validación. El resultado se copia a
# mano a utils/config.py para que las 4 arquitecturas usen la misma
# config en train.py --all.

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

LEARNING_RATES = [1e-3, 5e-4, 1e-4]
DROPOUT_RATES = [0.3, 0.4, 0.5]
SEARCH_EPOCHS = 12
PROXY_ARCHITECTURE = "mobilenet"


# Entrena el modelo proxy con cada combinación de lr/dropout y guarda un
# reporte con la mejor según val_loss.
def run_search():
  set_seed(SEED)

  train_ds = load_data("train")
  val_ds = load_data("val")

  results = []

  for lr in LEARNING_RATES:
    for dropout_rate in DROPOUT_RATES:
      print("\n" + "-" * 60)
      print(f"Probando lr={lr}, dropout_rate={dropout_rate}")
      print("-" * 60)

      set_seed(SEED)
      tf.keras.backend.clear_session()
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
