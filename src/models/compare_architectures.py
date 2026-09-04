import os
import sys
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
  sys.path.insert(0, SRC_DIR)

import pandas as pd
from models.evaluate import evaluate
from utils.config import ARCHITECTURES, MODELS_DIR, REPORTS_DIR


def compare_architectures(architectures=None):
  """Evalúa cada arquitectura entrenada sobre el MISMO test set y arma una
  tabla comparativa de accuracy / precision macro / recall macro / F1
  macro. Como train.py usa los mismos hiperparámetros para las 4 (ver
  utils/config.py), esta tabla compara arquitecturas de forma justa: la
  única variable que cambió entre corridas fue el backbone.
  """
  if architectures is None:
    architectures = ARCHITECTURES

  missing = [
      arch
      for arch in architectures
      if not os.path.exists(os.path.join(MODELS_DIR, arch, "weights.h5"))
  ]
  if missing:
    print(
        f"Aviso: las siguientes arquitecturas todavía no tienen pesos"
        f" entrenados y se van a omitir: {missing}"
    )
    print("Entrénalas con: python models/train.py --architecture <nombre>")

  rows = []
  for arch in architectures:
    if arch in missing:
      continue
    print("\n" + "#" * 70)
    print(f"# Evaluando: {arch.upper()}")
    print("#" * 70)
    metrics = evaluate(arch)
    rows.append({
        "arquitectura": arch,
        "accuracy": round(metrics["accuracy"], 4),
        "precision_macro": round(metrics["precision_macro"], 4),
        "recall_macro": round(metrics["recall_macro"], 4),
        "f1_macro": round(metrics["f1_macro"], 4),
    })

  if not rows:
    print("No hay ninguna arquitectura entrenada todavía. Nada que comparar.")
    return None

  comparison_df = pd.DataFrame(rows).set_index("arquitectura")

  lines = []
  lines.append("=" * 70)
  lines.append("COMPARACIÓN DE ARQUITECTURAS (mismos hiperparámetros para todas)")
  lines.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
  lines.append("=" * 70)
  lines.append("")
  lines.append(comparison_df.to_string())
  lines.append("")
  report_text = "\n".join(lines)

  os.makedirs(REPORTS_DIR, exist_ok=True)
  report_path = os.path.join(REPORTS_DIR, "comparacion_arquitecturas.txt")
  with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

  comparison_df.to_csv(os.path.join(REPORTS_DIR, "comparacion_arquitecturas.csv"))

  print("\n" + report_text)
  print(f"\nComparación guardada en: {report_path}")

  return comparison_df


if __name__ == "__main__":
  compare_architectures()
