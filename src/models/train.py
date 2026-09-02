import os
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import tensorflow as tf
from data.loader import load_data
from models.model_builder import build_model
from utils.config import MODELS_DIR

class CleanHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.history.keys():
            val = logs.get(k)
            if val is not None:
                self.history[k].append(float(val))

def train_architecture(architecture="mobilenet"):
    train_ds = load_data("train")
    val_ds = load_data("val")

    # Balanceo por frecuencias: 0: Black, 1: Broken, 2: Immature, 3: Insect Damage, 4: Premium
    class_weights = {0: 1.1, 1: 1.0, 2: 1.0, 3: 1.1, 4: 0.80}
    print(f"\nEntrenando arquitectura: {architecture.upper()}")
    print(f"Pesos de clase asignados: {class_weights}\n")

    model = build_model(architecture, is_training=True)

    # FASE 1: Calentamiento de la Cabeza Clasificadora
    print(f"=== FASE 1: Entrenamiento Base ({architecture.upper()}) ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.10),
        metrics=["accuracy"]
    )

    cb_history_p1 = CleanHistoryCallback()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weights,
        callbacks=[cb_history_p1],
        verbose=1
    )

    # FASE 2: Fine-Tuning
    print(f"\n=== FASE 2: Fine-Tuning ({architecture.upper()}) ===")
    base_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_layer = layer
            break

    if base_layer:
        base_layer.trainable = True
        # Descongelar últimas 30 capas
        for layer in base_layer.layers[:-30]:
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.10),
        metrics=["accuracy"]
    )

    save_dir = os.path.join(MODELS_DIR, architecture)
    os.makedirs(save_dir, exist_ok=True)

    weights_path = os.path.join(save_dir, "weights.h5")
    config_path = os.path.join(save_dir, "model_config.json")

    cb_history_p2 = CleanHistoryCallback()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            weights_path, save_best_only=True, monitor="val_accuracy", save_weights_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=8, restore_best_weights=True, monitor="val_accuracy"
        ),
        cb_history_p2
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    model.save_weights(weights_path)

    combined_history = {
        "accuracy": cb_history_p1.history["accuracy"] + cb_history_p2.history["accuracy"],
        "val_accuracy": cb_history_p1.history["val_accuracy"] + cb_history_p2.history["val_accuracy"],
        "loss": cb_history_p1.history["loss"] + cb_history_p2.history["loss"],
        "val_loss": cb_history_p1.history["val_loss"] + cb_history_p2.history["val_loss"]
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"architecture": architecture, "history": combined_history}, f, indent=4)

    print(f"\n¡Entrenamiento de {architecture.upper()} completado! Pesos en: {weights_path}")

if __name__ == "__main__":
    train_architecture("mobilenet")