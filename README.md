# CoffeeBeans-CNNs

Sistema de clasificación multiclase de granos de café arábica mediante Redes Neuronales Convolucionales (CNN) y Transfer Learning.

El proyecto busca clasificar imágenes de granos de café arábica en seis categorías: una condición Premium y cinco defectos físicos, utilizando modelos CNN preentrenados y comparando su desempeño mediante diferentes métricas de clasificación.

## 📌 Descripción

La identificación de defectos en granos de café mediante inspección visual manual puede ser un proceso lento, subjetivo y dependiente de la experiencia del evaluador.

Este proyecto propone desarrollar un sistema de visión artificial capaz de recibir una imagen de un grano de café y clasificarla automáticamente en una de las siguientes categorías:

- Premium
- Broken
- Cut
- Immature
- Insect Damage
- Partial Black

El sistema utiliza técnicas de aprendizaje profundo y Transfer Learning para aprovechar modelos CNN preentrenados en ImageNet.

## 🎯 Objetivo

Desarrollar un sistema de visión artificial basado en Redes Neuronales Convolucionales (CNN), mediante Transfer Learning, capaz de clasificar imágenes de granos de café arábica en seis categorías.

## 🧩 Clases

| Clase | Descripción |
|---|---|
| `Premium` | Granos de café arábica sin defectos visibles |
| `Broken` | Granos fragmentados o partidos |
| `Cut` | Granos con daños físicos superficiales o cortes |
| `Immature` | Granos que no alcanzaron su madurez |
| `Insect Damage` | Granos con daños ocasionados por insectos |
| `Partial Black` | Granos con presencia parcial de coloración negra |

## 📊 Dataset

El conjunto de datos utilizado en el proyecto, denominado `coffee_union_dataset`, fue construido mediante la combinación de dos fuentes públicas de imágenes de granos de café.

### Fuentes

- **USK-Coffee**
  - Sitio oficial: https://coffee.comvislab-usk.org/

- **Coffee Green Bean with 17 Defects**
  - Kaggle: https://www.kaggle.com/datasets/sujitraarw/coffee-green-bean-with-17-defects-original

Se seleccionaron únicamente las clases necesarias para el alcance definido del proyecto.

## 🔄 División de datos

El dataset se divide antes de aplicar Data Augmentation para evitar fuga de información entre los subconjuntos.

| Subconjunto | Porcentaje |
|---|---:|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

El conjunto de prueba se mantiene separado y no participa en el entrenamiento ni en el ajuste de los modelos.

## ⚙️ Pipeline

El sistema sigue el siguiente flujo:

```text
Imagen del grano
       │
       ▼
Carga de datos
       │
       ▼
Train / Validation / Test
       │
       ▼
Balanceo de clases
       │
       ▼
Data Augmentation
       │
       ▼
Preprocesamiento
       │
       ├── Resize 224 × 224
       ├── RGB
       └── Normalización
       │
       ▼
Transfer Learning
       │
       ├── ResNet
       ├── MobileNet
       ├── EfficientNet
       └── VGG
       │
       ▼
Entrenamiento
       │
       ▼
Evaluación
       │
       ├── Accuracy
       ├── Precision Macro
       ├── Recall Macro
       ├── F1-Score Macro
       └── Matriz de Confusión
       │
       ▼
Selección del mejor modelo
       │
       ▼
Módulo de inferencia
       │
       ▼
Clase predicha + confianza
```
## 🏗️ Arquitectura del proyecto
```
CoffeeBeans-CNNs/
│
├── data/
│   ├── raw/
│   │   └── # Datos originales de las fuentes públicas
│   │
│   └── processed/
│       └── # Dataset preparado para entrenamiento
│
├── models/
│   │
│   ├── resnet/
│   │   ├── model.h5
│   │   └── model_config.json
│   │
│   ├── mobilenet/
│   │   ├── model.h5
│   │   └── model_config.json
│   │
│   ├── efficientnet/
│   │   ├── model.h5
│   │   └── model_config.json
│   │
│   ├── vgg/
│   │   ├── model.h5
│   │   └── model_config.json
│   │
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessing.py
│   │
│   ├── features/
│   │   └── feature_extractor.py
│   │
│   ├── models/
│   │   ├── model_builder.py
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   ├── inference/
│   │   └── predict.py
│   │
│   └── utils/
│       ├── config.py
│       └── logger.py
│
├── scripts/
│   └── run_inference.py
│
├── results/
│   ├── metrics/
│   ├── figures/
│   └── confusion_matrices/
│
├── requirements.txt
└── README.md
```
