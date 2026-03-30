"""
Parametros de configuracion para el pipline
"""

from pathlib import Path
import numpy as np

# Reproducibility
RANDOM_SEED = 12345

# Dataset and subset sizes
MIN_QUBITS = 2
MAX_QUBITS = 8
N_TRAIN = 700
N_TEST = 300

# Autoencoder
AUTOENCODER_LEARNING_RATE = 1e-3
AUTOENCODER_EPOCHS = 20
AUTOENCODER_BATCH_SIZE = 32

# Quantum classifier optimizer
MAXITER = 40
OPTIMIZER_NAMES = ["cobyla", "adam"]

# Encoding sweep
ENCODING_REPS = [1, 2, 3]
ENCODING_SPECS = [
    {"name": "pauli_x", "kind": "pauli", "paulis": ["X"]},
    {"name": "pauli_y", "kind": "pauli", "paulis": ["Y"]},
    {"name": "z_feature_map", "kind": "z_feature_map"},
    {"name": "zz_feature_map_linear", "kind": "zz_feature_map", "entanglement": "linear"},
    {"name": "zz_feature_map_full", "kind": "zz_feature_map", "entanglement": "full"},
]

# Scaling for the quantum feature map input
FEATURE_RANGE = (-np.pi, np.pi)

# Rutas de output para las metricas
CSV_PATH = Path("metrics_qcnn_mnist_sweep.csv")
CM_OUT_DIR = Path("matrices_confusion")

# Columnas del CSV de resultados
CSV_FIELDNAMES = [
    "timestamp",
    "n_qubits",
    "encoding",
    "encoding_reps",
    "optimizer",
    "n_train",
    "n_test",
    "maxiter",
    "autoencoder_epochs",
    "autoencoder_batch_size",
    "train_seconds",
    "accuracy",
    "balanced_accuracy",
    "kappa",
    "precision",
    "recall",
    "f1",
    "tn",
    "fp",
    "fn",
    "tp",
    "explained_variance_sum",
]
