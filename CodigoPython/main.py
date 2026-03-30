"""Archivo main del programa completo"""

import csv
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler

import config
from autoencoder_model import build_autoencoder
from data_utils import load_mnist_binary, make_subsets
from experiment_utils import build_feature_map, build_optimizer, qubit_sweep_values, safe_name
from quantum_layers import build_qcnn


def run():
    algorithm_globals.random_seed = config.RANDOM_SEED
    estimator = Estimator()

    (train_images, train_labels), (test_images, test_labels) = load_mnist_binary()
    subsets = make_subsets(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        n_train=config.N_TRAIN,
        n_test=config.N_TEST,
        seed=config.RANDOM_SEED,
    )

    X_train_small = subsets["X_train_small"]
    y_train_small = subsets["y_train_small"]
    X_test_small = subsets["X_test_small"]
    y_test_small = subsets["y_test_small"]

    print("Subsets:", X_train_small.shape, X_test_small.shape)
    print("Qubit sweep:", config.MIN_QUBITS, "->", config.MAX_QUBITS)

    config.CM_OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not config.CSV_PATH.exists()

    with config.CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config.CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()

        for n_qubits_run in qubit_sweep_values(config.MIN_QUBITS, config.MAX_QUBITS):
            print(f"\n===== n_qubits = {n_qubits_run} =====")

            tf.keras.backend.clear_session()
            autoencoder, encoder = build_autoencoder(
                input_dim=X_train_small.shape[1],
                latent_dim=n_qubits_run,
                learning_rate=config.AUTOENCODER_LEARNING_RATE,
            )
            autoencoder.fit(
                np.asarray(X_train_small),
                np.asarray(X_train_small),
                epochs=config.AUTOENCODER_EPOCHS,
                batch_size=config.AUTOENCODER_BATCH_SIZE,
                shuffle=True,
                verbose=0,
            )

            X_train_red = encoder.predict(np.asarray(X_train_small), verbose=0)
            X_test_red = encoder.predict(np.asarray(X_test_small), verbose=0)

            scaler = MinMaxScaler(feature_range=config.FEATURE_RANGE)
            X_train_q = scaler.fit_transform(X_train_red)
            X_test_q = scaler.transform(X_test_red)

            stages = int(np.log2(n_qubits_run))
            ansatz = build_qcnn(start_qubits=n_qubits_run, stages=stages)

            for encoding_spec in config.ENCODING_SPECS:
                encoding_name = encoding_spec["name"]
                for reps in config.ENCODING_REPS:
                    print(f"\n--- Encoding: {encoding_name} | reps={reps} ---")

                    feature_map = build_feature_map(encoding_spec, n_qubits_run, reps)

                    circuit = QuantumCircuit(n_qubits_run)
                    circuit.compose(feature_map, range(n_qubits_run), inplace=True)
                    circuit.compose(ansatz, range(n_qubits_run), inplace=True)

                    observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits_run - 1), 1)])
                    qnn = EstimatorQNN(
                        circuit=circuit.decompose(),
                        observables=observable,
                        input_params=feature_map.parameters,
                        weight_params=ansatz.parameters,
                        estimator=estimator,
                    )

                    for opt_name in config.OPTIMIZER_NAMES:
                        print(f"    -> Optimizer: {opt_name}")

                        classifier = NeuralNetworkClassifier(
                            qnn,
                            optimizer=build_optimizer(opt_name, config.MAXITER),
                        )

                        t0 = time.perf_counter()
                        classifier.fit(np.asarray(X_train_q), np.asarray(y_train_small))
                        train_seconds = time.perf_counter() - t0

                        y_predict = classifier.predict(np.asarray(X_test_q))
                        y_test = np.asarray(y_test_small)

                        acc = accuracy_score(y_test, y_predict)
                        bal_acc = balanced_accuracy_score(y_test, y_predict)
                        kappa = cohen_kappa_score(y_test, y_predict)
                        precision = precision_score(y_test, y_predict, pos_label=1, zero_division=0)
                        recall = recall_score(y_test, y_predict, pos_label=1, zero_division=0)
                        f1 = f1_score(y_test, y_predict, pos_label=1, zero_division=0)

                        cm = confusion_matrix(y_test, y_predict, labels=[-1, 1])
                        tn, fp, fn, tp = cm.ravel()

                        cm_base = (
                            f"cm_{int(n_qubits_run)}q_"
                            f"{safe_name(encoding_name)}_r{int(reps)}_"
                            f"{safe_name(opt_name)}"
                        )
                        np.savetxt(config.CM_OUT_DIR / f"{cm_base}.csv", cm, delimiter=",", fmt="%d")

                        fig, ax = plt.subplots()
                        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-1, 1]).plot(
                            ax=ax, values_format="d"
                        )
                        ax.set_title(
                            f"Confusion Matrix - {int(n_qubits_run)} qubits - "
                            f"{encoding_name} - reps={reps} - {opt_name}"
                        )
                        fig.tight_layout()
                        fig.savefig(config.CM_OUT_DIR / f"{cm_base}.png", dpi=200)
                        plt.close(fig)

                        print(f"Accuracy: {acc:.4f} | Balanced: {bal_acc:.4f} | Kappa: {kappa:.4f}")
                        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
                        print(f"Train seconds: {train_seconds:.2f}")

                        row = {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "n_qubits": int(n_qubits_run),
                            "encoding": str(encoding_name),
                            "encoding_reps": int(reps),
                            "optimizer": str(opt_name),
                            "n_train": int(config.N_TRAIN),
                            "n_test": int(config.N_TEST),
                            "maxiter": int(config.MAXITER),
                            "autoencoder_epochs": int(config.AUTOENCODER_EPOCHS),
                            "autoencoder_batch_size": int(config.AUTOENCODER_BATCH_SIZE),
                            "train_seconds": float(train_seconds),
                            "accuracy": float(acc),
                            "balanced_accuracy": float(bal_acc),
                            "kappa": float(kappa),
                            "precision": float(precision),
                            "recall": float(recall),
                            "f1": float(f1),
                            "tn": int(tn),
                            "fp": int(fp),
                            "fn": int(fn),
                            "tp": int(tp),
                            "explained_variance_sum": None,
                        }
                        writer.writerow(row)

        print(f"\nSweep completado. CSV: {os.path.abspath(config.CSV_PATH)}")


if __name__ == "__main__":
    run()
