"""Funciones para el loop principal"""

import re
from typing import Iterable

from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit_machine_learning.optimizers import ADAM, COBYLA


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))


def qubit_sweep_values(min_q: int, max_q: int) -> Iterable[int]:
    q = int(min_q)
    if q < 2:
        q = 2
    while q <= int(max_q):
        yield q
        q *= 2


def build_feature_map(spec: dict, n_qubits: int, reps: int):
    kind = spec["kind"]
    if kind == "pauli":
        return PauliFeatureMap(feature_dimension=n_qubits, paulis=spec["paulis"], reps=reps)
    if kind == "z_feature_map":
        return ZFeatureMap(feature_dimension=n_qubits, reps=reps)
    if kind == "zz_feature_map":
        return ZZFeatureMap(feature_dimension=n_qubits, reps=reps, entanglement=spec["entanglement"])
    raise ValueError(f"Unsupported encoding spec: {spec}")


def build_optimizer(name: str, maxiter: int):
    normalized = name.lower()
    if normalized == "cobyla":
        return COBYLA(maxiter=maxiter)
    if normalized == "adam":
        return ADAM(maxiter=maxiter)
    raise ValueError(f"Unsupported optimizer: {name}")
