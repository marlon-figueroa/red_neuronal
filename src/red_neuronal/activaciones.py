"""Funciones de activación, implementadas como capas sin parámetros."""

from __future__ import annotations

import numpy as np

from .capas import Capa


def sigmoide(x: np.ndarray) -> np.ndarray:
    # Equivalente a 1 / (1 + e^-x) pero sin desbordamiento para |x| grande.
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class ReLU(Capa):
    def adelante(self, x: np.ndarray, entrenando: bool = False) -> np.ndarray:
        self._mascara = x > 0
        return x * self._mascara

    def atras(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._mascara


class LeakyReLU(Capa):
    def __init__(self, alfa: float = 0.01) -> None:
        self.alfa = alfa

    def adelante(self, x: np.ndarray, entrenando: bool = False) -> np.ndarray:
        self._x = x
        return np.where(x > 0, x, self.alfa * x)

    def atras(self, grad: np.ndarray) -> np.ndarray:
        return grad * np.where(self._x > 0, 1.0, self.alfa)

    def config(self) -> dict:
        return {"alfa": self.alfa}


class Sigmoide(Capa):
    def adelante(self, x: np.ndarray, entrenando: bool = False) -> np.ndarray:
        self._salida = sigmoide(x)
        return self._salida

    def atras(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._salida * (1.0 - self._salida)


class Tanh(Capa):
    def adelante(self, x: np.ndarray, entrenando: bool = False) -> np.ndarray:
        self._salida = np.tanh(x)
        return self._salida

    def atras(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1.0 - self._salida**2)


ACTIVACIONES = {
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "sigmoide": Sigmoide,
    "tanh": Tanh,
}
