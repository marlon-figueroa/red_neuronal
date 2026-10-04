"""Funciones de pérdida.

Las pérdidas de clasificación reciben los *logits* (salida lineal de la última
capa) y aplican sigmoide/softmax internamente, lo que es numéricamente estable
y simplifica el gradiente a (probabilidad - objetivo).
"""

from __future__ import annotations

import numpy as np

from .activaciones import sigmoide


class Perdida:
    def calcular(self, salida: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError

    def gradiente(self, salida: np.ndarray, y: np.ndarray) -> np.ndarray:
        """dL/dsalida, ya promediado sobre el lote."""
        raise NotImplementedError

    def activar(self, salida: np.ndarray) -> np.ndarray:
        """Convierte la salida cruda de la red en la predicción final."""
        return salida

    def clases(self, salida: np.ndarray) -> np.ndarray:
        raise TypeError(f"{type(self).__name__} no es una pérdida de clasificación.")

    def precision(self, salida: np.ndarray, y: np.ndarray) -> float | None:
        """Fracción de aciertos, o None si la tarea no es de clasificación."""
        return None


class ErrorCuadraticoMedio(Perdida):
    def calcular(self, salida: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((salida - y.reshape(salida.shape)) ** 2))

    def gradiente(self, salida: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2.0 * (salida - y.reshape(salida.shape)) / salida.size


class EntropiaCruzadaBinaria(Perdida):
    """Para una sola neurona de salida con objetivos 0/1."""

    def calcular(self, salida: np.ndarray, y: np.ndarray) -> float:
        y = y.reshape(salida.shape)
        perdidas = np.maximum(salida, 0) - salida * y + np.log1p(np.exp(-np.abs(salida)))
        return float(np.mean(perdidas))

    def gradiente(self, salida: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (sigmoide(salida) - y.reshape(salida.shape)) / salida.size

    def activar(self, salida: np.ndarray) -> np.ndarray:
        return sigmoide(salida)

    def clases(self, salida: np.ndarray) -> np.ndarray:
        return (salida > 0).astype(int).ravel()

    def precision(self, salida: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.clases(salida) == y.ravel()))


class EntropiaCruzada(Perdida):
    """Softmax + entropía cruzada para clasificación multiclase.

    `y` puede ser un vector de etiquetas enteras o una matriz one-hot.
    """

    @staticmethod
    def _etiquetas(y: np.ndarray) -> np.ndarray:
        return y.argmax(axis=1) if y.ndim == 2 else y.astype(int)

    @staticmethod
    def _log_softmax(salida: np.ndarray) -> np.ndarray:
        z = salida - salida.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

    def calcular(self, salida: np.ndarray, y: np.ndarray) -> float:
        etiquetas = self._etiquetas(y)
        log_p = self._log_softmax(salida)
        return float(-np.mean(log_p[np.arange(len(etiquetas)), etiquetas]))

    def gradiente(self, salida: np.ndarray, y: np.ndarray) -> np.ndarray:
        etiquetas = self._etiquetas(y)
        grad = np.exp(self._log_softmax(salida))
        grad[np.arange(len(etiquetas)), etiquetas] -= 1.0
        return grad / len(etiquetas)

    def activar(self, salida: np.ndarray) -> np.ndarray:
        return np.exp(self._log_softmax(salida))

    def clases(self, salida: np.ndarray) -> np.ndarray:
        return salida.argmax(axis=1)

    def precision(self, salida: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.clases(salida) == self._etiquetas(y)))
