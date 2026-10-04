"""Capas de la red: cada una sabe propagar hacia adelante y hacia atrás."""

from __future__ import annotations

import numpy as np

Parametro = tuple[np.ndarray, np.ndarray]


class Capa:
    """Interfaz común para todas las capas."""

    def adelante(self, x: np.ndarray, entrenando: bool = False) -> np.ndarray:
        raise NotImplementedError

    def atras(self, grad: np.ndarray) -> np.ndarray:
        """Recibe dL/dsalida y devuelve dL/dentrada, guardando los gradientes propios."""
        raise NotImplementedError

    def parametros(self) -> list[Parametro]:
        """Pares (parámetro, gradiente) que el optimizador debe actualizar."""
        return []

    def config(self) -> dict:
        """Argumentos necesarios para reconstruir la capa al cargar un modelo."""
        return {}

    def __repr__(self) -> str:
        argumentos = ", ".join(f"{k}={v!r}" for k, v in self.config().items())
        return f"{type(self).__name__}({argumentos})"


class Densa(Capa):
    """Capa totalmente conectada: y = x @ W + b."""

    def __init__(
        self,
        entradas: int,
        salidas: int,
        inicializacion: str = "he",
        semilla: int | None = None,
    ) -> None:
        if inicializacion == "he":
            escala = np.sqrt(2.0 / entradas)
        elif inicializacion == "xavier":
            escala = np.sqrt(2.0 / (entradas + salidas))
        else:
            raise ValueError(f"Inicialización desconocida: {inicializacion!r}")

        rng = np.random.default_rng(semilla)
        self.entradas = entradas
        self.salidas = salidas
        self.inicializacion = inicializacion
        self.W = rng.normal(0.0, escala, size=(entradas, salidas))
        self.b = np.zeros(salidas)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def adelante(self, x: np.ndarray, entrenando: bool = False) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def atras(self, grad: np.ndarray) -> np.ndarray:
        self.dW[...] = self._x.T @ grad
        self.db[...] = grad.sum(axis=0)
        return grad @ self.W.T

    def parametros(self) -> list[Parametro]:
        return [(self.W, self.dW), (self.b, self.db)]

    def config(self) -> dict:
        return {
            "entradas": self.entradas,
            "salidas": self.salidas,
            "inicializacion": self.inicializacion,
        }


class Dropout(Capa):
    """Apaga neuronas al azar durante el entrenamiento (dropout invertido)."""

    def __init__(self, tasa: float = 0.5, semilla: int | None = None) -> None:
        if not 0.0 <= tasa < 1.0:
            raise ValueError("La tasa de dropout debe estar en [0, 1).")
        self.tasa = tasa
        self._rng = np.random.default_rng(semilla)
        self._mascara: np.ndarray | None = None

    def adelante(self, x: np.ndarray, entrenando: bool = False) -> np.ndarray:
        if not entrenando or self.tasa == 0.0:
            self._mascara = None
            return x
        self._mascara = (self._rng.random(x.shape) >= self.tasa) / (1.0 - self.tasa)
        return x * self._mascara

    def atras(self, grad: np.ndarray) -> np.ndarray:
        return grad if self._mascara is None else grad * self._mascara

    def config(self) -> dict:
        return {"tasa": self.tasa}
