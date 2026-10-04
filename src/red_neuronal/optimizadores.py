"""Optimizadores: actualizan los parámetros a partir de sus gradientes."""

from __future__ import annotations

import numpy as np

from .capas import Parametro


class Optimizador:
    def __init__(self, tasa_aprendizaje: float, decaimiento_peso: float = 0.0) -> None:
        self.tasa_aprendizaje = tasa_aprendizaje
        self.decaimiento_peso = decaimiento_peso

    def paso(self, parametros: list[Parametro]) -> None:
        raise NotImplementedError

    def _gradiente(self, parametro: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.decaimiento_peso:
            return grad + self.decaimiento_peso * parametro
        return grad


class SGD(Optimizador):
    """Descenso de gradiente estocástico, opcionalmente con momento."""

    def __init__(
        self,
        tasa_aprendizaje: float = 0.01,
        momento: float = 0.0,
        decaimiento_peso: float = 0.0,
    ) -> None:
        super().__init__(tasa_aprendizaje, decaimiento_peso)
        self.momento = momento
        self._velocidades: dict[int, np.ndarray] = {}

    def paso(self, parametros: list[Parametro]) -> None:
        for i, (p, g) in enumerate(parametros):
            g = self._gradiente(p, g)
            if self.momento:
                v = self._velocidades.setdefault(i, np.zeros_like(p))
                v *= self.momento
                v -= self.tasa_aprendizaje * g
                p += v
            else:
                p -= self.tasa_aprendizaje * g


class Adam(Optimizador):
    def __init__(
        self,
        tasa_aprendizaje: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        decaimiento_peso: float = 0.0,
    ) -> None:
        super().__init__(tasa_aprendizaje, decaimiento_peso)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}
        self._t = 0

    def paso(self, parametros: list[Parametro]) -> None:
        self._t += 1
        correccion1 = 1.0 - self.beta1**self._t
        correccion2 = 1.0 - self.beta2**self._t
        for i, (p, g) in enumerate(parametros):
            g = self._gradiente(p, g)
            m = self._m.setdefault(i, np.zeros_like(p))
            v = self._v.setdefault(i, np.zeros_like(p))
            m[...] = self.beta1 * m + (1.0 - self.beta1) * g
            v[...] = self.beta2 * v + (1.0 - self.beta2) * g * g
            p -= self.tasa_aprendizaje * (m / correccion1) / (np.sqrt(v / correccion2) + self.epsilon)


OPTIMIZADORES = {"sgd": SGD, "adam": Adam}
