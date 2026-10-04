"""Compara la retropropagación con gradientes numéricos (diferencias finitas)."""

import numpy as np
import pytest

from red_neuronal import (
    Densa,
    EntropiaCruzada,
    EntropiaCruzadaBinaria,
    ErrorCuadraticoMedio,
    LeakyReLU,
    RedNeuronal,
    Sigmoide,
    Tanh,
)


def _gradiente_numerico(red, X, y, parametro, eps=1e-6):
    grad = np.zeros_like(parametro)
    for idx in np.ndindex(parametro.shape):
        original = parametro[idx]
        parametro[idx] = original + eps
        mas = red.perdida.calcular(red.adelante(X), y)
        parametro[idx] = original - eps
        menos = red.perdida.calcular(red.adelante(X), y)
        parametro[idx] = original
        grad[idx] = (mas - menos) / (2 * eps)
    return grad


@pytest.mark.parametrize(
    "perdida, salidas, objetivo",
    [
        (ErrorCuadraticoMedio(), 2, lambda rng, n: rng.normal(size=(n, 2))),
        (EntropiaCruzadaBinaria(), 1, lambda rng, n: rng.integers(0, 2, n)),
        (EntropiaCruzada(), 3, lambda rng, n: rng.integers(0, 3, n)),
    ],
)
def test_retropropagacion_coincide_con_gradiente_numerico(perdida, salidas, objetivo):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 4))
    y = objetivo(rng, 6)
    red = RedNeuronal(
        [
            Densa(4, 5, semilla=1),
            Tanh(),
            Densa(5, 4, semilla=2),
            LeakyReLU(0.1),
            Densa(4, 3, semilla=3),
            Sigmoide(),
            Densa(3, salidas, semilla=4),
        ],
        perdida,
    )

    red.atras(perdida.gradiente(red.adelante(X), y))
    analiticos = [g.copy() for _, g in red.parametros()]

    for (parametro, _), analitico in zip(red.parametros(), analiticos):
        numerico = _gradiente_numerico(red, X, y, parametro)
        np.testing.assert_allclose(analitico, numerico, rtol=1e-4, atol=1e-7)
