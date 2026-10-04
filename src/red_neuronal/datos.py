"""Conjuntos de datos de ejemplo y utilidades de preprocesamiento."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def xor() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    return X, y


def espiral(
    n_por_clase: int = 100,
    clases: int = 3,
    ruido: float = 0.2,
    semilla: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Espirales entrelazadas en 2D: no separables linealmente."""
    rng = np.random.default_rng(semilla)
    X = np.zeros((n_por_clase * clases, 2))
    y = np.zeros(n_por_clase * clases, dtype=int)
    for c in range(clases):
        idx = slice(n_por_clase * c, n_por_clase * (c + 1))
        radio = np.linspace(0.0, 1.0, n_por_clase)
        angulo = np.linspace(c * 4.0, (c + 1) * 4.0, n_por_clase) + rng.normal(0, ruido, n_por_clase)
        X[idx] = np.column_stack([radio * np.sin(angulo), radio * np.cos(angulo)])
        y[idx] = c
    return X, y


def circulos(
    n_por_clase: int = 250,
    ruido: float = 0.1,
    factor: float = 0.5,
    semilla: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dos círculos concéntricos (clase 0 afuera, clase 1 adentro)."""
    rng = np.random.default_rng(semilla)
    angulos = rng.uniform(0, 2 * np.pi, 2 * n_por_clase)
    radios = np.r_[np.ones(n_por_clase), np.full(n_por_clase, factor)]
    X = np.column_stack([radios * np.cos(angulos), radios * np.sin(angulos)])
    X += rng.normal(0, ruido, X.shape)
    y = np.r_[np.zeros(n_por_clase, dtype=int), np.ones(n_por_clase, dtype=int)]
    return X, y


def cargar_csv(
    ruta: str | Path,
    objetivo: int | str = -1,
    tarea: str = "clasificacion",
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    """Carga un CSV con encabezado. Todas las columnas salvo `objetivo` deben ser numéricas.

    `objetivo` puede ser el índice o el nombre de la columna a predecir.
    Devuelve (X, y, clases). En clasificación, `y` son índices enteros sobre
    `clases`; en regresión, `y` es float y `clases` es None.
    """
    with open(ruta, newline="", encoding="utf-8") as f:
        lector = csv.reader(f)
        encabezado = next(lector)
        filas = [fila for fila in lector if fila]

    columna = encabezado.index(objetivo) if isinstance(objetivo, str) else objetivo % len(encabezado)
    X = np.array([[float(v) for j, v in enumerate(fila) if j != columna] for fila in filas])
    valores = [fila[columna].strip() for fila in filas]

    if tarea == "regresion":
        return X, np.array(valores, dtype=float), None
    if tarea != "clasificacion":
        raise ValueError("tarea debe ser 'clasificacion' o 'regresion'.")

    unicos = set(valores)
    try:
        clases = sorted(unicos, key=float)
    except ValueError:
        clases = sorted(unicos)
    indice = {c: i for i, c in enumerate(clases)}
    return X, np.array([indice[v] for v in valores]), clases


def dividir(
    X: np.ndarray,
    y: np.ndarray,
    prop_prueba: float = 0.2,
    semilla: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide en (X_entrenamiento, X_prueba, y_entrenamiento, y_prueba)."""
    indices = np.random.default_rng(semilla).permutation(len(X))
    corte = int(len(X) * (1 - prop_prueba))
    ent, pru = indices[:corte], indices[corte:]
    return X[ent], X[pru], y[ent], y[pru]


class Estandarizador:
    """Lleva cada característica a media 0 y desviación estándar 1."""

    def __init__(self, media: np.ndarray | None = None, desv: np.ndarray | None = None) -> None:
        self.media = None if media is None else np.asarray(media, dtype=float)
        self.desv = None if desv is None else np.asarray(desv, dtype=float)

    def ajustar(self, X: np.ndarray) -> "Estandarizador":
        self.media = X.mean(axis=0)
        desv = X.std(axis=0)
        self.desv = np.where(desv == 0, 1.0, desv)
        return self

    def transformar(self, X: np.ndarray) -> np.ndarray:
        return (X - self.media) / self.desv

    def ajustar_transformar(self, X: np.ndarray) -> np.ndarray:
        return self.ajustar(X).transformar(X)
