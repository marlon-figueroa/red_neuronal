"""Gráficas opcionales (requiere matplotlib: pip install -e '.[graficos]')."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .red import RedNeuronal


def _pyplot():
    try:
        import matplotlib
    except ImportError as error:
        raise ImportError(
            "Las gráficas requieren matplotlib. Instálalo con: pip install -e '.[graficos]'"
        ) from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def graficar_historial(historial: dict[str, list[float]], ruta: str | Path) -> None:
    plt = _pyplot()
    con_precision = bool(historial["precision"])
    fig, ejes = plt.subplots(1, 2 if con_precision else 1, figsize=(12 if con_precision else 6, 4))
    ejes = np.atleast_1d(ejes)

    ejes[0].plot(historial["perdida"], label="entrenamiento")
    if historial["perdida_val"]:
        ejes[0].plot(historial["perdida_val"], label="validación")
    ejes[0].set(title="Pérdida", xlabel="Época")
    ejes[0].legend()

    if con_precision:
        ejes[1].plot(historial["precision"], label="entrenamiento")
        if historial["precision_val"]:
            ejes[1].plot(historial["precision_val"], label="validación")
        ejes[1].set(title="Precisión", xlabel="Época", ylim=(0, 1.05))
        ejes[1].legend()

    _guardar(fig, ruta)


def graficar_frontera(red: RedNeuronal, X: np.ndarray, y: np.ndarray, ruta: str | Path) -> None:
    """Frontera de decisión para clasificación con 2 características."""
    if X.shape[1] != 2:
        raise ValueError("La frontera de decisión solo se puede graficar con 2 características.")
    plt = _pyplot()
    margen = 0.5
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - margen, X[:, 0].max() + margen, 300),
        np.linspace(X[:, 1].min() - margen, X[:, 1].max() + margen, 300),
    )
    Z = red.predecir_clases(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolors="k", s=20)
    ax.set_title("Frontera de decisión")
    _guardar(fig, ruta)


def _guardar(fig, ruta: str | Path) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    _pyplot().close(fig)
