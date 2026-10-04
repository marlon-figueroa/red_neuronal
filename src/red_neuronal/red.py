"""La red neuronal: une capas, pérdida y optimizador, y sabe entrenarse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from . import perdidas as _perdidas
from .activaciones import ACTIVACIONES, LeakyReLU, ReLU, Sigmoide, Tanh
from .capas import Capa, Densa, Dropout, Parametro
from .optimizadores import Adam, Optimizador

_CAPAS = {cls.__name__: cls for cls in (Densa, Dropout, ReLU, LeakyReLU, Sigmoide, Tanh)}
_PERDIDAS = {
    cls.__name__: cls
    for cls in (
        _perdidas.ErrorCuadraticoMedio,
        _perdidas.EntropiaCruzadaBinaria,
        _perdidas.EntropiaCruzada,
    )
}


class RedNeuronal:
    def __init__(
        self,
        capas: Sequence[Capa],
        perdida: _perdidas.Perdida,
        optimizador: Optimizador | None = None,
        metadatos: dict | None = None,
    ) -> None:
        self.capas = list(capas)
        self.perdida = perdida
        self.optimizador = optimizador or Adam()
        self.metadatos = metadatos or {}

    # ------------------------------------------------------------------ cálculo
    def adelante(self, X: np.ndarray, entrenando: bool = False) -> np.ndarray:
        for capa in self.capas:
            X = capa.adelante(X, entrenando)
        return X

    def atras(self, grad: np.ndarray) -> None:
        for capa in reversed(self.capas):
            grad = capa.atras(grad)

    def parametros(self) -> list[Parametro]:
        return [par for capa in self.capas for par in capa.parametros()]

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Probabilidades (clasificación) o valores (regresión)."""
        return self.perdida.activar(self.adelante(X))

    def predecir_clases(self, X: np.ndarray) -> np.ndarray:
        return self.perdida.clases(self.adelante(X))

    def evaluar(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float | None]:
        """Devuelve (pérdida, precisión). La precisión es None en regresión."""
        salida = self.adelante(X)
        return self.perdida.calcular(salida, y), self.perdida.precision(salida, y)

    # ------------------------------------------------------------ entrenamiento
    def entrenar(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epocas: int = 100,
        tam_lote: int = 32,
        datos_validacion: tuple[np.ndarray, np.ndarray] | None = None,
        paciencia: int | None = None,
        semilla: int | None = None,
        cada: int = 10,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Entrena con descenso de gradiente por mini-lotes.

        Si se indica `paciencia` (y hay datos de validación), se detiene cuando la
        pérdida de validación no mejora en esa cantidad de épocas y restaura los
        mejores pesos encontrados.
        """
        rng = np.random.default_rng(semilla)
        historial: dict[str, list[float]] = {
            "perdida": [],
            "precision": [],
            "perdida_val": [],
            "precision_val": [],
        }
        mejor_perdida = np.inf
        mejores_pesos: list[np.ndarray] | None = None
        epocas_sin_mejora = 0

        for epoca in range(1, epocas + 1):
            indices = rng.permutation(len(X))
            for inicio in range(0, len(X), tam_lote):
                lote = indices[inicio : inicio + tam_lote]
                salida = self.adelante(X[lote], entrenando=True)
                self.atras(self.perdida.gradiente(salida, y[lote]))
                self.optimizador.paso(self.parametros())

            perdida, precision = self.evaluar(X, y)
            historial["perdida"].append(perdida)
            if precision is not None:
                historial["precision"].append(precision)

            mensaje = f"Época {epoca:>4}/{epocas} - pérdida: {perdida:.4f}"
            if precision is not None:
                mensaje += f" - precisión: {precision:.2%}"

            if datos_validacion is not None:
                perdida_val, precision_val = self.evaluar(*datos_validacion)
                historial["perdida_val"].append(perdida_val)
                mensaje += f" - val_pérdida: {perdida_val:.4f}"
                if precision_val is not None:
                    historial["precision_val"].append(precision_val)
                    mensaje += f" - val_precisión: {precision_val:.2%}"

                if paciencia is not None:
                    if perdida_val < mejor_perdida:
                        mejor_perdida = perdida_val
                        mejores_pesos = [p.copy() for p, _ in self.parametros()]
                        epocas_sin_mejora = 0
                    else:
                        epocas_sin_mejora += 1

            if verbose and (epoca % cada == 0 or epoca == 1 or epoca == epocas):
                print(mensaje)

            if paciencia is not None and epocas_sin_mejora >= paciencia:
                if verbose:
                    print(f"Parada temprana en la época {epoca}: sin mejora en {paciencia} épocas.")
                break

        if mejores_pesos is not None:
            for (p, _), mejor in zip(self.parametros(), mejores_pesos):
                p[...] = mejor
        return historial

    # -------------------------------------------------------------- utilidades
    def resumen(self) -> str:
        lineas = []
        total = 0
        for i, capa in enumerate(self.capas):
            n = sum(p.size for p, _ in capa.parametros())
            total += n
            lineas.append(f"  {i:>2}. {capa!r:<55} parámetros: {n}")
        lineas.append(f"  Pérdida: {type(self.perdida).__name__}")
        lineas.append(f"  Total de parámetros entrenables: {total}")
        return "\n".join(lineas)

    def guardar(self, ruta: str | Path) -> None:
        """Guarda arquitectura, pesos y metadatos en un único archivo .npz."""
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        arquitectura = {
            "capas": [{"tipo": type(c).__name__, "config": c.config()} for c in self.capas],
            "perdida": type(self.perdida).__name__,
            "metadatos": self.metadatos,
        }
        pesos = {f"p{i}": p for i, (p, _) in enumerate(self.parametros())}
        np.savez(ruta, arquitectura=np.array(json.dumps(arquitectura)), **pesos)

    @classmethod
    def cargar(cls, ruta: str | Path, optimizador: Optimizador | None = None) -> "RedNeuronal":
        with np.load(ruta, allow_pickle=False) as archivo:
            arquitectura = json.loads(str(archivo["arquitectura"]))
            capas = [_CAPAS[c["tipo"]](**c["config"]) for c in arquitectura["capas"]]
            red = cls(
                capas,
                _PERDIDAS[arquitectura["perdida"]](),
                optimizador,
                arquitectura.get("metadatos"),
            )
            for i, (p, _) in enumerate(red.parametros()):
                p[...] = archivo[f"p{i}"]
        return red


def crear_mlp(
    entradas: int,
    ocultas: Sequence[int],
    salidas: int,
    perdida: _perdidas.Perdida,
    optimizador: Optimizador | None = None,
    activacion: str = "relu",
    dropout: float = 0.0,
    semilla: int | None = None,
) -> RedNeuronal:
    """Construye un perceptrón multicapa: [Densa -> activación (-> Dropout)]* -> Densa."""
    if activacion not in ACTIVACIONES:
        raise ValueError(f"Activación desconocida: {activacion!r}. Opciones: {list(ACTIVACIONES)}")
    inicializacion = "he" if activacion in ("relu", "leaky_relu") else "xavier"
    semillas = np.random.SeedSequence(semilla).generate_state(2 * len(ocultas) + 1)

    capas: list[Capa] = []
    anterior = entradas
    for i, neuronas in enumerate(ocultas):
        capas.append(Densa(anterior, neuronas, inicializacion, semilla=int(semillas[2 * i])))
        capas.append(ACTIVACIONES[activacion]())
        if dropout > 0:
            capas.append(Dropout(dropout, semilla=int(semillas[2 * i + 1])))
        anterior = neuronas
    capas.append(Densa(anterior, salidas, "xavier", semilla=int(semillas[-1])))
    return RedNeuronal(capas, perdida, optimizador)
