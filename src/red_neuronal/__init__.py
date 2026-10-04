"""Red neuronal (perceptrón multicapa) implementada desde cero con NumPy."""

from .activaciones import LeakyReLU, ReLU, Sigmoide, Tanh
from .capas import Capa, Densa, Dropout
from .optimizadores import SGD, Adam
from .perdidas import EntropiaCruzada, EntropiaCruzadaBinaria, ErrorCuadraticoMedio
from .red import RedNeuronal, crear_mlp

__all__ = [
    "Adam",
    "Capa",
    "Densa",
    "Dropout",
    "EntropiaCruzada",
    "EntropiaCruzadaBinaria",
    "ErrorCuadraticoMedio",
    "LeakyReLU",
    "RedNeuronal",
    "ReLU",
    "SGD",
    "Sigmoide",
    "Tanh",
    "crear_mlp",
]

__version__ = "0.1.0"
