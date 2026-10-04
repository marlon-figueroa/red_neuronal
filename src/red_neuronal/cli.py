"""Interfaz de línea de comandos: entrenar, evaluar y predecir."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import datos
from .activaciones import ACTIVACIONES
from .optimizadores import SGD, Adam
from .perdidas import EntropiaCruzada, ErrorCuadraticoMedio
from .red import RedNeuronal, crear_mlp


def _columna(valor: str) -> int | str:
    try:
        return int(valor)
    except ValueError:
        return valor


def _agregar_args_datos(parser: argparse.ArgumentParser) -> None:
    grupo = parser.add_argument_group("datos")
    grupo.add_argument("--datos", choices=["xor", "espiral", "circulos", "csv"], default="espiral")
    grupo.add_argument("--csv", type=Path, help="Ruta del CSV (con --datos csv).")
    grupo.add_argument(
        "--objetivo", type=_columna, default=-1, help="Columna objetivo del CSV (índice o nombre)."
    )
    grupo.add_argument("--tarea", choices=["clasificacion", "regresion"], default="clasificacion")
    grupo.add_argument("--muestras", type=int, default=200, help="Muestras por clase (datos sintéticos).")
    grupo.add_argument("--clases", type=int, default=3, help="Número de clases (espiral).")
    grupo.add_argument("--ruido", type=float, default=0.2, help="Ruido (datos sintéticos).")
    grupo.add_argument("--semilla", type=int, default=42)


def _cargar_datos(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[str] | None, str]:
    """Devuelve (X, y, clases, tarea)."""
    if args.datos == "xor":
        X, y = datos.xor()
        return X, y, ["0", "1"], "clasificacion"
    if args.datos == "espiral":
        X, y = datos.espiral(args.muestras, args.clases, args.ruido, args.semilla)
        return X, y, [str(c) for c in range(args.clases)], "clasificacion"
    if args.datos == "circulos":
        X, y = datos.circulos(args.muestras, args.ruido / 2, semilla=args.semilla)
        return X, y, ["0", "1"], "clasificacion"
    if args.csv is None:
        raise SystemExit("Con --datos csv debes indicar --csv RUTA.")
    X, y, clases = datos.cargar_csv(args.csv, args.objetivo, args.tarea)
    return X, y, clases, args.tarea


def _estandarizador(red: RedNeuronal) -> datos.Estandarizador | None:
    meta = red.metadatos
    if "media" not in meta:
        return None
    return datos.Estandarizador(meta["media"], meta["desv"])


def _imprimir_metricas(titulo: str, perdida: float, precision: float | None) -> None:
    texto = f"{titulo} - pérdida: {perdida:.4f}"
    if precision is not None:
        texto += f" - precisión: {precision:.2%}"
    print(texto)


def comando_entrenar(args: argparse.Namespace) -> None:
    X, y, clases, tarea = _cargar_datos(args)

    if args.datos == "xor":
        X_ent, X_pru, y_ent, y_pru = X, X, y, y
    else:
        X_ent, X_pru, y_ent, y_pru = datos.dividir(X, y, args.prop_prueba, args.semilla)

    estandarizador = None
    if args.datos == "csv":
        estandarizador = datos.Estandarizador().ajustar(X_ent)
        X_ent, X_pru = estandarizador.transformar(X_ent), estandarizador.transformar(X_pru)

    if tarea == "clasificacion":
        perdida, salidas = EntropiaCruzada(), len(clases)
    else:
        perdida, salidas = ErrorCuadraticoMedio(), 1

    if args.optimizador == "adam":
        optimizador = Adam(args.lr, decaimiento_peso=args.decaimiento_peso)
    else:
        optimizador = SGD(args.lr, momento=args.momento, decaimiento_peso=args.decaimiento_peso)

    red = crear_mlp(
        X.shape[1],
        args.ocultas,
        salidas,
        perdida,
        optimizador,
        activacion=args.activacion,
        dropout=args.dropout,
        semilla=args.semilla,
    )
    print(f"Datos: {len(X_ent)} de entrenamiento, {len(X_pru)} de prueba, {X.shape[1]} características")
    print("Arquitectura:")
    print(red.resumen())
    print()

    historial = red.entrenar(
        X_ent,
        y_ent,
        epocas=args.epocas,
        tam_lote=args.lote,
        datos_validacion=None if args.datos == "xor" else (X_pru, y_pru),
        paciencia=args.paciencia,
        semilla=args.semilla,
        cada=args.cada,
    )
    print()
    _imprimir_metricas("Resultado en prueba", *red.evaluar(X_pru, y_pru))

    if args.guardar:
        red.metadatos = {"tarea": tarea, "clases": clases}
        if estandarizador is not None:
            red.metadatos.update(media=estandarizador.media.tolist(), desv=estandarizador.desv.tolist())
        red.guardar(args.guardar)
        print(f"Modelo guardado en {args.guardar}")

    if args.graficar:
        from .graficos import graficar_frontera, graficar_historial

        args.graficar.mkdir(parents=True, exist_ok=True)
        graficar_historial(historial, args.graficar / "historial.png")
        print(f"Gráfica guardada en {args.graficar / 'historial.png'}")
        if tarea == "clasificacion" and X.shape[1] == 2:
            X_todo = X if estandarizador is None else estandarizador.transformar(X)
            graficar_frontera(red, X_todo, y, args.graficar / "frontera.png")
            print(f"Gráfica guardada en {args.graficar / 'frontera.png'}")


def comando_evaluar(args: argparse.Namespace) -> None:
    red = RedNeuronal.cargar(args.modelo)
    X, y, _, _ = _cargar_datos(args)
    estandarizador = _estandarizador(red)
    if estandarizador is not None:
        X = estandarizador.transformar(X)
    _imprimir_metricas(f"Evaluación sobre {len(X)} muestras", *red.evaluar(X, y))


def comando_predecir(args: argparse.Namespace) -> None:
    red = RedNeuronal.cargar(args.modelo)
    X = np.array([[float(v) for v in fila.split(",")] for fila in args.entrada])
    estandarizador = _estandarizador(red)
    X_red = X if estandarizador is None else estandarizador.transformar(X)

    if red.metadatos.get("tarea", "clasificacion") == "regresion":
        for fila, valor in zip(X, red.predecir(X_red).ravel()):
            print(f"{fila.tolist()} -> {valor:.4f}")
        return

    clases = red.metadatos.get("clases")
    probabilidades = red.predecir(X_red)
    for fila, p in zip(X, probabilidades):
        indice = int(p.argmax())
        nombre = clases[indice] if clases else str(indice)
        print(f"{fila.tolist()} -> clase {nombre} (confianza {p[indice]:.2%})")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="red-neuronal", description="Entrena y usa una red neuronal hecha con NumPy."
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("entrenar", help="Entrena una red nueva.")
    _agregar_args_datos(p)
    modelo = p.add_argument_group("modelo")
    modelo.add_argument("--ocultas", type=int, nargs="+", default=[64, 64], help="Neuronas por capa oculta.")
    modelo.add_argument("--activacion", choices=list(ACTIVACIONES), default="relu")
    modelo.add_argument("--dropout", type=float, default=0.0)
    ent = p.add_argument_group("entrenamiento")
    ent.add_argument("--optimizador", choices=["adam", "sgd"], default="adam")
    ent.add_argument("--lr", type=float, default=0.01, help="Tasa de aprendizaje.")
    ent.add_argument("--momento", type=float, default=0.9, help="Momento (solo SGD).")
    ent.add_argument("--decaimiento-peso", type=float, default=0.0, help="Regularización L2.")
    ent.add_argument("--epocas", type=int, default=200)
    ent.add_argument("--lote", type=int, default=32, help="Tamaño del mini-lote.")
    ent.add_argument("--paciencia", type=int, default=None, help="Épocas sin mejora antes de parar.")
    ent.add_argument("--prop-prueba", type=float, default=0.2, help="Proporción de datos para prueba.")
    ent.add_argument("--cada", type=int, default=10, help="Imprimir progreso cada N épocas.")
    salida = p.add_argument_group("salida")
    salida.add_argument("--guardar", type=Path, help="Ruta .npz donde guardar el modelo.")
    salida.add_argument("--graficar", type=Path, help="Carpeta donde guardar las gráficas.")
    p.set_defaults(funcion=comando_entrenar)

    p = sub.add_parser("evaluar", help="Evalúa un modelo guardado.")
    p.add_argument("--modelo", type=Path, required=True)
    _agregar_args_datos(p)
    p.set_defaults(funcion=comando_evaluar)

    p = sub.add_parser("predecir", help="Predice con un modelo guardado.")
    p.add_argument("--modelo", type=Path, required=True)
    p.add_argument(
        "--entrada",
        action="append",
        required=True,
        help=(
            "Valores separados por coma, p. ej. --entrada 0.1,0.5 (se puede repetir). "
            "Con negativos usa --entrada=-0.1,0.5."
        ),
    )
    p.set_defaults(funcion=comando_predecir)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = construir_parser().parse_args(argv)
    args.funcion(args)


if __name__ == "__main__":
    main()
