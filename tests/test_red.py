import numpy as np
import pytest

from red_neuronal import SGD, Adam, EntropiaCruzada, ErrorCuadraticoMedio, RedNeuronal, crear_mlp
from red_neuronal import datos
from red_neuronal.cli import main


def test_aprende_xor():
    X, y = datos.xor()
    red = crear_mlp(2, [8], 2, EntropiaCruzada(), Adam(0.05), activacion="tanh", semilla=0)
    red.entrenar(X, y, epocas=300, tam_lote=4, semilla=0, verbose=False)
    np.testing.assert_array_equal(red.predecir_clases(X), y)


def test_aprende_espiral_con_dropout():
    X, y = datos.espiral(100, 3, semilla=0)
    X_ent, X_pru, y_ent, y_pru = datos.dividir(X, y, semilla=0)
    red = crear_mlp(2, [64, 64], 3, EntropiaCruzada(), Adam(0.01), dropout=0.1, semilla=0)
    red.entrenar(X_ent, y_ent, epocas=200, semilla=0, verbose=False)
    _, precision = red.evaluar(X_pru, y_pru)
    assert precision > 0.9


def test_regresion_con_sgd():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(300, 1))
    y = np.sin(3 * X).ravel()
    red = crear_mlp(1, [32, 32], 1, ErrorCuadraticoMedio(), SGD(0.05, momento=0.9), activacion="tanh", semilla=0)
    historial = red.entrenar(X, y, epocas=200, semilla=0, verbose=False)
    assert historial["perdida"][-1] < 0.01
    assert historial["precision"] == []


def test_parada_temprana_restaura_mejores_pesos():
    X, y = datos.circulos(100, semilla=0)
    X_ent, X_pru, y_ent, y_pru = datos.dividir(X, y, semilla=0)
    red = crear_mlp(2, [16], 2, EntropiaCruzada(), Adam(0.05), semilla=0)
    historial = red.entrenar(
        X_ent, y_ent, epocas=500, datos_validacion=(X_pru, y_pru), paciencia=10, semilla=0, verbose=False
    )
    assert len(historial["perdida"]) < 500
    perdida_val, _ = red.evaluar(X_pru, y_pru)
    assert perdida_val == pytest.approx(min(historial["perdida_val"]))


def test_guardar_y_cargar(tmp_path):
    X, y = datos.espiral(50, 3, semilla=1)
    red = crear_mlp(2, [16, 8], 3, EntropiaCruzada(), dropout=0.2, semilla=1)
    red.entrenar(X, y, epocas=5, verbose=False)
    red.metadatos = {"clases": ["a", "b", "c"]}
    ruta = tmp_path / "modelo.npz"
    red.guardar(ruta)

    cargada = RedNeuronal.cargar(ruta)
    np.testing.assert_allclose(cargada.predecir(X), red.predecir(X))
    assert cargada.metadatos == {"clases": ["a", "b", "c"]}


def test_cli_csv_entrenar_evaluar_predecir(tmp_path, capsys):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2)) * [10, 0.1] + [50, 0]
    etiquetas = np.where(X[:, 0] - 50 > 0, "alto", "bajo")
    csv = tmp_path / "datos.csv"
    lineas = ["x1,x2,clase"] + [f"{a},{b},{c}" for (a, b), c in zip(X, etiquetas)]
    csv.write_text("\n".join(lineas))
    modelo = tmp_path / "modelo.npz"

    comunes = ["--datos", "csv", "--csv", str(csv), "--objetivo", "clase"]
    main(["entrenar", *comunes, "--epocas", "50", "--guardar", str(modelo), "--graficar", str(tmp_path)])
    main(["evaluar", "--modelo", str(modelo), *comunes])
    main(["predecir", "--modelo", str(modelo), "--entrada", "70,0", "--entrada", "30,0"])

    salida = capsys.readouterr().out
    assert "[70.0, 0.0] -> clase alto" in salida
    assert "[30.0, 0.0] -> clase bajo" in salida
    assert (tmp_path / "historial.png").exists()
    assert (tmp_path / "frontera.png").exists()
