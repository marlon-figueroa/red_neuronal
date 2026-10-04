"""Ejemplo de uso de la librería desde Python (sin la CLI).

Ejecuta: python ejemplos/entrenar_espiral.py
"""

from red_neuronal import Adam, Densa, Dropout, EntropiaCruzada, ReLU, RedNeuronal
from red_neuronal import datos

X, y = datos.espiral(n_por_clase=200, clases=3, semilla=0)
X_ent, X_pru, y_ent, y_pru = datos.dividir(X, y, prop_prueba=0.2, semilla=0)

# La arquitectura se arma capa por capa; también puedes usar crear_mlp(...).
red = RedNeuronal(
    capas=[
        Densa(2, 64, semilla=1),
        ReLU(),
        Dropout(0.1, semilla=2),
        Densa(64, 64, semilla=3),
        ReLU(),
        Densa(64, 3, inicializacion="xavier", semilla=4),
    ],
    perdida=EntropiaCruzada(),
    optimizador=Adam(tasa_aprendizaje=0.01),
)
print(red.resumen())

historial = red.entrenar(
    X_ent,
    y_ent,
    epocas=300,
    tam_lote=32,
    datos_validacion=(X_pru, y_pru),
    paciencia=40,
    semilla=0,
    cada=25,
)

perdida, precision = red.evaluar(X_pru, y_pru)
print(f"\nPrueba -> pérdida: {perdida:.4f}, precisión: {precision:.2%}")

red.guardar("modelos/espiral.npz")
print("Modelo guardado en modelos/espiral.npz")

try:
    from red_neuronal.graficos import graficar_frontera, graficar_historial

    graficar_historial(historial, "resultados/historial.png")
    graficar_frontera(red, X, y, "resultados/frontera.png")
    print("Gráficas guardadas en resultados/")
except ImportError as error:
    print(error)
