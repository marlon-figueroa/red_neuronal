---
title: "Procedimiento de la red neuronal"
subtitle: "Definición, funcionamiento y aprendizaje del perceptrón multicapa del proyecto `red_neuronal`"
author: "Marlon E. Figueroa"
date: "Octubre de 2026"
lang: es
documentclass: article
fontsize: 11pt
geometry: margin=2.5cm
toc: true
toc-depth: 2
numbersections: true
colorlinks: true
linkcolor: blue
urlcolor: blue
header-includes: |
  \usepackage{amsmath,amssymb}
  \usepackage{float}
  \floatplacement{figure}{H}
  \AtBeginDocument{\renewcommand{\tablename}{Tabla}}
---

# Introducción

Este documento describe **cómo está definida, cómo funciona y cómo aprende** la red
neuronal implementada en el proyecto `red_neuronal`. La implementación está hecha
desde cero con NumPy, por lo que cada concepto matemático de este documento tiene
una correspondencia directa con una clase o función del código.

La red es un **perceptrón multicapa** (MLP, *multilayer perceptron*): una sucesión de
capas densas, cada una seguida de una función de activación no lineal. Se entrena
con **descenso de gradiente por mini-lotes**, y los gradientes se obtienen mediante
**retropropagación** (*backpropagation*).

Notación usada a lo largo del documento:

| Símbolo | Significado |
| --- | --- |
| $N$ | Número de ejemplos en un lote |
| $d$ | Número de características de entrada |
| $\mathbf{x} \in \mathbb{R}^{d}$ | Un ejemplo de entrada (vector fila) |
| $X \in \mathbb{R}^{N \times d}$ | Lote de entradas, un ejemplo por fila |
| $W^{(l)}, \mathbf{b}^{(l)}$ | Pesos y sesgos de la capa densa $l$ |
| $Z^{(l)}$ | Salida lineal (pre-activación) de la capa $l$ |
| $A^{(l)}$ | Salida de la activación de la capa $l$, con $A^{(0)} = X$ |
| $\varphi$ | Función de activación |
| $\mathcal{L}$ | Función de pérdida |
| $\eta$ | Tasa de aprendizaje |
| $\theta$ | Cualquier parámetro entrenable ($W$ o $\mathbf{b}$) |

# Arquitectura del proyecto

El código está dividido en módulos con responsabilidades separadas. La clase
`RedNeuronal` coordina a los demás: contiene una lista de capas, una función de
pérdida y un optimizador.

```{.mermaid caption="Módulos del paquete y sus dependencias." width="90%"}
flowchart LR
    CLI["cli.py<br/>entrenar · evaluar · predecir"] --> RED
    CLI --> DATOS["datos.py<br/>xor · espiral · círculos · CSV<br/>dividir · Estandarizador"]
    CLI --> GRAF["graficos.py<br/>historial · frontera"]
    RED["red.py<br/>RedNeuronal · crear_mlp"] --> CAPAS["capas.py<br/>Capa · Densa · Dropout"]
    RED --> ACT["activaciones.py<br/>ReLU · LeakyReLU<br/>Sigmoide · Tanh"]
    RED --> PERD["perdidas.py<br/>ECM · Entropía cruzada"]
    RED --> OPT["optimizadores.py<br/>SGD · Adam"]
    ACT -.->|"heredan de Capa"| CAPAS
```

| Módulo | Contenido | Sección |
| --- | --- | --- |
| `capas.py` | Interfaz `Capa`, capa `Densa`, `Dropout` | [Capas](#capas) |
| `activaciones.py` | `ReLU`, `LeakyReLU`, `Sigmoide`, `Tanh` | [Activaciones](#funciones-de-activación) |
| `perdidas.py` | Error cuadrático medio y entropías cruzadas | [Pérdidas](#funciones-de-pérdida) |
| `optimizadores.py` | `SGD` (con momento) y `Adam` | [Optimizadores](#optimizadores) |
| `red.py` | `RedNeuronal` y `crear_mlp` | [Entrenamiento](#proceso-de-entrenamiento) |
| `datos.py` | Conjuntos de datos y preprocesamiento | [Evaluación](#evaluación-y-generalización) |

# La neurona artificial

## Modelo matemático

La neurona es la unidad básica de cálculo. Recibe un vector de entradas
$\mathbf{x} = (x_1, \dots, x_d)$, multiplica cada entrada por un **peso** $w_i$, suma
los resultados junto con un **sesgo** $b$ y aplica una **función de activación**
$\varphi$:

$$
z = \sum_{i=1}^{d} w_i x_i + b = \mathbf{w} \cdot \mathbf{x} + b,
\qquad
a = \varphi(z).
$$

- Los **pesos** $w_i$ indican cuánto influye cada entrada y en qué sentido (positivo o
  negativo).
- El **sesgo** $b$ desplaza el umbral a partir del cual la neurona se activa.
- La **activación** $\varphi$ introduce no linealidad; sin ella, la neurona sería una
  simple función lineal.

```{.mermaid caption="Una neurona: suma ponderada de las entradas más el sesgo, seguida de una activación." width="80%"}
flowchart LR
    x1(("x₁")) -->|"w₁"| S
    x2(("x₂")) -->|"w₂"| S
    xd(("x_d")) -->|"w_d"| S
    b(("1")) -->|"b"| S
    S(("Σ<br/>z = w·x + b")) --> F["φ(z)<br/>activación"]
    F --> A(("a"))
```

Los pesos y el sesgo son los **parámetros** de la neurona: son los valores que la red
ajusta durante el entrenamiento. Aprender consiste exactamente en encontrar valores
de $\mathbf{w}$ y $b$ que hagan que la salida $a$ se parezca a la salida deseada.

## Interpretación geométrica y necesidad de varias capas

La ecuación $\mathbf{w} \cdot \mathbf{x} + b = 0$ define un **hiperplano** (una recta en
dos dimensiones). Una sola neurona solo puede separar las clases con esa frontera
lineal. El ejemplo clásico de su limitación es **XOR**:

| $x_1$ | $x_2$ | XOR |
| :---: | :---: | :---: |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Ninguna recta separa los puntos con salida 1 de los puntos con salida 0. Al apilar
neuronas en capas con activaciones no lineales, cada capa transforma el espacio de
entrada y la red puede formar fronteras curvas arbitrarias. Por eso el proyecto
resuelve XOR con una capa oculta de 8 neuronas, y las espirales con dos capas
ocultas de 64.

# Funciones de activación

Las activaciones se implementan como capas sin parámetros en `activaciones.py`. Se
aplican **elemento a elemento**, así que la derivada que se necesita en la
retropropagación es simplemente $\varphi'(z)$ en cada posición.

| Nombre | $\varphi(z)$ | $\varphi'(z)$ | Clase |
| --- | --- | --- | --- |
| ReLU | $\max(0, z)$ | $1$ si $z > 0$, si no $0$ | `ReLU` |
| Leaky ReLU | $z$ si $z > 0$, si no $\alpha z$ | $1$ si $z > 0$, si no $\alpha$ | `LeakyReLU` |
| Sigmoide | $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ | $\sigma(z)\,(1 - \sigma(z))$ | `Sigmoide` |
| Tangente hiperbólica | $\tanh(z)$ | $1 - \tanh^2(z)$ | `Tanh` |

Observaciones:

- **ReLU** es la activación por defecto. Es barata de calcular y no se satura para
  valores positivos, lo que mantiene gradientes útiles en redes profundas.
- **Leaky ReLU** usa una pendiente pequeña $\alpha$ (por defecto $0.01$) para $z < 0$,
  de modo que las neuronas nunca quedan con gradiente exactamente cero.
- **Sigmoide** y **Tanh** se saturan: para $|z|$ grande su derivada es casi cero.
  Funcionan bien en redes pequeñas, como la de XOR.
- La sigmoide se calcula con la identidad
  $\sigma(z) = \tfrac{1}{2}\left(1 + \tanh\left(\tfrac{z}{2}\right)\right)$, que es
  equivalente pero evita el desbordamiento de $e^{-z}$ cuando $z$ es muy negativo.

Las activaciones guardan en `adelante` lo necesario para la derivada (la máscara
$z > 0$ en ReLU, la salida en Sigmoide y Tanh), y lo reutilizan en `atras`:

```python
class Sigmoide(Capa):
    def adelante(self, x, entrenando=False):
        self._salida = sigmoide(x)
        return self._salida

    def atras(self, grad):
        return grad * self._salida * (1.0 - self._salida)
```

# Capas

Todas las capas heredan de la clase `Capa`, que define un contrato de cuatro
métodos:

| Método | Responsabilidad |
| --- | --- |
| `adelante(x, entrenando)` | Calcula la salida de la capa y guarda lo necesario para el retroceso |
| `atras(grad)` | Recibe $\partial\mathcal{L}/\partial\,\text{salida}$, calcula los gradientes de sus parámetros y devuelve $\partial\mathcal{L}/\partial\,\text{entrada}$ |
| `parametros()` | Devuelve pares (parámetro, gradiente) para el optimizador |
| `config()` | Devuelve los argumentos necesarios para reconstruir la capa al cargar un modelo |

Gracias a este contrato, la red no necesita saber qué tipo de capa está usando:
simplemente encadena `adelante` en orden y `atras` en orden inverso.

## Capa densa

Una capa densa (`Densa`) es un conjunto de $m$ neuronas que reciben las mismas $n$
entradas. En lugar de calcular neurona por neurona, se agrupan los pesos en una
matriz y se procesa un lote completo de $N$ ejemplos con una sola multiplicación:

$$
Z = X W + \mathbf{b},
\qquad
X \in \mathbb{R}^{N \times n},\;
W \in \mathbb{R}^{n \times m},\;
\mathbf{b} \in \mathbb{R}^{m},\;
Z \in \mathbb{R}^{N \times m}.
$$

La columna $j$ de $W$ contiene los pesos de la neurona $j$, y el vector $\mathbf{b}$ se
suma a cada fila (NumPy lo hace automáticamente mediante *broadcasting*). La capa
tiene $n \cdot m + m$ parámetros.

```python
class Densa(Capa):
    def adelante(self, x, entrenando=False):
        self._x = x
        return x @ self.W + self.b

    def atras(self, grad):
        self.dW[...] = self._x.T @ grad
        self.db[...] = grad.sum(axis=0)
        return grad @ self.W.T
```

## Inicialización de los pesos

Si todos los pesos empezaran iguales, todas las neuronas de una capa calcularían lo
mismo y recibirían el mismo gradiente: nunca se diferenciarían. Por eso los pesos se
inicializan **al azar** con una distribución normal de media cero, y los sesgos en
cero. La desviación estándar se elige para que la varianza de las activaciones se
mantenga estable de capa en capa:

$$
\text{He: } W_{ij} \sim \mathcal{N}\!\left(0, \frac{2}{n}\right)
\qquad\qquad
\text{Xavier: } W_{ij} \sim \mathcal{N}\!\left(0, \frac{2}{n + m}\right)
$$

- **He** compensa que ReLU anula la mitad de los valores; se usa con `relu` y
  `leaky_relu`.
- **Xavier** (Glorot) es adecuada para activaciones simétricas como `tanh` y
  `sigmoide`, y para la capa de salida.

## Dropout

`Dropout` es una técnica de regularización. Durante el entrenamiento apaga cada
neurona con probabilidad $p$ (la *tasa*), lo que obliga a la red a no depender de
neuronas concretas. Se usa la variante **invertida**: los valores que sobreviven se
escalan por $1/(1-p)$ para que el valor esperado no cambie.

$$
M_{ij} \sim \text{Bernoulli}(1 - p),
\qquad
\text{salida} = \frac{X \odot M}{1 - p},
\qquad
\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial\,\text{salida}} \odot \frac{M}{1 - p}.
$$

Al evaluar o predecir (`entrenando=False`), la capa no hace nada. Gracias al escalado
invertido, no hace falta corregir nada en ese momento.

# Definición de la red

La función `crear_mlp` construye la red a partir de una lista de tamaños de capas
ocultas, siguiendo siempre el patrón:

$$
\underbrace{[\,\text{Densa} \to \text{activación} \to (\text{Dropout})\,]}_{\text{una vez por capa oculta}} \;\to\; \text{Densa de salida}.
$$

La última capa densa **no tiene activación**: produce valores reales llamados
***logits***. La transformación final (softmax, sigmoide o identidad) la aplica la
función de pérdida, por razones de estabilidad numérica que se explican en la
sección de [pérdidas](#funciones-de-pérdida).

La configuración por defecto de la CLI para las espirales es $2 \to 64 \to 64 \to 3$:

```{.mermaid caption="Red por defecto para las espirales: dos capas ocultas de 64 neuronas y 3 salidas." width="100%"}
flowchart LR
    X["Entrada<br/>N×2"] --> D1["Oculta 1<br/>Densa 2→64 + ReLU<br/>192 parámetros"]
    D1 --> D2["Oculta 2<br/>Densa 64→64 + ReLU<br/>4160 parámetros"]
    D2 --> D3["Salida<br/>Densa 64→3<br/>195 parámetros"]
    D3 -->|"logits"| S["Softmax<br/>probabilidades N×3"]
```

El número total de parámetros es la suma de los de cada capa densa:

$$
(2 \cdot 64 + 64) + (64 \cdot 64 + 64) + (64 \cdot 3 + 3) = 192 + 4160 + 195 = 4547.
$$

A nivel de neuronas, cada neurona de una capa está conectada con **todas** las
neuronas de la capa anterior. La siguiente figura muestra una versión reducida
$2 \to 3 \to 2$ para que las conexiones sean visibles:

```{.mermaid caption="Conexiones completas en una red reducida 2-3-2. Cada flecha es un peso." width="55%"}
flowchart LR
    subgraph Entrada
        x1(("x₁"))
        x2(("x₂"))
    end
    subgraph Oculta["Capa oculta (ReLU)"]
        h1(("h₁"))
        h2(("h₂"))
        h3(("h₃"))
    end
    subgraph Salida["Salida (logits)"]
        y1(("z₁"))
        y2(("z₂"))
    end
    x1 --> h1 & h2 & h3
    x2 --> h1 & h2 & h3
    h1 --> y1 & y2
    h2 --> y1 & y2
    h3 --> y1 & y2
```

# Propagación hacia adelante

Para obtener una predicción, la entrada atraviesa las capas en orden. Para una red
con $L$ capas densas:

$$
\begin{aligned}
A^{(0)} &= X, \\
Z^{(l)} &= A^{(l-1)} W^{(l)} + \mathbf{b}^{(l)}, & l &= 1, \dots, L, \\
A^{(l)} &= \varphi\big(Z^{(l)}\big), & l &= 1, \dots, L-1, \\
\hat{Y} &= g\big(Z^{(L)}\big),
\end{aligned}
$$

donde $g$ es la transformación de salida que aplica la pérdida (softmax en
clasificación multiclase). En el código, esto es un simple recorrido:

```python
def adelante(self, X, entrenando=False):
    for capa in self.capas:
        X = capa.adelante(X, entrenando)
    return X
```

# Funciones de pérdida

La pérdida $\mathcal{L}$ mide qué tan lejos están las predicciones de los valores
correctos. Es el número que el entrenamiento intenta **minimizar**. Cada pérdida de
`perdidas.py` implementa:

- `calcular(salida, y)`: el valor de $\mathcal{L}$, promediado sobre el lote.
- `gradiente(salida, y)`: $\partial \mathcal{L} / \partial Z^{(L)}$, el punto de partida
  de la retropropagación.
- `activar(salida)`: la transformación $g$ que convierte logits en predicciones.

## Error cuadrático medio (regresión)

Para predecir valores continuos, con $k$ salidas por ejemplo:

$$
\mathcal{L}_{\text{ECM}} = \frac{1}{Nk} \sum_{i=1}^{N} \sum_{j=1}^{k} \big(\hat{y}_{ij} - y_{ij}\big)^2,
\qquad
\frac{\partial \mathcal{L}}{\partial \hat{y}_{ij}} = \frac{2}{Nk} \big(\hat{y}_{ij} - y_{ij}\big).
$$

Aquí $g$ es la identidad: la salida de la red es directamente la predicción.

## Entropía cruzada binaria (dos clases)

Con una sola neurona de salida $z$ y objetivo $y \in \{0, 1\}$, la probabilidad
predicha es $p = \sigma(z)$ y la pérdida es

$$
\mathcal{L}_{\text{ECB}} = -\frac{1}{N} \sum_{i=1}^{N} \Big[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \Big].
$$

Calcular primero $p$ y luego $\log p$ puede dar $\log 0$. Por eso se trabaja
directamente con el logit, usando la forma equivalente y estable:

$$
\ell(z, y) = \max(z, 0) - z\,y + \log\!\big(1 + e^{-|z|}\big),
\qquad
\frac{\partial \mathcal{L}}{\partial z_i} = \frac{\sigma(z_i) - y_i}{N}.
$$

## Softmax y entropía cruzada (varias clases)

Para $C$ clases, la capa de salida produce $C$ logits por ejemplo. La función
**softmax** los convierte en probabilidades que suman 1:

$$
p_{ic} = \operatorname{softmax}(\mathbf{z}_i)_c = \frac{e^{z_{ic}}}{\sum_{k=1}^{C} e^{z_{ik}}}.
$$

Si $c_i$ es la clase correcta del ejemplo $i$, la entropía cruzada penaliza que la
probabilidad asignada a esa clase sea baja:

$$
\mathcal{L}_{\text{EC}} = -\frac{1}{N} \sum_{i=1}^{N} \log p_{i c_i}.
$$

Para evitar desbordamientos se calcula el **logaritmo de softmax** restando antes el
máximo de cada fila, lo que no cambia el resultado:

$$
\log p_{ic} = (z_{ic} - m_i) - \log \sum_{k=1}^{C} e^{z_{ik} - m_i},
\qquad
m_i = \max_k z_{ik}.
$$

La gran ventaja de combinar softmax con entropía cruzada es que el gradiente
respecto a los logits es extremadamente simple. Con $\mathbf{y}_i$ el vector *one-hot*
de la clase correcta:

$$
\frac{\partial \mathcal{L}}{\partial z_{ic}} = \frac{p_{ic} - y_{ic}}{N}.
$$

**Derivación.** Para un ejemplo, $\ell = -\log p_{c^*} = -z_{c^*} + \log \sum_k e^{z_k}$.
Derivando respecto a $z_c$: el primer término aporta $-1$ solo si $c = c^*$, y el
segundo aporta $e^{z_c} / \sum_k e^{z_k} = p_c$. Por tanto
$\partial \ell / \partial z_c = p_c - y_c$.

Es decir: **probabilidad predicha menos valor correcto**. Si la red asigna
probabilidad 0.9 a la clase correcta, el gradiente en esa posición es $-0.1$ (empuja a
subirla un poco); si asigna 0.3 a una clase incorrecta, el gradiente es $+0.3$ (empuja a
bajarla).

| Tarea (salidas) | Pérdida | $g$ | $\partial\mathcal{L}/\partial z$ |
| ---------------- | ------------------------------ | ---------- | ------------------ |
| Regresión ($k$) | `ErrorCuadraticoMedio` | identidad | $\tfrac{2}{Nk}(\hat{y} - y)$ |
| Binaria (1) | `EntropiaCruzadaBinaria` | sigmoide | $\tfrac{1}{N}(\sigma(z) - y)$ |
| Multiclase ($C$) | `EntropiaCruzada` | softmax | $\tfrac{1}{N}(p - y)$ |

# Retropropagación

## La regla de la cadena

Para mejorar la red hay que saber cómo cambia la pérdida cuando cambia cada
parámetro, es decir, el **gradiente** $\partial \mathcal{L} / \partial \theta$. La red es
una composición de funciones, así que el gradiente se obtiene con la **regla de la
cadena**. Por ejemplo, para los pesos de la última capa:

$$
\frac{\partial \mathcal{L}}{\partial W^{(L)}}
= \frac{\partial \mathcal{L}}{\partial Z^{(L)}} \cdot \frac{\partial Z^{(L)}}{\partial W^{(L)}}.
$$

Y para una capa anterior, la cadena se alarga atravesando las capas posteriores. La
retropropagación organiza este cálculo de forma eficiente: recorre la red **de la
salida hacia la entrada**, y cada capa recibe el gradiente respecto a su salida,
calcula el de sus parámetros y pasa a la capa anterior el gradiente respecto a su
entrada. Así, cada derivada intermedia se calcula una sola vez.

```{.mermaid caption="Un paso de entrenamiento: propagación hacia adelante, retropropagación y actualización." width="100%"}
sequenceDiagram
    participant R as RedNeuronal
    participant D1 as Densa 1
    participant A as ReLU
    participant D2 as Densa 2
    participant P as Pérdida
    participant O as Optimizador
    R->>D1: adelante(X)
    D1->>A: Z¹ = X·W¹ + b¹
    A->>D2: A¹ = ReLU(Z¹)
    D2->>P: Z² = A¹·W² + b² (logits)
    Note over P: calcula ∂L/∂Z² = (p − y)/N
    P-->>D2: atras(∂L/∂Z²)
    Note over D2: dW² = A¹ᵀ·G · db² = Σ G
    D2-->>A: ∂L/∂A¹ = G·W²ᵀ
    A-->>D1: ∂L/∂Z¹ = ∂L/∂A¹ ⊙ 1[Z¹ > 0]
    Note over D1: dW¹ = Xᵀ·G · db¹ = Σ G
    R->>O: paso(parámetros, gradientes)
    Note over O: θ ← θ − η · actualización
```

## Gradientes de la capa densa

Sea $G = \partial \mathcal{L} / \partial Z$ el gradiente que llega a una capa densa con
$Z = XW + \mathbf{b}$. Como $Z_{ij} = \sum_k X_{ik} W_{kj} + b_j$:

- Cada peso $W_{kj}$ afecta a la columna $j$ de $Z$ en todos los ejemplos, con
  coeficiente $X_{ik}$. Sumando sobre el lote:
  $\dfrac{\partial \mathcal{L}}{\partial W_{kj}} = \sum_i X_{ik}\, G_{ij}$.
- Cada sesgo $b_j$ se suma a la columna $j$ con coeficiente 1:
  $\dfrac{\partial \mathcal{L}}{\partial b_j} = \sum_i G_{ij}$.
- Cada entrada $X_{ik}$ afecta a toda la fila $i$ de $Z$ con coeficientes $W_{kj}$:
  $\dfrac{\partial \mathcal{L}}{\partial X_{ik}} = \sum_j G_{ij} W_{kj}$.

En forma matricial, exactamente lo que implementa `Densa.atras`:

$$
\boxed{\;
\frac{\partial \mathcal{L}}{\partial W} = X^{\top} G,
\qquad
\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \sum_{i=1}^{N} G_{i,:},
\qquad
\frac{\partial \mathcal{L}}{\partial X} = G\, W^{\top}
\;}
$$

Las dimensiones coinciden: $X^{\top} G$ es $(n \times N)(N \times m) = n \times m$, igual
que $W$; y $G W^{\top}$ es $(N \times m)(m \times n) = N \times n$, igual que $X$.

Como las pérdidas ya dividen su gradiente entre $N$, `dW` y `db` son gradientes
**promedio** del lote, y la escala de la tasa de aprendizaje no depende del tamaño
del lote.

## Gradientes de las activaciones

Para una activación elemento a elemento $A = \varphi(Z)$:

$$
\frac{\partial \mathcal{L}}{\partial Z} = \frac{\partial \mathcal{L}}{\partial A} \odot \varphi'(Z),
$$

donde $\odot$ es el producto elemento a elemento. En ReLU esto equivale a dejar pasar
el gradiente donde $Z > 0$ y anularlo en el resto.

## Grafo de cómputo de una capa

```{.mermaid caption="Grafo de cómputo de una capa oculta $l$. Arriba, el cálculo hacia adelante; abajo, el flujo del gradiente, que llega desde las capas siguientes." width="95%"}
flowchart TB
    subgraph ADEL["Hacia adelante"]
        direction LR
        AP["A⁽ˡ⁻¹⁾"] --> ZN["Z⁽ˡ⁾ = A⁽ˡ⁻¹⁾ W⁽ˡ⁾ + b⁽ˡ⁾"]
        ZN --> AN["A⁽ˡ⁾ = φ(Z⁽ˡ⁾)"]
        AN --> LN["capas siguientes<br/>y pérdida L"]
    end
    subgraph ATR["Hacia atrás"]
        direction LR
        GA["∂L/∂A⁽ˡ⁾<br/>(viene de la capa l+1)"] --> GZ["G = ∂L/∂Z⁽ˡ⁾<br/>= ∂L/∂A⁽ˡ⁾ ⊙ φ′(Z⁽ˡ⁾)"]
        GZ --> GW["∂L/∂W⁽ˡ⁾ = A⁽ˡ⁻¹⁾ᵀ G"]
        GZ --> GB["∂L/∂b⁽ˡ⁾ = Σᵢ Gᵢ"]
        GZ --> GP["∂L/∂A⁽ˡ⁻¹⁾ = G W⁽ˡ⁾ᵀ<br/>(pasa a la capa l−1)"]
    end
    ADEL -.->|"retropropagación"| ATR
```

# Optimizadores

Una vez calculados los gradientes, el optimizador actualiza los parámetros. El
gradiente apunta en la dirección en que la pérdida **crece** más rápido, así que se
avanza en sentido contrario.

## Descenso de gradiente estocástico (SGD)

$$
\theta \leftarrow \theta - \eta\, \nabla_\theta \mathcal{L}.
$$

Se llama *estocástico* porque el gradiente se calcula sobre un mini-lote aleatorio y
no sobre todos los datos: es una estimación ruidosa pero barata del gradiente real.
La tasa de aprendizaje $\eta$ controla el tamaño del paso. Si es muy grande, la
pérdida oscila o diverge; si es muy pequeña, el aprendizaje es lento.

## SGD con momento

El momento acumula una "velocidad" $\mathbf{v}$ que promedia los gradientes recientes.
Así se suavizan las oscilaciones y se acelera el avance en direcciones consistentes:

$$
\mathbf{v} \leftarrow \mu\, \mathbf{v} - \eta\, \nabla_\theta \mathcal{L},
\qquad
\theta \leftarrow \theta + \mathbf{v},
$$

con $\mu \in [0, 1)$ el coeficiente de momento (0.9 por defecto en la CLI).

## Adam

Adam (*adaptive moment estimation*) mantiene, para cada parámetro, una media móvil
del gradiente $\mathbf{m}$ y de su cuadrado $\mathbf{v}$. Con $\mathbf{g}_t$ el gradiente en
el paso $t$:

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1\, \mathbf{m}_{t-1} + (1 - \beta_1)\, \mathbf{g}_t,
& \hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1 - \beta_1^{t}}, \\
\mathbf{v}_t &= \beta_2\, \mathbf{v}_{t-1} + (1 - \beta_2)\, \mathbf{g}_t^{2},
& \hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1 - \beta_2^{t}}, \\
\theta_t &= \theta_{t-1} - \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}.
\end{aligned}
$$

- $\hat{\mathbf{m}}_t$ y $\hat{\mathbf{v}}_t$ son las **correcciones de sesgo**: como $\mathbf{m}$
  y $\mathbf{v}$ empiezan en cero, sin corrección serían demasiado pequeños en los
  primeros pasos.
- Dividir por $\sqrt{\hat{\mathbf{v}}_t}$ da a cada parámetro su propio tamaño de paso:
  los parámetros con gradientes grandes o ruidosos avanzan con más cuidado.
- Valores por defecto: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. La CLI usa
  $\eta = 0.01$.

## Regularización L2

Con `decaimiento_peso` $= \lambda > 0$, se suma $\lambda\,\theta$ al gradiente antes de
actualizar. Equivale a añadir $\tfrac{\lambda}{2} \lVert \theta \rVert^2$ a la pérdida, lo
que favorece pesos pequeños y reduce el sobreajuste:

$$
\mathbf{g} \leftarrow \nabla_\theta \mathcal{L} + \lambda\, \theta.
$$

# Proceso de entrenamiento

## Épocas y mini-lotes

- Un **mini-lote** es un subconjunto de `tam_lote` ejemplos (32 por defecto). Con cada
  mini-lote se da **un paso** de optimización.
- Una **época** es una pasada completa por todos los datos de entrenamiento. Al
  inicio de cada época los ejemplos se **barajan**, para que los lotes sean distintos
  y el gradiente no tenga sesgos por el orden de los datos.

Con 480 ejemplos y lotes de 32, cada época tiene $\lceil 480 / 32 \rceil = 15$ pasos.

## Algoritmo

El método `RedNeuronal.entrenar` implementa el siguiente algoritmo:

1. Para cada época $e = 1, \dots, E$:
    a. Barajar los índices de los ejemplos.
    b. Para cada mini-lote $(X_B, Y_B)$:
        i. **Adelante:** $Z^{(L)} = \text{red}(X_B)$, con dropout activo.
        ii. **Gradiente de la pérdida:** $G = \partial \mathcal{L}(Z^{(L)}, Y_B) / \partial Z^{(L)}$.
        iii. **Atrás:** propagar $G$ por las capas en orden inverso, llenando `dW` y `db`.
        iv. **Actualizar:** el optimizador modifica cada $\theta$ con su gradiente.
    c. Evaluar pérdida y precisión en entrenamiento y validación (sin dropout).
    d. Si hay `paciencia`, comprobar la parada temprana.
2. Si hubo parada temprana, restaurar los mejores pesos.
3. Devolver el historial de métricas.

```{.mermaid caption="Flujo completo del entrenamiento, con parada temprana opcional." width="72%"}
flowchart TD
    INI(["Inicio"]) --> EP{"¿Quedan<br/>épocas?"}
    EP -- "sí" --> BAR["Barajar ejemplos"]
    BAR --> LOTE{"¿Quedan<br/>mini-lotes?"}
    LOTE -- "sí" --> ADEL["Propagación hacia adelante<br/>(dropout activo)"]
    ADEL --> GRAD["Gradiente de la pérdida<br/>∂L/∂Z"]
    GRAD --> ATR["Retropropagación<br/>dW, db en cada capa"]
    ATR --> ACT["Optimizador actualiza θ"]
    ACT --> LOTE
    LOTE -- "no" --> EVAL["Evaluar entrenamiento<br/>y validación"]
    EVAL --> MEJ{"¿Mejoró la pérdida<br/>de validación?"}
    MEJ -- "sí" --> GUA["Guardar copia de los pesos<br/>sin_mejora = 0"]
    MEJ -- "no" --> INC["sin_mejora += 1"]
    GUA --> EP
    INC --> PAC{"¿sin_mejora ≥<br/>paciencia?"}
    PAC -- "no" --> EP
    PAC -- "sí" --> RES["Restaurar mejores pesos"]
    EP -- "no" --> RES
    RES --> FIN(["Devolver historial"])
```

El núcleo del bucle en el código es:

```python
for inicio in range(0, len(X), tam_lote):
    lote = indices[inicio : inicio + tam_lote]
    salida = self.adelante(X[lote], entrenando=True)
    self.atras(self.perdida.gradiente(salida, y[lote]))
    self.optimizador.paso(self.parametros())
```

## Parada temprana

Si se indica `paciencia` y hay datos de validación, la red guarda una copia de los
pesos cada vez que la pérdida de validación alcanza un nuevo mínimo. Si pasan
`paciencia` épocas sin mejorar, el entrenamiento se detiene y se restauran los
mejores pesos. Así se evita seguir entrenando cuando la red ya empezó a sobreajustar.

# Ejemplo numérico paso a paso

Para ver el aprendizaje en acción, se sigue un paso de entrenamiento de **una sola
neurona** con activación sigmoide y entropía cruzada binaria, con un único ejemplo.

**Datos iniciales:** $\mathbf{x} = (1,\, 2)$, objetivo $y = 1$, pesos
$\mathbf{w} = (0.5,\, -0.25)$, sesgo $b = 0.1$ y tasa de aprendizaje $\eta = 0.1$.

**1. Propagación hacia adelante.**

$$
z = 0.5 \cdot 1 + (-0.25) \cdot 2 + 0.1 = 0.1,
\qquad
p = \sigma(0.1) = 0.52498.
$$

La neurona asigna una probabilidad de 52.5 % a la clase correcta: casi un volado.

**2. Pérdida.**

$$
\mathcal{L} = -\log(0.52498) = 0.64440.
$$

**3. Retropropagación.** Con softmax/sigmoide y entropía cruzada, el gradiente
respecto al logit es probabilidad menos objetivo:

$$
\frac{\partial \mathcal{L}}{\partial z} = p - y = 0.52498 - 1 = -0.47502.
$$

Por la regla de la cadena, como $z = w_1 x_1 + w_2 x_2 + b$:

$$
\frac{\partial \mathcal{L}}{\partial w_1} = -0.47502 \cdot 1 = -0.47502,
\qquad
\frac{\partial \mathcal{L}}{\partial w_2} = -0.47502 \cdot 2 = -0.95004,
\qquad
\frac{\partial \mathcal{L}}{\partial b} = -0.47502.
$$

Los gradientes son negativos: aumentar estos parámetros **reduce** la pérdida.

**4. Actualización (SGD).**

$$
\begin{aligned}
w_1 &\leftarrow 0.5 - 0.1 \cdot (-0.47502) = 0.54750, \\
w_2 &\leftarrow -0.25 - 0.1 \cdot (-0.95004) = -0.15500, \\
b &\leftarrow 0.1 - 0.1 \cdot (-0.47502) = 0.14750.
\end{aligned}
$$

**5. Comprobación.** Con los nuevos parámetros:

$$
z = 0.5475 - 0.31 + 0.1475 = 0.38501,
\qquad
p = \sigma(0.38501) = 0.59508,
\qquad
\mathcal{L} = 0.51906.
$$

Tras un solo paso, la probabilidad de la clase correcta subió de 52.5 % a 59.5 % y la
pérdida bajó de 0.644 a 0.519. Entrenar una red consiste en repetir este mismo
proceso miles de veces, con miles de parámetros a la vez.

# Evaluación y generalización

## Métricas

- **Pérdida:** el valor de $\mathcal{L}$ sobre un conjunto de datos.
- **Precisión** (*accuracy*), solo en clasificación: fracción de ejemplos cuya clase
  predicha coincide con la correcta. La clase predicha es la de mayor logit, que es
  también la de mayor probabilidad:

$$
\text{precisión} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\!\left[\arg\max_c z_{ic} = c_i\right].
$$

## Entrenamiento, validación y prueba

Una red puede memorizar los datos de entrenamiento sin aprender el patrón general.
Para detectarlo, `datos.dividir` separa los datos (80 % / 20 % por defecto):

- **Entrenamiento:** los únicos datos con los que se calculan gradientes.
- **Validación / prueba:** datos que la red nunca usa para aprender; miden si
  **generaliza**.

Si la pérdida de entrenamiento sigue bajando mientras la de validación sube, la red
está **sobreajustando**. Las herramientas del proyecto para combatirlo son el dropout,
la regularización L2 y la parada temprana.

## Estandarización

Cuando las características tienen escalas muy distintas (por ejemplo, metros y
milímetros), el entrenamiento se vuelve lento e inestable. Con datos CSV, la CLI
aplica el `Estandarizador`, ajustado **solo** con los datos de entrenamiento:

$$
x'_{j} = \frac{x_j - \mu_j}{s_j},
$$

donde $\mu_j$ y $s_j$ son la media y la desviación estándar de la característica $j$.
Esos valores se guardan dentro del modelo para aplicar la misma transformación al
predecir.

# Verificación de los gradientes

Un error en la retropropagación no produce excepciones: la red simplemente aprende
mal. Por eso `tests/test_gradientes.py` compara los gradientes analíticos con una
aproximación numérica por **diferencias centrales**:

$$
\frac{\partial \mathcal{L}}{\partial \theta_k} \approx
\frac{\mathcal{L}(\theta_k + \varepsilon) - \mathcal{L}(\theta_k - \varepsilon)}{2\varepsilon},
\qquad \varepsilon = 10^{-6}.
$$

La prueba construye una red con capas `Densa`, `Tanh`, `LeakyReLU` y `Sigmoide`, y
verifica cada parámetro con las tres funciones de pérdida. Ambos métodos deben
coincidir con un error relativo menor que $10^{-4}$. Cualquier capa nueva puede
validarse de la misma forma.

# Persistencia del modelo

`RedNeuronal.guardar` escribe un archivo `.npz` (formato comprimible de NumPy) que
contiene:

- `arquitectura`: un JSON con el tipo y la configuración (`config()`) de cada capa, el
  nombre de la pérdida y los metadatos (tarea, nombres de clases y parámetros de
  estandarización).
- `p0`, `p1`, ...: los arreglos de pesos y sesgos, en el mismo orden que
  `parametros()`.

`RedNeuronal.cargar` reconstruye las capas a partir del JSON y copia los pesos. El
archivo se lee con `allow_pickle=False`, de modo que cargar un modelo no ejecuta
código arbitrario.

# Resumen

| Concepto | Fórmula clave | Implementación |
| --- | --- | --- |
| Neurona | $a = \varphi(\mathbf{w} \cdot \mathbf{x} + b)$ | Columna de `Densa.W` |
| Capa densa | $Z = XW + \mathbf{b}$ | `Densa.adelante` |
| Activación | $A = \varphi(Z)$ | `activaciones.py` |
| Pérdida | $\mathcal{L} = -\frac{1}{N}\sum_i \log p_{ic_i}$ | `EntropiaCruzada` |
| Gradiente de salida | $\partial\mathcal{L}/\partial Z = (P - Y)/N$ | `Perdida.gradiente` |
| Retropropagación | $\partial\mathcal{L}/\partial W = X^{\top} G$, $\partial\mathcal{L}/\partial X = G W^{\top}$ | `Densa.atras` |
| Actualización | $\theta \leftarrow \theta - \eta\, \mathbf{g}$ | `SGD`, `Adam` |
| Entrenamiento | épocas $\times$ mini-lotes | `RedNeuronal.entrenar` |

En una frase: **la red transforma las entradas capa por capa, mide su error con la
pérdida, calcula con la regla de la cadena cuánto contribuyó cada peso a ese error y
ajusta cada peso un poco en la dirección que lo reduce**, repitiendo el proceso
hasta que las predicciones son buenas también en datos que nunca vio.
