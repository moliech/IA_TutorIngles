"""
src/perceptron_proyecto.py

Implementación desde cero de un Perceptrón con función sigmoide, pérdida logarítmica
(entropía cruzada binaria) y descenso del gradiente utilizando únicamente NumPy.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_LIMPIO = os.path.join(BASE_DIR, "data", "proyecto_limpio.csv")
RUTA_BASE = os.path.join(BASE_DIR, "data", "lecturas_ingles.csv")
CARPETA_GRAFICOS = os.path.join(BASE_DIR, "graficos")


def suma_ponderada(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Calcula la combinación lineal vectorizada z = X @ w + b para todas las filas."""
    return X @ w + b


def sigmoide(z: np.ndarray) -> np.ndarray:
    """Función de activación sigmoide: convierte el puntaje z en probabilidad (0, 1)."""
    return 1.0 / (1.0 + np.exp(-z))


def predecir_probabilidad(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Retorna el vector de probabilidades estimadas por la neurona."""
    return sigmoide(suma_ponderada(X, w, b))


def calcular_perdida(y_real: np.ndarray, y_prob: np.ndarray) -> float:
    """Calcula la pérdida logarítmica (Binary Cross-Entropy)."""
    epsilon = 1e-9
    y_prob_segura = np.clip(y_prob, epsilon, 1.0 - epsilon)
    return float(-np.mean(y_real * np.log(y_prob_segura) + (1.0 - y_real) * np.log(1.0 - y_prob_segura)))


def calcular_exactitud(y_real: np.ndarray, y_prob: np.ndarray) -> float:
    """Calcula la proporción de predicciones correctas usando un umbral de 0.5."""
    predicciones = (y_prob >= 0.5).astype(int)
    return float(np.mean(predicciones == y_real))


def calcular_gradientes(X: np.ndarray, y_real: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, float]:
    """Calcula las derivadas parciales de la pérdida respecto a los pesos (w) y el sesgo (b)."""
    error = y_prob - y_real
    gradiente_w = (X.T @ error) / len(y_real)
    gradiente_b = float(np.mean(error))
    return gradiente_w, gradiente_b


def cargar_y_estandarizar_datos() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga el dataset del proyecto, define la variable objetivo binaria (0/1)
    y aplica estandarización Z-score a las dos variables numéricas de entrada.
    """
    ruta_activa = RUTA_LIMPIO if os.path.exists(RUTA_LIMPIO) else RUTA_BASE
    if not os.path.exists(ruta_activa):
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {ruta_activa}")

    df = pd.read_csv(ruta_activa)

    # Variable objetivo 0/1: 0 = Básico (A1, A2) | 1 = Intermedio-Avanzado (B1, B2)
    df["objetivo_avanzado"] = df["nivel_cefr"].isin(["B1", "B2", "C1", "C2"]).astype(int)

    X = df[["conteo_palabras", "dificultad_lexica"]].to_numpy(dtype=float)
    y = df["objetivo_avanzado"].to_numpy(dtype=int)

    # Estandarización Z-score: x_esc = (x - media) / desviacion
    media = X.mean(axis=0)
    desviacion = X.std(axis=0)
    X_esc = (X - media) / desviacion

    return df, X, X_esc, y, media, desviacion


def exportar_graficos_modelo(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    media: np.ndarray,
    desviacion: np.ndarray,
    historial_perdida: list[float],
    historial_exactitud: list[float],
) -> None:
    """Genera y guarda la curva de aprendizaje y la frontera de decisión en graficos/."""
    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)

    # Gráfico 1: Curva de Aprendizaje
    fig, (eje1, eje2) = plt.subplots(1, 2, figsize=(12, 4.8))
    eje1.plot(range(1, len(historial_perdida) + 1), historial_perdida, color="#EF4444", linewidth=2.2)
    eje1.set_title("Curva de Aprendizaje: Pérdida Logarítmica", fontsize=12, fontweight="bold")
    eje1.set_xlabel("Época")
    eje1.set_ylabel("Pérdida (Binary Cross-Entropy)")
    eje1.grid(True, linestyle=":", alpha=0.5)

    eje2.plot(range(1, len(historial_exactitud) + 1), historial_exactitud, color="#10B981", linewidth=2.2)
    eje2.set_title("Evolución de la Exactitud durante el Entrenamiento", fontsize=12, fontweight="bold")
    eje2.set_xlabel("Época")
    eje2.set_ylabel("Exactitud (0.0 - 1.0)")
    eje2.set_ylim(0.0, 1.05)
    eje2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    ruta_curva = os.path.join(CARPETA_GRAFICOS, "curva_aprendizaje.png")
    plt.savefig(ruta_curva, dpi=300)
    plt.close()
    print(f"[OK] Gráfico de curva de aprendizaje exportado en: {ruta_curva}")

    # Gráfico 2: Frontera de Decisión
    x1_original = np.linspace(X[:, 0].min() - 15, X[:, 0].max() + 15, 100)
    x1_esc = (x1_original - media[0]) / desviacion[0]
    x2_esc = -(w[0] * x1_esc + b) / w[1]
    x2_original = x2_esc * desviacion[1] + media[1]

    plt.figure(figsize=(9, 6))
    plt.scatter(
        X[y == 0, 0], X[y == 0, 1],
        color="#3B82F6", marker="s", s=95, edgecolors="#0F172A", alpha=0.85,
        label="Básico: A1 / A2 (Objetivo = 0)"
    )
    plt.scatter(
        X[y == 1, 0], X[y == 1, 1],
        color="#10B981", marker="o", s=95, edgecolors="#0F172A", alpha=0.85,
        label="Intermedio-Avanzado: B1 / B2 (Objetivo = 1)"
    )
    plt.plot(
        x1_original, x2_original,
        color="#EF4444", linewidth=2.5, linestyle="--",
        label="Frontera de Decisión (Probabilidad = 0.5)"
    )

    plt.ylim(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3)
    plt.title("Frontera de Decisión del Perceptrón (Sigmoide + Descenso del Gradiente)", fontsize=12, fontweight="bold")
    plt.xlabel("Conteo de Palabras de la Lectura (x1)")
    plt.ylabel("Índice de Dificultad Léxica 1.0 - 5.0 (x2)")
    plt.legend(loc="upper left", fontsize=9.5)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    ruta_frontera = os.path.join(CARPETA_GRAFICOS, "frontera_decision.png")
    plt.savefig(ruta_frontera, dpi=300)
    plt.close()
    print(f"[OK] Gráfico de frontera de decisión exportado en: {ruta_frontera}")


def ejecutar_perceptron_proyecto() -> dict:
    """Ejecuta el entrenamiento completo del perceptrón del proyecto y retorna sus métricas."""
    print("\n" + "=" * 72)
    print("  PERCEPTRÓN CON DESCENSO DEL GRADIENTE — IA_TUTORINGLES")
    print("=" * 72)

    df, X, X_esc, y, media, desviacion = cargar_y_estandarizar_datos()

    print(f"Dataset cargado: X tiene forma {X.shape} | y tiene forma {y.shape}")
    print("Variable objetivo (0/1): 0 = Básico (A1/A2) [20 textos] | 1 = Avanzado (B1/B2) [20 textos]")
    print(f"Media original:        palabras = {media[0]:.2f}, dificultad = {media[1]:.2f}")
    print(f"Desviación estándar:   palabras = {desviacion[0]:.2f}, dificultad = {desviacion[1]:.2f}")
    print(f"Verificación Z-score:  media_esc = {np.round(X_esc.mean(axis=0), 4)}, std_esc = {np.round(X_esc.std(axis=0), 4)}\n")

    w = np.zeros(2, dtype=float)
    b = 0.0
    tasa_aprendizaje = 0.1
    epocas = 500
    historial_perdida = []
    historial_exactitud = []

    y_prob_inicial = predecir_probabilidad(X_esc, w, b)
    perdida_inicial = calcular_perdida(y, y_prob_inicial)

    for epoca in range(1, epocas + 1):
        y_prob = predecir_probabilidad(X_esc, w, b)
        gradiente_w, gradiente_b = calcular_gradientes(X_esc, y, y_prob)

        w = w - tasa_aprendizaje * gradiente_w
        b = b - tasa_aprendizaje * gradiente_b

        perdida_actual = calcular_perdida(y, y_prob)
        exactitud_actual = calcular_exactitud(y, y_prob)
        historial_perdida.append(perdida_actual)
        historial_exactitud.append(exactitud_actual)

        if epoca == 1 or epoca % 100 == 0:
            print(
                f"Época {epoca:>3} | Pérdida: {perdida_actual:.4f} | "
                f"Exactitud: {exactitud_actual:.2%} | w=[{w[0]:.4f}, {w[1]:.4f}] b={b:.4f}"
            )

    y_prob_final = predecir_probabilidad(X_esc, w, b)
    perdida_final = calcular_perdida(y, y_prob_final)
    exactitud_final = calcular_exactitud(y, y_prob_final)

    print("\n" + "-" * 72)
    print("  RESUMEN DE RESULTADOS DEL MODELO")
    print("-" * 72)
    print(f"• Pesos finales:    w1 (palabras) = {w[0]:.4f} | w2 (dificultad) = {w[1]:.4f} | sesgo b = {b:.4f}")
    print(f"• Pérdida Log-Loss: Inicial = {perdida_inicial:.4f}  --->  Final = {perdida_final:.4f}")
    print(f"• Exactitud final:  {exactitud_final:.2%} ({int(np.sum((y_prob_final >= 0.5) == y))}/{len(y)} lecturas clasificadas)")
    print("-" * 72 + "\n")

    exportar_graficos_modelo(X, y, w, b, media, desviacion, historial_perdida, historial_exactitud)

    print("\n--- [PRUEBA DE PREDICCIÓN CON 3 LECTURAS NUEVAS] ---")
    lecturas_nuevas = np.array([
        [115.0, 1.6],
        [450.0, 4.6],
        [265.0, 3.3],
    ])
    lecturas_nuevas_esc = (lecturas_nuevas - media) / desviacion
    probabilidades_nuevas = predecir_probabilidad(lecturas_nuevas_esc, w, b)

    for (palabras, dificultad), prob in zip(lecturas_nuevas, probabilidades_nuevas):
        nivel_pred = "AVANZADO (B1/B2 -> 1)" if prob >= 0.5 else "BÁSICO (A1/A2 -> 0)"
        print(f"Palabras: {palabras:>5.0f} | Dificultad: {dificultad:>3.1f} -> Probabilidad: {prob:>6.2%} -> {nivel_pred}")

    return {
        "w1": float(w[0]),
        "w2": float(w[1]),
        "b": float(b),
        "perdida_inicial": perdida_inicial,
        "perdida_final": perdida_final,
        "exactitud": exactitud_final * 100.0,
    }


if __name__ == "__main__":
    ejecutar_perceptron_proyecto()