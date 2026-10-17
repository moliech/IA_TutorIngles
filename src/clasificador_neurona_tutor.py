"""
src/clasificador_neurona_tutor.py
Aplicación de la Neurona Artificial al proyecto IA_TutorIngles.
Entrena un Perceptrón para clasificar lecturas en Básico (0) vs Avanzado (1)
en función del conteo de palabras y dificultad léxica normalizadas.
Autor: Jhon Esteban Molina Echavarría / IA_TutorIngles
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuronas_compuertas import Perceptron


def cargar_y_preparar_dataset(ruta_csv: str = "data/lecturas_ingles.csv"):
    """
    Carga el dataset de lecturas, normaliza características continuas
    y genera la etiqueta binaria de clasificación.
    """
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el dataset en: {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    # 1. Definir Target Binario: A1/A2 = 0 (Básico), B1/B2 = 1 (Avanzado)
    df["etiqueta_binaria"] = df["nivel_cefr"].apply(lambda n: 0 if n in ["A1", "A2"] else 1)

    # 2. Normalización Min-Max de Características Continuas
    palabras_min, palabras_max = df["conteo_palabras"].min(), df["conteo_palabras"].max()
    dif_min, dif_max = df["dificultad_lexica"].min(), df["dificultad_lexica"].max()

    df["palabras_norm"] = (df["conteo_palabras"] - palabras_min) / (palabras_max - palabras_min)
    df["dificultad_norm"] = (df["dificultad_lexica"] - dif_min) / (dif_max - dif_min)

    X = df[["palabras_norm", "dificultad_norm"]].to_numpy()
    y = df["etiqueta_binaria"].to_numpy()

    return df, X, y


def entrenar_clasificador_tutor(X: np.ndarray, y: np.ndarray):
    """Instancia y entrena el perceptrón sobre los datos del tutor."""
    clasificador = Perceptron(
        num_entradas=2,
        tasa_aprendizaje=0.05,
        epocas_max=100,
        nombre="Perceptron-TutorLectura"
    )
    convergio = clasificador.entrenar(X, y)

    # Evaluación de Exactitud (Accuracy)
    predicciones = clasificador.predecir(X)
    aciertos = np.sum(predicciones == y)
    exactitud = (aciertos / len(y)) * 100.0

    print(f"\n[EVALUACIÓN] Exactitud del clasificador neuronal: {exactitud:.1f}% ({aciertos}/{len(y)} textos clasificados)")
    return clasificador, exactitud


def generar_grafico_frontera_tutor(df: pd.DataFrame, clasificador: Perceptron, ruta_salida: str = "graficos/frontera_lecturas_tutor.png"):
    """
    Genera un diagrama de dispersión con la frontera de decisión lineal
    que divide las lecturas Básicas de las Avanzadas.
    """
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    plt.figure(figsize=(9, 6))

    # Puntos según etiqueta
    basicos = df[df["etiqueta_binaria"] == 0]
    avanzados = df[df["etiqueta_binaria"] == 1]

    plt.scatter(basicos["palabras_norm"], basicos["dificultad_norm"], color="#3B82F6", label="Básico (A1 / A2)", s=90, edgecolors="#1E293B", alpha=0.85)
    plt.scatter(avanzados["palabras_norm"], avanzados["dificultad_norm"], color="#10B981", label="Avanzado (B1 / B2)", s=90, edgecolors="#1E293B", alpha=0.85)

    # Línea de separación w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b)/w2
    x_vals = np.linspace(0, 1, 100)
    if abs(clasificador.pesos[1]) > 1e-6:
        y_vals = -(clasificador.pesos[0] * x_vals + clasificador.sesgo) / clasificador.pesos[1]
        plt.plot(x_vals, y_vals, color="#EF4444", linewidth=2.5, linestyle="--", label="Frontera Neuronal de Decisión")

    plt.title("Clasificación Adaptativa de Lecturas con Perceptrón Simple", fontsize=13, fontweight="bold", color="#1E293B")
    plt.xlabel("Conteo de Palabras (Normalizado 0 - 1)", fontsize=11)
    plt.ylabel("Dificultad Léxica (Normalizada 0 - 1)", fontsize=11)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()
    print(f"[OK] Gráfico de clasificación del tutor exportado en: {ruta_salida}")


def ejecutar_clasificador_tutor(ruta_csv: str = "data/lecturas_ingles.csv"):
    """Función de orquestación del clasificador de lecturas."""
    print("\n" + "=" * 60)
    print("  CLASIFICADOR NEURONAL ADAPTATIVO — IA TUTOR DE INGLÉS   ")
    print("=" * 60)
    df, X, y = cargar_y_preparar_dataset(ruta_csv)
    clasificador, exactitud = entrenar_clasificador_tutor(X, y)
    generar_grafico_frontera_tutor(df, clasificador)
    return clasificador, exactitud


if __name__ == "__main__":
    ejecutar_clasificador_tutor()
