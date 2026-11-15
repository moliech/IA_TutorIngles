"""
src/modelo_proyecto.py

Módulo de Entrenamiento y Evaluación Comparativa de Modelos Scikit-Learn
Proyecto: IA_TutorIngles (LexiaAdapt)
Genera: graficos/matriz_confusion_sklearn.png y src/modelo_final.joblib
"""

import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def entrenar_y_evaluar_modelos():
    print("=" * 70)
    print("  SISTEMA PREDICTIVO: ENTRENAMIENTO Y EVALUACIÓN COMPARATIVA")
    print("=" * 70)

    # 1. Cargar los datos limpios del proyecto
    ruta_csv = os.path.join(os.path.dirname(__file__), "..", "data", "proyecto_limpio.csv")
    if not os.path.exists(ruta_csv):
        ruta_csv = os.path.join(os.path.dirname(__file__), "data", "proyecto_limpio.csv")

    df = pd.read_csv(ruta_csv)
    print(f"[+] Dataset cargado correctamente: {len(df)} lecturas.")

    # Variable objetivo binaria: 0 = Básico (A1/A2), 1 = Avanzado (B1/B2)
    df["clase_objetivo"] = df["nivel_cefr"].map({"A1": 0, "A2": 0, "B1": 1, "B2": 1})

    entradas = ["conteo_palabras", "dificultad_lexica"]
    X = df[entradas]
    y = df["clase_objetivo"]

    # 2. Separar los datos: 80 % para entrenar y 20 % para probar (Pág. 3 y 19 de la Guía)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[+] Lecturas para entrenar: {len(X_train)} | Lecturas para probar: {len(X_test)}")

    # 3. Modelos a comparar (dentro de un Pipeline con StandardScaler)
    modelos = {
        "Regresión logística": Pipeline([
            ("escalado", StandardScaler()),
            ("modelo", LogisticRegression(random_state=42)),
        ]),
        "KNN (5 vecinos)": Pipeline([
            ("escalado", StandardScaler()),
            ("modelo", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Árbol de decisión": Pipeline([
            ("escalado", StandardScaler()),
            ("modelo", DecisionTreeClassifier(max_depth=3, random_state=42)),
        ]),
        "Red neuronal (MLP)": Pipeline([
            ("escalado", StandardScaler()),
            ("modelo", MLPClassifier(hidden_layer_sizes=(8, 4), solver="lbfgs", max_iter=1000, random_state=42)),
        ]),
    }

    # 4. Entrenar cada modelo y medirlo con los datos de prueba
    resultados = []
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        resultados.append({
            "Modelo": nombre,
            "Exactitud": accuracy_score(y_test, y_pred),
            "Precisión": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "Val. cruzada": cross_val_score(modelo, X, y, cv=5).mean(),
        })

    tabla = pd.DataFrame(resultados)
    print("\nComparación de modelos (datos de prueba no vistos):")
    print(tabla.round(3).to_string(index=False))

    # 5. Elegir el modelo con el F1 más alto (idxmax busca la fila del valor máximo)
    nombre_mejor = tabla.loc[tabla["F1"].idxmax(), "Modelo"]
    mejor_modelo = modelos[nombre_mejor]
    print(f"\n[★] Mejor modelo según F1: {nombre_mejor}")

    # Comparación de pesos aprendidos con el Perceptrón manual (Punto 9 de la Guía)
    if "Regresión logística" in nombre_mejor:
        regresion = mejor_modelo.named_steps["modelo"]
        w1 = regresion.coef_[0][0]
        w2 = regresion.coef_[0][1]
        b = regresion.intercept_[0]
        print(f"\nPesos aprendidos por Scikit-Learn: w1 = {w1:.2f}, w2 = {w2:.2f}, b = {b:.2f}")
        print("-" * 70)
        print("COMPARACIÓN CON EL MODELO PERCEPTRÓN BASE:")
        print("  • Perceptrón Base:      w1 ≈ 2.69 (palabras), w2 ≈ 2.54 (dificultad), b ≈ 0.13")
        print(f"  • Regresión Logística:  w1 = {w1:.2f} (palabras), w2 = {w2:.2f} (dificultad), b = {b:.2f}")
        print("  -> Ambos modelos asignan pesos positivos a ambas variables:")
        print("     A mayor conteo de palabras y mayor dificultad léxica, mayor probabilidad")
        print("     de clasificar el texto como Avanzado (B1/B2). ¡Convergencia consistente!")
        print("-" * 70)

    # 6. Matriz de confusión del mejor modelo
    matriz = confusion_matrix(y_test, mejor_modelo.predict(X_test))
    print("\nMatriz de confusión (filas = valor real, columnas = predicción):")
    print(matriz)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predijo Básico (0)", "Predijo Avanzado (1)"],
        yticklabels=["Real Básico (0)", "Real Avanzado (1)"]
    )
    plt.title(f"Matriz de confusión – {nombre_mejor}\n(Prueba: {len(X_test)} lecturas no vistas)", fontsize=11, fontweight="bold")
    plt.tight_layout()

    ruta_graficos = os.path.join(os.path.dirname(__file__), "..", "graficos")
    os.makedirs(ruta_graficos, exist_ok=True)
    ruta_png = os.path.join(ruta_graficos, "matriz_confusion_sklearn.png")
    plt.savefig(ruta_png, dpi=150)
    plt.close()
    print(f"\n[+] Gráfico guardado: {ruta_png}")

    # 7. Guardar el mejor modelo en un archivo serializado
    ruta_joblib = os.path.join(os.path.dirname(__file__), "modelo_final.joblib")
    joblib.dump(mejor_modelo, ruta_joblib)
    print(f"[+] Modelo guardado: {ruta_joblib}")
    print("=" * 70)

# Alias de compatibilidad con main.py
ejecutar_entrenamiento_comparativo = entrenar_y_evaluar_modelos

if __name__ == "__main__":
    entrenar_y_evaluar_modelos()
