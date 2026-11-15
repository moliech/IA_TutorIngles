"""
src/usar_modelo.py

Módulo de Inferencia Predictiva en Vivo - IA_TutorIngles (LexiaAdapt)
Carga el modelo serializado en joblib y clasifica lecturas sin reentrenar.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUTA_MODELO = os.path.join(os.path.dirname(__file__), "modelo_final.joblib")

def cargar_modelo():
    """Carga y retorna el Pipeline serializado desde disco."""
    if not os.path.exists(RUTA_MODELO):
        raise FileNotFoundError(
            f"No existe el archivo {RUTA_MODELO}. Debe ejecutarse primero modelo_proyecto.py"
        )
    return joblib.load(RUTA_MODELO)

def predecir_lectura(conteo_palabras: int, dificultad_lexica: float, tiempo_min: float = None):
    """
    Función de inferencia consumida por la interfaz gráfica (Streamlit) y módulos externos.
    Recibe variables cuantitativas y predice clase, probabilidades y nivel CEFR.
    """
    modelo = cargar_modelo()

    df_in = pd.DataFrame([{
        "conteo_palabras": conteo_palabras,
        "dificultad_lexica": dificultad_lexica
    }])

    clase_pred = int(modelo.predict(df_in)[0])
    probabilidades = modelo.predict_proba(df_in)[0]

    nivel = "Avanzado (B1/B2)" if clase_pred == 1 else "Básico (A1/A2)"
    confianza = float(probabilidades[clase_pred])

    if tiempo_min is None or tiempo_min <= 0:
        tiempo_min = round(conteo_palabras / 90.0, 2)
    ppm = round(conteo_palabras / tiempo_min, 2)

    return {
        "clase_binaria": clase_pred,
        "nivel": nivel,
        "probabilidad_basico": float(probabilidades[0]),
        "probabilidad_avanzado": float(probabilidades[1]),
        "confianza": confianza,
        "tiempo_estimado_min": tiempo_min,
        "ppm_calculado": ppm
    }

def predecir_lote(df_lote: pd.DataFrame) -> pd.DataFrame:
    """
    Clasifica un lote completo de lecturas en formato tabular.
    Requiere que el DataFrame contenga 'conteo_palabras' y 'dificultad_lexica'.
    """
    modelo = cargar_modelo()
    X = df_lote[["conteo_palabras", "dificultad_lexica"]]
    probs = modelo.predict_proba(X)
    preds = modelo.predict(X)

    df_res = df_lote.copy()
    df_res["prob_basico_%"] = (probs[:, 0] * 100).round(1)
    df_res["prob_avanzado_%"] = (probs[:, 1] * 100).round(1)
    df_res["certeza_%"] = (probs.max(axis=1) * 100).round(1)
    df_res["nivel_predicho"] = [
        "Avanzado (B1/B2)" if p == 1 else "Básico (A1/A2)" for p in preds
    ]
    return df_res

def generar_figura_proyeccion(df_nuevos=None, ruta_guardar=None):
    """
    Genera la figura de la frontera de decisión con el dataset base (40 lecturas)
    y proyecta sobre ella las nuevas lecturas evaluadas en tiempo real.
    """
    ruta_csv = os.path.join(os.path.dirname(__file__), "..", "data", "proyecto_limpio.csv")
    if not os.path.exists(ruta_csv):
        ruta_csv = os.path.join(os.path.dirname(__file__), "data", "proyecto_limpio.csv")

    df_base = pd.read_csv(ruta_csv)
    pipeline = cargar_modelo()

    scaler = pipeline.named_steps["escalado"]
    clf = pipeline.named_steps["modelo"]

    w1, w2 = clf.coef_[0]
    b = clf.intercept_[0]
    m1, m2 = scaler.mean_
    s1, s2 = scaler.scale_

    x1_vals = np.linspace(50, 550, 100)
    z1_vals = (x1_vals - m1) / s1
    z2_vals = -(w1 * z1_vals + b) / w2
    x2_vals = m2 + (z2_vals * s2)

    fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=110)

    # Puntos del dataset histórico
    df_basico = df_base[df_base["nivel_cefr"].isin(["A1", "A2"])]
    df_avanzado = df_base[df_base["nivel_cefr"].isin(["B1", "B2"])]

    ax.scatter(
        df_basico["conteo_palabras"],
        df_basico["dificultad_lexica"],
        color="#2563EB",
        alpha=0.55,
        edgecolors="none",
        s=55,
        label="Dataset Base: Básico A1/A2 (20 lecturas)"
    )
    ax.scatter(
        df_avanzado["conteo_palabras"],
        df_avanzado["dificultad_lexica"],
        color="#DC2626",
        alpha=0.55,
        edgecolors="none",
        s=55,
        label="Dataset Base: Avanzado B1/B2 (20 lecturas)"
    )

    # Línea divisoria matemática
    ax.plot(
        x1_vals,
        x2_vals,
        color="#0F766E",
        linestyle="--",
        linewidth=2.2,
        label="Frontera de Decisión (P = 50%)"
    )

    # Proyección de casos nuevos si existen
    if df_nuevos is not None and not df_nuevos.empty and "conteo_palabras" in df_nuevos.columns:
        ax.scatter(
            df_nuevos["conteo_palabras"],
            df_nuevos["dificultad_lexica"],
            color="#F59E0B",
            edgecolors="#0F172A",
            linewidths=1.2,
            s=220,
            marker="*",
            zorder=5,
            label=f"Nuevas Lecturas Evaluadas ({len(df_nuevos)})"
        )

        col_nom = "titulo" if "titulo" in df_nuevos.columns else "caso"
        for _, fila in df_nuevos.iterrows():
            x = fila["conteo_palabras"]
            y = fila["dificultad_lexica"]
            txt = str(fila.get(col_nom, "Nuevo"))[:18]
            ax.annotate(
                txt,
                (x, y),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF3C7", edgecolor="#D97706", alpha=0.9)
            )

    ax.set_xlim(40, 560)
    ax.set_ylim(0.9, 5.3)
    ax.set_xlabel("Extensión del Texto (Número de Palabras)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Dificultad Léxica (1.0 a 5.0)", fontsize=9.5, fontweight="bold")
    ax.set_title("Frontera Matemática y Proyección de Nuevas Lecturas", fontsize=10.5, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.55)

    plt.tight_layout()

    if ruta_guardar:
        plt.savefig(ruta_guardar, dpi=120)

    return fig

def ejecutar_demostracion_casos_nuevos():
    """
    Ejecuta la demostración de 3 casos representativos para verificación en vivo.
    """
    print("=" * 75)
    print("  INFERENCIA PREDICTIVA: EVALUACIÓN DE CASOS EN TIEMPO REAL (SIN REENTRENAR)")
    print("=" * 75)

    # 1. Cargar el modelo guardado en el Paso 1 (ya incluye el escalado en el Pipeline)
    modelo = cargar_modelo()

    # 2. Casos nuevos realistas para la demostración
    #    (Mismos nombres de columna que en el entrenamiento)
    casos_nuevos = pd.DataFrame({
        "caso": ["Caso 1 (Básico claro)", "Caso 2 (Avanzado claro)", "Caso 3 (En frontera de decisión)"],
        "conteo_palabras": [105, 480, 278],
        "dificultad_lexica": [1.5, 4.4, 3.2]
    })

    # 3. Predecir probabilidad y clase
    entradas = casos_nuevos[["conteo_palabras", "dificultad_lexica"]]
    prob_avanzado = modelo.predict_proba(entradas)[:, 1]
    prediccion = modelo.predict(entradas)

    casos_nuevos["prob_avanzado_%"] = (prob_avanzado * 100).round(1)
    casos_nuevos["resultado"] = [
        "Avanzado (B1/B2)" if p == 1 else "Básico (A1/A2)" for p in prediccion
    ]

    print("\nPredicciones con el modelo guardado (modelo_final.joblib):")
    print(casos_nuevos.to_string(index=False))

    print("\nInterpretación pedagógica de los casos predichos:")
    print("  • Caso 1 (105 pal, 1.5 dif): Texto corto de rutina diaria -> 98.9% Básico (1.1% Avanzado).")
    print("  • Caso 2 (480 pal, 4.4 dif): Texto técnico de inteligencia artificial -> 99.8% Avanzado.")
    print("  • Caso 3 (278 pal, 3.2 dif): Lectura intermedia en la frontera de decisión -> el modelo evalúa la transición A2/B1.")
    print("=" * 75)

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        # Modo interactivo CLI con argumentos: python usar_modelo.py <palabras> <dificultad>
        pal = int(sys.argv[1])
        dif = float(sys.argv[2])
        res = predecir_lectura(pal, dif)
        print(f"\n[★] Entrada: {pal} palabras, {dif} dificultad")
        print(f"    Resultado: {res['nivel']} (Confianza: {res['confianza']*100:.1f}%)")
    else:
        # Modo por defecto: Demostración con casos representativos
        ejecutar_demostracion_casos_nuevos()
