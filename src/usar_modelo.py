"""
Módulo de Inferencia Predictiva - IA_TutorIngles
Requisito 6 de los Lineamientos del Proyecto Final (Clase 12)
Carga el modelo serializado en joblib y clasifica lecturas nuevas sin reentrenar.
"""

import os
import sys
import joblib
import pandas as pd

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
    Predice el nivel de una lectura.
    Retorna diccionario con clase, probabilidad y nivel pedagógico.
    """
    if tiempo_min is None or tiempo_min <= 0:
        tiempo_min = round(conteo_palabras / 90.0, 2)
    ppm = round(conteo_palabras / tiempo_min, 2)

    X_nuevo = pd.DataFrame([{
        "conteo_palabras": conteo_palabras,
        "dificultad_lexica": dificultad_lexica,
        "tiempo_lectura_est_min": tiempo_min,
        "velocidad_lectura_ppm": ppm
    }])

    modelo = cargar_modelo()
    clase_pred = modelo.predict(X_nuevo)[0]
    probabilidades = modelo.predict_proba(X_nuevo)[0]

    nivel = "Avanzado (B1/B2)" if clase_pred == 1 else "Básico (A1/A2)"
    confianza = probabilidades[clase_pred]

    return {
        "clase_binaria": int(clase_pred),
        "nivel": nivel,
        "probabilidad_basico": float(probabilidades[0]),
        "probabilidad_avanzado": float(probabilidades[1]),
        "confianza": float(confianza),
        "tiempo_estimado_min": tiempo_min,
        "ppm_calculado": ppm
    }

if __name__ == "__main__":
    print("=" * 65)
    print("   DEMOSTRACIÓN DE INFERENCIA EN VIVO (usar_modelo.py)")
    print("=" * 65)

    palabras = int(sys.argv[1]) if len(sys.argv) > 1 else 320
    dificultad = float(sys.argv[2]) if len(sys.argv) > 2 else 3.4

    print(f"[+] Caso de entrada: {palabras} palabras | Dificultad Léxica: {dificultad}/5.0")
    try:
        resultado = predecir_lectura(palabras, dificultad)
        print(f"\n[★] Predicción del Sistema: {resultado['nivel']}")
        print(f"    - Confianza del Modelo:      {resultado['confianza'] * 100:.2f}%")
        print(f"    - Probabilidad Nivel Básico:   {resultado['probabilidad_basico'] * 100:.2f}%")
        print(f"    - Probabilidad Nivel Avanzado: {resultado['probabilidad_avanzado'] * 100:.2f}%")
        print(f"    - Velocidad estimada:         {resultado['ppm_calculado']} PPM")
    except FileNotFoundError as e:
        print(f"\n[!] Advertencia: {e}")
        print("    Para generar el artefacto entrenado, ejecute 'python src/modelo_proyecto.py'.")
    print("=" * 65)
