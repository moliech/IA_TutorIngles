
import os
import pandas as pd
from cargar_datos import cargar_lecturas
from eda_numpy import calcular_estadisticas_numpy
from visualizaciones import generar_todas_las_visualizaciones
from generar_informe import generar_informe
from preparacion_datos_pandas import (
    explorar_y_limpiar_datos,
    generar_graficos_seaborn,
    preparar_datos_ml,
)
from neuronas_compuertas import ejecutar_laboratorio_compuertas
from clasificador_neurona_tutor import ejecutar_clasificador_tutor
from perceptron_proyecto import ejecutar_perceptron_proyecto

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=========================================================")
    print("      SISTEMA IA TUTOR DE INGLÉS — EJECUCIÓN PRINCIPAL   ")
    print("=========================================================")

    ruta_csv = os.path.join(BASE_DIR, "data", "lecturas_ingles.csv")
    ruta_informe = os.path.join(BASE_DIR, "informes", "informe_proyecto.md")

    print("\n[INFO] 1. Cargando dataset base de lecturas...")
    datos_completos, array_numerico = cargar_lecturas(ruta_csv)
    df = pd.DataFrame(datos_completos)
    print(f"       -> Se cargaron correctamente {len(df)} lecturas con 7 variables.")

    print("\n[INFO] 2. Calculando estadísticas exploratorias (EDA con NumPy)...")
    estadisticas = calcular_estadisticas_numpy(array_numerico)
    print(f"       -> Promedio general de palabras: {estadisticas['palabras']['media']:.1f}")
    print(f"       -> Promedio dificultad léxica: {estadisticas['dificultad']['media']:.2f}")

    print("\n[INFO] 3. Generando visualizaciones estadísticas (Matplotlib)...")
    generar_todas_las_visualizaciones(df)

    print("\n[INFO] 4. Generando informe ejecutivo dinámico en Markdown...")
    generar_informe(ruta_csv=ruta_csv, ruta_salida=ruta_informe)

    print("\n[INFO] 5. Ejecutando preparación de datos con Pandas, Seaborn y One-Hot Encoding...")
    df_limpio = explorar_y_limpiar_datos(ruta_csv)
    generar_graficos_seaborn(df_limpio)
    preparar_datos_ml(df_limpio)

    print("\n[INFO] 6. Ejecutando Laboratorio de Compuertas Lógicas...")
    ejecutar_laboratorio_compuertas()

    print("\n[INFO] 7. Ejecutando Clasificador con Función Escalón...")
    ejecutar_clasificador_tutor(ruta_csv)

    print("\n[INFO] 8. Ejecutando Perceptrón con Sigmoide y Descenso del Gradiente...")
    ejecutar_perceptron_proyecto()

    print("\n=========================================================")
    print("  [ÉXITO] Proceso finalizado correctamente sin errores.  ")
    print("=========================================================\n")


if __name__ == "__main__":
    main()