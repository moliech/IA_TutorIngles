import pandas as pd
from cargar_datos import cargar_lecturas
from eda_numpy import calcular_estadisticas_numpy
from visualizaciones import generar_todas_las_visualizaciones
from generar_informe import generar_informe
from preparacion_datos_pandas import (
    explorar_y_limpiar_datos,
    generar_graficos_seaborn,
    preparar_datos_ml
)

def main():
    print("=========================================================")
    print("      SISTEMA IA TUTOR DE INGLÉS — EJECUCIÓN PRINCIPAL   ")
    print("=========================================================")
    
    ruta_csv = "data/lecturas_ingles.csv"
    
    print("\n[INFO] 1. Cargando dataset de lecturas...")
    datos_completos, array_numerico = cargar_lecturas(ruta_csv)
    df = pd.DataFrame(datos_completos)
    print(f"       -> Se cargaron correctamente {len(df)} lecturas con 7 variables.")

    print("\n[INFO] 2. Calculando estadísticas exploratorias (EDA con NumPy)...")
    estadisticas = calcular_estadisticas_numpy(array_numerico)
    print(f"       -> Promedio general de palabras: {estadisticas['palabras']['media']:.1f}")
    print(f"       -> Promedio dificultad léxica: {estadisticas['dificultad']['media']:.2f}")

    print("\n[INFO] 3. Generando visualizaciones estadísticas (Matplotlib)...")
    generar_todas_las_visualizaciones(df)

    generar_informe(
        ruta_csv="data/lecturas_ingles.csv",
        ruta_salida="informe_proyecto.md"
    )
    print(
        "\n[INFO] 5. Ejecutando análisis con Pandas, "
        "Seaborn y One-Hot Encoding (Clase 8)..."
    )

    df_limpio = explorar_y_limpiar_datos(
        "data/lecturas_ingles.csv"
    )

    generar_graficos_seaborn(df_limpio)

    df_ml = preparar_datos_ml(df_limpio)
    print("Analisis finalizado correctamente.")

    print("\n=========================================================")
    print("  [ÉXITO] Proceso finalizado correctamente sin errores.  ")
    print("=========================================================\n")

if __name__ == "__main__":
    main()