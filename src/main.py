import pandas as pd
from cargar_datos import cargar_lecturas
from eda_numpy import calcular_estadisticas_numpy
from visualizaciones import generar_todas_las_visualizaciones
from generar_informe import generar_informe

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

    print("\n[INFO] 4. Generando informe ejecutivo dinámico en Markdown...")
    generar_informe(ruta_csv=ruta_csv, ruta_salida="informe_proyecto.md")

    print("\n=========================================================")
    print("  [ÉXITO] Proceso finalizado correctamente sin errores.  ")
    print("=========================================================\n")

if __name__ == "__main__":
    main()