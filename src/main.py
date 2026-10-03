import pandas as pd

from visualizaciones import generar_grafico_distribucion_tiempos
from generar_informe import generar_informe
from preparacion_datos_pandas import (
    explorar_y_limpiar_datos,
    generar_graficos_seaborn,
    preparar_datos_ml
)

def main():
    print("Iniciando analisis del tutor de ingles...")

    df = pd.read_csv("data/lecturas_ingles.csv")

    generar_grafico_distribucion_tiempos(df)

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


if __name__ == "__main__":
    main()