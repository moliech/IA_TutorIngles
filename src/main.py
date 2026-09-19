import pandas as pd

from visualizaciones import generar_grafico_distribucion_tiempos
from generar_informe import generar_informe


def main():
    print("Iniciando analisis del tutor de ingles...")

    df = pd.read_csv("data/lecturas_ingles.csv")

    generar_grafico_distribucion_tiempos(df)

    generar_informe(
        ruta_csv="data/lecturas_ingles.csv",
        ruta_salida="informe_proyecto.md"
    )

    print("Analisis finalizado correctamente.")


if __name__ == "__main__":
    main()