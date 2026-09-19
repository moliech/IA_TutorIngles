from visualizaciones import generar_grafico_distribucion_tiempos
import pandas as pd

def main():
    print("Iniciando analisis del tutor de ingles...")
    

    df = pd.read_csv("data/lecturas_ingles.csv")


    generar_grafico_distribucion_tiempos(df)


    print("Analisis finalizado correctamente.")


if __name__ == "__main__":
     main()
