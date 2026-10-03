"""
src/preparacion_datos_pandas.py

Módulo especializado en exploración con Pandas, visualización con Seaborn
y codificación One-Hot Encoding para el sistema IA_TutorIngles.
"""

import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(style="whitegrid", context="talk")


def explorar_y_limpiar_datos(
    ruta_csv: str = "data/lecturas_ingles.csv"
) -> pd.DataFrame:

    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en la ruta: {ruta_csv}"
        )

    try:
        df = pd.read_csv(ruta_csv)

        print("\n--- [PANDAS] Exploración Inicial del Dataset ---")

        print(
            f"Dimensiones del DataFrame: "
            f"{df.shape[0]} filas, {df.shape[1]} columnas."
        )

        print("\nInformación General de Variables:")
        print(df.info())

        print("\nEstadísticas Descriptivas:")
        print(df.describe())

        print("\nValores Nulos por Columna:")
        print(df.isnull().sum())

        df_limpio = df.copy()

        if df_limpio.isnull().sum().sum() > 0:

            mediana_dif = float(
                df_limpio["dificultad_lexica"].median()
            )

            df_limpio["dificultad_lexica"] = (
                df_limpio["dificultad_lexica"]
                .fillna(mediana_dif)
            )

            print(
                f"\n[INFO] Imputación realizada en "
                f"'dificultad_lexica' con la mediana: {mediana_dif}"
            )

        df_limpio["velocidad_lectura_ppm"] = (
            df_limpio["conteo_palabras"]
            / df_limpio["tiempo_lectura_est_min"]
        ).round(1)

        print(
            "\n[OK] Columna derivada "
            "'velocidad_lectura_ppm' generada exitosamente."
        )

        return df_limpio

    except Exception as e:

        print(
            f"[ERROR] Fallo crítico durante la exploración de datos: {e}"
        )

        raise


def generar_graficos_seaborn(
    df: pd.DataFrame,
    carpeta_salida: str = "graficos/"
) -> None:

    os.makedirs(carpeta_salida, exist_ok=True)

    try:

        plt.figure(figsize=(9, 5.5))

        sns.histplot(
            data=df,
            x="velocidad_lectura_ppm",
            kde=True,
            bins=8
        )

        plt.title(
            "Distribución de Velocidad de Lectura Estimada (PPM)"
        )

        plt.xlabel(
            "Palabras por Minuto (PPM)"
        )

        plt.ylabel(
            "Frecuencia de Lecturas"
        )

        plt.tight_layout()

        ruta_g1 = os.path.join(
            carpeta_salida,
            "seaborn_distribucion_ppm.png"
        )

        plt.savefig(
            ruta_g1,
            dpi=300
        )

        plt.close()

        print(
            f"[OK] Gráfico Seaborn 1 exportado en: {ruta_g1}"
        )

        plt.figure(figsize=(9, 6))

        sns.boxplot(
            data=df,
            x="nivel_cefr",
            y="conteo_palabras",
            hue="nivel_cefr",
            legend=False,
            palette="viridis"
        )

        plt.title(
            "Distribución de Extensión de Palabras por Nivel CEFR"
        )

        plt.xlabel(
            "Nivel CEFR"
        )

        plt.ylabel(
            "Conteo de Palabras"
        )

        plt.tight_layout()

        ruta_g2 = os.path.join(
            carpeta_salida,
            "seaborn_palabras_por_cefr.png"
        )

        plt.savefig(
            ruta_g2,
            dpi=300
        )

        plt.close()

        print(
            f"[OK] Gráfico Seaborn 2 exportado en: {ruta_g2}"
        )

        plt.figure(figsize=(10, 6))

        sns.scatterplot(
            data=df,
            x="dificultad_lexica",
            y="tiempo_lectura_est_min",
            hue="categoria",
            style="nivel_cefr",
            s=120
        )

        plt.title(
            "Relación entre Dificultad Léxica y Tiempo Estimado de Lectura"
        )

        plt.xlabel(
            "Dificultad Léxica (1.0 - 5.0)"
        )

        plt.ylabel(
            "Tiempo Estimado (minutos)"
        )

        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=9
        )

        plt.tight_layout()

        ruta_g3 = os.path.join(
            carpeta_salida,
            "seaborn_dificultad_vs_tiempo.png"
        )

        plt.savefig(
            ruta_g3,
            dpi=300
        )

        plt.close()

        print(
            f"[OK] Gráfico Seaborn 3 exportado en: {ruta_g3}"
        )

    except Exception as e:

        print(
            f"[ERROR] Error al generar visualizaciones Seaborn: {e}"
        )

        raise


def preparar_datos_ml(
    df: pd.DataFrame,
    ruta_salida: str = "data/lecturas_preparadas_ml.csv"
) -> pd.DataFrame:

    try:

        df_ml = df.copy()

        df_ml = pd.get_dummies(
            df_ml,
            columns=[
                "nivel_cefr",
                "categoria"
            ],
            prefix=[
                "cefr",
                "cat"
            ]
        )

        columnas_eliminar = [
            "id",
            "titulo"
        ]

        df_ml = df_ml.drop(
            columns=columnas_eliminar,
            errors="ignore"
        )

        os.makedirs(
            os.path.dirname(ruta_salida),
            exist_ok=True
        )

        df_ml.to_csv(
            ruta_salida,
            index=False
        )

        print(
            f"\n[OK] Dataset preparado para ML "
            f"guardado exitosamente en: {ruta_salida}"
        )

        return df_ml

    except Exception as e:

        print(
            f"[ERROR] Error durante la codificación One-Hot: {e}"
        )

        raise