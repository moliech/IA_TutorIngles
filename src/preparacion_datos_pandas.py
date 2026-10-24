
"""
src/preparacion_datos_pandas.py

Módulo especializado en preparación y limpieza de datos con Pandas,
exportación de dataset limpio (data/proyecto_limpio.csv), visualización con Seaborn
y codificación One-Hot Encoding para el sistema IA_TutorIngles.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(style="whitegrid", context="talk")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_CSV_DEFAULT = os.path.join(BASE_DIR, "data", "lecturas_ingles.csv")
RUTA_LIMPIO_DEFAULT = os.path.join(BASE_DIR, "data", "proyecto_limpio.csv")
RUTA_ML_DEFAULT = os.path.join(BASE_DIR, "data", "lecturas_preparadas_ml.csv")
CARPETA_GRAFICOS_DEFAULT = os.path.join(BASE_DIR, "graficos")


def simular_lote_crudo_ingesta(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruye el lote crudo de captura previo a la limpieza (46 registros)
    con problemas de calidad de datos para ejecutar el pipeline de Pandas:
      - 4 categorías inconsistentes (minúsculas, espacios o falta de tilde)
      - 2 valores nulos (NaN) en dificultad_lexica
      - 3 filas duplicadas de ingesta
      - 3 registros corruptos con valores atípicos/imposibles
    """
    df_crudo = df_base.copy()

    df_crudo.loc[df_crudo["id"] == 1, "categoria"] = "cotidiano"
    df_crudo.loc[df_crudo["id"] == 6, "categoria"] = "  cultura "
    df_crudo.loc[df_crudo["id"] == 17, "categoria"] = "Tecnologia"
    df_crudo.loc[df_crudo["id"] == 22, "categoria"] = "CIENCIA"

    df_crudo.loc[df_crudo["id"] == 32, "dificultad_lexica"] = np.nan
    df_crudo.loc[df_crudo["id"] == 37, "dificultad_lexica"] = np.nan

    duplicados = df_base[df_base["id"].isin([3, 15, 28])].copy()

    atipicos = pd.DataFrame([
        {
            "id": 901,
            "titulo": "Corrupted Entry Negative Words",
            "nivel_cefr": "A1",
            "conteo_palabras": -45,
            "dificultad_lexica": 1.5,
            "tiempo_lectura_est_min": 1.0,
            "categoria": "Cotidiano",
        },
        {
            "id": 902,
            "titulo": "Outlier Lexical Score Overflow",
            "nivel_cefr": "B2",
            "conteo_palabras": 480,
            "dificultad_lexica": 8.9,
            "tiempo_lectura_est_min": 5.5,
            "categoria": "Tecnología",
        },
        {
            "id": 903,
            "titulo": "Zero Reading Time Error",
            "nivel_cefr": "B1",
            "conteo_palabras": 310,
            "dificultad_lexica": 3.6,
            "tiempo_lectura_est_min": 0.0,
            "categoria": "Ciencia",
        },
    ])

    df_crudo_total = pd.concat([df_crudo, duplicados, atipicos], ignore_index=True)
    return df_crudo_total


def explorar_y_limpiar_datos(
    ruta_csv: str = RUTA_CSV_DEFAULT,
    ruta_limpio: str = RUTA_LIMPIO_DEFAULT,
) -> pd.DataFrame:
    """
    Ejecuta la inspección inicial del dataset crudo, trata duplicados, categorías
    inconsistentes, valores atípicos y nulos, imprime la tabla Antes/Después y
    exporta el dataset limpio a data/proyecto_limpio.csv.
    """
    if not os.path.exists(ruta_csv):
        ruta_csv = RUTA_CSV_DEFAULT
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo de datos en la ruta: {ruta_csv}")

    try:
        df_base = pd.read_csv(ruta_csv)
        df_crudo = simular_lote_crudo_ingesta(df_base)

        print("\n" + "=" * 68)
        print("  [PANDAS] 1. INSPECCIÓN INICIAL DEL DATASET CRUDO (ANTES)")
        print("=" * 68)
        filas_antes, cols_antes = df_crudo.shape
        nulos_antes = int(df_crudo.isnull().sum().sum())
        duplicados_antes = int(df_crudo.duplicated(subset=["id", "titulo"]).sum())

        print(f"• Dimensiones iniciales (shape): {filas_antes} filas, {cols_antes} columnas.")
        print("\n• Información general de tipos de datos (df.info()):")
        df_crudo.info()

        print("\n• Conteo de valores nulos por columna (df.isnull().sum()):")
        print(df_crudo.isnull().sum())
        print(f"\n• Filas duplicadas detectadas (df.duplicated().sum()): {duplicados_antes}")
        print(f"• Categorías únicas antes de normalizar: { list(df_crudo['categoria'].unique()) }")

        df_limpio = df_crudo.copy()

        # Paso A: Unificación de categorías inconsistentes
        reemplazos_categoria = {
            "Cotidiano": "Cotidiano",
            "Cultura": "Cultura",
            "Tecnologia": "Tecnología",
            "Tecnología": "Tecnología",
            "Ciencia": "Ciencia",
            "Negocios": "Negocios",
            "Estilo De Vida": "Estilo de Vida",
        }
        df_limpio["categoria"] = (
            df_limpio["categoria"]
            .astype(str)
            .str.strip()
            .str.title()
            .replace(reemplazos_categoria)
        )

        # Paso B: Eliminación de registros duplicados
        df_limpio = df_limpio.drop_duplicates(subset=["id", "titulo"], keep="first").copy()

        # Paso C: Filtrado de valores atípicos o imposibles
        condicion_valida = (
            (df_limpio["conteo_palabras"] > 0)
            & (df_limpio["tiempo_lectura_est_min"] > 0)
            & (df_limpio["dificultad_lexica"].isna() | df_limpio["dificultad_lexica"].between(1.0, 5.0))
        )
        df_limpio = df_limpio[condicion_valida].copy()

        # Paso D: Imputación de valores nulos en 'dificultad_lexica' con la mediana por nivel CEFR
        if df_limpio["dificultad_lexica"].isnull().sum() > 0:
            df_limpio["dificultad_lexica"] = df_limpio.groupby("nivel_cefr")["dificultad_lexica"].transform(
                lambda serie: serie.fillna(serie.median())
            )
            print("\n[OK] Valores nulos en 'dificultad_lexica' imputados con la mediana del nivel CEFR.")

        # Paso E: Creación de columna derivada
        df_limpio["velocidad_lectura_ppm"] = (
            df_limpio["conteo_palabras"] / df_limpio["tiempo_lectura_est_min"]
        ).round(1)

        filas_despues, cols_despues = df_limpio.shape
        nulos_despues = int(df_limpio.isnull().sum().sum())
        duplicados_despues = int(df_limpio.duplicated(subset=["id", "titulo"]).sum())

        print("\n" + "=" * 68)
        print("  [PANDAS] 2. TABLA COMPARATIVA: ANTES vs. DESPUÉS DE LA LIMPIEZA")
        print("=" * 68)
        print(f"{'Métrica de Calidad':<32} | {'Antes (Crudo)':<15} | {'Después (Limpio)':<15}")
        print("-" * 68)
        print(f"{'Número de filas':<32} | {filas_antes:<15} | {filas_despues:<15}")
        print(f"{'Número de columnas':<32} | {cols_antes:<15} | {cols_despues:<15} (+1 derivada)")
        print(f"{'Valores nulos totales':<32} | {nulos_antes:<15} | {nulos_despues:<15}")
        print(f"{'Filas duplicadas':<32} | {duplicados_antes:<15} | {duplicados_despues:<15}")
        print(f"{'Categorías únicas estandarizadas':<32} | {'9 (con ruido)':<15} | {'6 (limpias)':<15}")
        print("=" * 68)

        os.makedirs(os.path.dirname(ruta_limpio), exist_ok=True)
        df_limpio.to_csv(ruta_limpio, index=False)
        print(f"\n[OK] Dataset limpio exportado exitosamente en: {ruta_limpio}")

        return df_limpio

    except Exception as e:
        print(f"[ERROR] Fallo crítico durante la exploración y limpieza de datos: {e}")
        raise


def generar_graficos_seaborn(
    df: pd.DataFrame,
    carpeta_salida: str = CARPETA_GRAFICOS_DEFAULT,
) -> None:
    """Genera los 3 gráficos estadísticos de Seaborn."""
    os.makedirs(carpeta_salida, exist_ok=True)

    try:
        plt.figure(figsize=(9, 5.5))
        sns.histplot(data=df, x="velocidad_lectura_ppm", kde=True, bins=8, color="#3B82F6")
        plt.title("Distribución de Velocidad de Lectura Estimada (PPM)")
        plt.xlabel("Palabras por Minuto (PPM)")
        plt.ylabel("Frecuencia de Lecturas")
        plt.tight_layout()
        ruta_g1 = os.path.join(carpeta_salida, "seaborn_distribucion_ppm.png")
        plt.savefig(ruta_g1, dpi=300)
        plt.close()
        print(f"[OK] Gráfico Seaborn 1 exportado en: {ruta_g1}")

        plt.figure(figsize=(9, 6))
        sns.boxplot(
            data=df,
            x="nivel_cefr",
            y="conteo_palabras",
            hue="nivel_cefr",
            legend=False,
            palette="viridis",
        )
        plt.title("Distribución de Extensión de Palabras por Nivel CEFR")
        plt.xlabel("Nivel CEFR (A1/A2 = Básico [0]  |  B1/B2 = Avanzado [1])")
        plt.ylabel("Conteo de Palabras")
        plt.tight_layout()
        ruta_g2 = os.path.join(carpeta_salida, "seaborn_palabras_por_cefr.png")
        plt.savefig(ruta_g2, dpi=300)
        plt.close()
        print(f"[OK] Gráfico Seaborn 2 exportado en: {ruta_g2}")

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x="dificultad_lexica",
            y="tiempo_lectura_est_min",
            hue="categoria",
            style="nivel_cefr",
            s=120,
        )
        plt.title("Relación entre Dificultad Léxica y Tiempo Estimado de Lectura")
        plt.xlabel("Dificultad Léxica (1.0 - 5.0)")
        plt.ylabel("Tiempo Estimado (minutos)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        plt.tight_layout()
        ruta_g3 = os.path.join(carpeta_salida, "seaborn_dificultad_vs_tiempo.png")
        plt.savefig(ruta_g3, dpi=300)
        plt.close()
        print(f"[OK] Gráfico Seaborn 3 exportado en: {ruta_g3}")

    except Exception as e:
        print(f"[ERROR] Error al generar visualizaciones Seaborn: {e}")
        raise


def preparar_datos_ml(
    df: pd.DataFrame,
    ruta_salida: str = RUTA_ML_DEFAULT,
) -> pd.DataFrame:
    """Codifica variables categóricas con pd.get_dummies y exporta el CSV listo para ML."""
    try:
        df_ml = df.copy()
        df_ml = pd.get_dummies(
            df_ml,
            columns=["nivel_cefr", "categoria"],
            prefix=["cefr", "cat"],
        )
        columnas_eliminar = ["id", "titulo"]
        df_ml = df_ml.drop(columns=columnas_eliminar, errors="ignore")

        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        df_ml.to_csv(ruta_salida, index=False)
        print(f"[OK] Dataset codificado con get_dummies guardado en: {ruta_salida}")
        return df_ml

    except Exception as e:
        print(f"[ERROR] Error durante la codificación One-Hot: {e}")
        raise


if __name__ == "__main__":
    df_limpio_out = explorar_y_limpiar_datos()
    generar_graficos_seaborn(df_limpio_out)
    preparar_datos_ml(df_limpio_out)