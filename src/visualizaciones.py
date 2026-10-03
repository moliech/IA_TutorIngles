import os
import pandas as pd
import matplotlib.pyplot as plt

def generar_grafico_cefr_palabras(df, ruta_salida="graficos/distribucion_cefr_palabras.png"):
    """
    Genera un gráfico de barras con el promedio de palabras por nivel CEFR.
    """
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    
    promedio_palabras = df.groupby("nivel_cefr")["conteo_palabras"].mean()
    niveles_ordenados = ["A1", "A2", "B1", "B2", "C1", "C2"]
    promedio_palabras = promedio_palabras.reindex([n for n in niveles_ordenados if n in promedio_palabras.index])
    
    plt.figure(figsize=(9, 5.5))
    barras = plt.bar(promedio_palabras.index, promedio_palabras.values, color='#2B6CB0', alpha=0.85, width=0.55)
    
    for barra in barras:
        yval = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2.0, yval + 5, f'{yval:.0f} pal.', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1A365D')
        
    plt.title("Distribución del Conteo Promedio de Palabras por Nivel CEFR", fontsize=13, fontweight='bold', pad=14, color='#1A365D')
    plt.xlabel("Nivel CEFR (Marco Común Europeo)", fontsize=11, labelpad=8, color='#2D3748')
    plt.ylabel("Promedio de Palabras por Texto", fontsize=11, labelpad=8, color='#2D3748')
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.ylim(0, max(promedio_palabras.values) * 1.15)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()
    print(f"[OK] Gráfico 1 guardado en: {ruta_salida}")

def generar_grafico_dificultad_extension(df, ruta_salida="graficos/relacion_dificultad_extension.png"):
    """
    Genera un scatter plot que analiza la relación entre el conteo de palabras y la dificultad léxica.
    """
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    
    plt.figure(figsize=(9, 5.5))
    plt.scatter(df["conteo_palabras"], df["dificultad_lexica"], color='#2B6CB0', s=70, alpha=0.75, edgecolors='#1A365D', linewidth=1)
    
    plt.title("Relación entre la Extensión del Texto y la Dificultad Léxica", fontsize=13, fontweight='bold', pad=14, color='#1A365D')
    plt.xlabel("Cantidad de Palabras por Lectura", fontsize=11, labelpad=8, color='#2D3748')
    plt.ylabel("Índice de Dificultad Léxica (0.0 - 1.0)", fontsize=11, labelpad=8, color='#2D3748')
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()
    print(f"[OK] Gráfico 2 guardado en: {ruta_salida}")

def generar_grafico_distribucion_tiempos(df, ruta_salida="graficos/distribucion_tiempos_categoria.png"):
    """
    Genera un diagrama de cajas (Boxplot) de los tiempos estimados de lectura agrupados por categoría temática.
    """
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    categorias = df["categoria"].unique()
    datos_por_cat = [df[df["categoria"] == cat]["tiempo_lectura_est_min"] for cat in categorias]

    plt.figure(figsize=(10, 6))
    try:
        bp = plt.boxplot(datos_por_cat, patch_artist=True, tick_labels=categorias)
    except TypeError:
        bp = plt.boxplot(datos_por_cat, patch_artist=True, labels=categorias)

    colores = ["#2B6CB0", "#4299E1", "#63B3ED", "#90CDF4", "#CBD5E0"]
    for patch, color in zip(bp["boxes"], colores[:len(categorias)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    plt.title("Distribución de Tiempos Estimados de Lectura por Categoría Temática", fontsize=13, fontweight='bold', pad=14, color='#1A365D')
    plt.xlabel("Categoría Temática", fontsize=11, labelpad=8, color='#2D3748')
    plt.ylabel("Tiempo Estimado de Lectura (minutos)", fontsize=11, labelpad=8, color='#2D3748')
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()
    print(f"[OK] Gráfico 3 guardado en: {ruta_salida}")

def generar_todas_las_visualizaciones(df):
    """
    Invoca secuencialmente la generación de los tres gráficos estadísticos.
    """
    generar_grafico_cefr_palabras(df)
    generar_grafico_dificultad_extension(df)
    generar_grafico_distribucion_tiempos(df)

if __name__ == "__main__":
    df_test = pd.read_csv("data/lecturas_ingles.csv")
    generar_todas_las_visualizaciones(df_test)
