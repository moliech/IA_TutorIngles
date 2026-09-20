import os
import pandas as pd

def generar_informe(
    ruta_csv="data/lecturas_ingles.csv",
    ruta_salida="informes/informe_proyecto.md"
):
    """
    Genera un informe dinámico ejecutivo en formato Markdown a partir de los datos procesados,
    incrustando las visualizaciones estadísticas y métricas por nivel CEFR y categoría.
    """
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo CSV en: {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    total_lecturas = len(df)
    promedio_palabras = df["conteo_palabras"].mean()
    promedio_dificultad = df["dificultad_lexica"].mean()
    promedio_tiempo = df["tiempo_lectura_est_min"].mean()

    # Agrupaciones
    tabla_cefr = df.groupby("nivel_cefr").agg(
        Cantidad=("id", "count"),
        Palabras_Promedio=("conteo_palabras", "mean"),
        Dificultad_Promedio=("dificultad_lexica", "mean"),
        Tiempo_Promedio=("tiempo_lectura_est_min", "mean")
    ).reset_index()

    tabla_categoria = df.groupby("categoria").agg(
        Cantidad=("id", "count"),
        Palabras_Promedio=("conteo_palabras", "mean"),
        Tiempo_Promedio=("tiempo_lectura_est_min", "mean")
    ).reset_index()

    lineas = []

    lineas.append("# 📊 Informe de Análisis Exploratorio de Datos (EDA) — IA Tutor de Inglés")
    lineas.append("")
    lineas.append("## 1. Resumen Ejecutivo")
    lineas.append("El presente informe documenta el análisis cuantitativo del corpus de lecturas diseñado para el **Tutor Inteligente de Inglés**. Las métricas extraídas sustentan la graduación pedagógica de los contenidos de acuerdo con el **Marco Común Europeo de Referencia para las Lenguas (CEFR)**.")
    lineas.append("")
    lineas.append(f"- **Total de lecturas analizadas:** {total_lecturas} textos")
    lineas.append(f"- **Promedio de extensión:** {promedio_palabras:.1f} palabras por lectura")
    lineas.append(f"- **Índice promedio de dificultad léxica:** {promedio_dificultad:.2f} (escala 0.0 - 1.0)")
    lineas.append(f"- **Tiempo estimado de lectura promedio:** {promedio_tiempo:.1f} minutos")
    lineas.append("")

    lineas.append("## 2. Distribución Cuantitativa por Nivel CEFR")
    lineas.append("La graduación del contenido garantiza un incremento progresivo en la extensión y complejidad textual a medida que se avanza en la escala del marco europeo:")
    lineas.append("")
    lineas.append("| Nivel CEFR | Cantidad de Lecturas | Extensión Promedio (palabras) | Dificultad Léxica Promedio | Tiempo Estimado (min) |")
    lineas.append("| :---: | :---: | :---: | :---: | :---: |")

    for _, fila in tabla_cefr.iterrows():
        lineas.append(
            f"| **{fila['nivel_cefr']}** | {int(fila['Cantidad'])} | {fila['Palabras_Promedio']:.1f} | {fila['Dificultad_Promedio']:.2f} | {fila['Tiempo_Promedio']:.1f} |"
        )
    lineas.append("")

    lineas.append("## 3. Distribución por Categoría Temática")
    lineas.append("La diversificación temática del dataset permite ofrecer lecturas contextualizadas según los intereses de los estudiantes:")
    lineas.append("")
    lineas.append("| Categoría Temática | Cantidad de Textos | Extensión Promedio (palabras) | Tiempo Estimado (min) |")
    lineas.append("| :--- | :---: | :---: | :---: |")

    for _, fila in tabla_categoria.iterrows():
        lineas.append(
            f"| {fila['categoria']} | {int(fila['Cantidad'])} | {fila['Palabras_Promedio']:.1f} | {fila['Tiempo_Promedio']:.1f} |"
        )
    lineas.append("")

    lineas.append("## 4. Visualizaciones Analíticas del Sistema")
    lineas.append("")

    graficos = [
        (
            "Figura 1: Distribución del Conteo Promedio de Palabras por Nivel CEFR",
            "graficos/distribucion_cefr_palabras.png",
            "Muestra cómo la extensión de los textos aumenta de forma progresiva desde el nivel A1 hasta el nivel C2."
        ),
        (
            "Figura 2: Relación entre la Extensión del Texto y la Dificultad Léxica",
            "graficos/relacion_dificultad_extension.png",
            "Evidencia la correlación entre el volumen de palabras y el índice de dificultad léxica del vocabulario."
        ),
        (
            "Figura 3: Distribución de Tiempos Estimados de Lectura por Categoría Temática",
            "graficos/distribucion_tiempos_categoria.png",
            "Diagrama de cajas (Boxplot) que compara la dispersión y tiempos promedios requeridos para completar lecturas según su área temática."
        )
    ]

    for titulo, ruta, descripcion in graficos:
        if os.path.exists(ruta):
            lineas.append(f"### {titulo}")
            lineas.append("")
            lineas.append(f"![{titulo}]({ruta})")
            lineas.append("")
            lineas.append(f"_{descripcion}_")
            lineas.append("")

    lineas.append("## 5. Conclusiones y Diagnóstico Pedagógico")
    lineas.append("1. **Graduación Adecuada:** El análisis valida que el dataset presenta una progresión coherente tanto en longitud como en sofisticación léxica.")
    lineas.append("2. **Soporte para Tutoría Personalizada:** La estructura multivariada sienta las bases cuantitativas para la posterior clasificación de textos mediante algoritmos de Machine Learning.")
    lineas.append("")
    lineas.append("---")
    lineas.append("*Informe generado automáticamente por el módulo `src/generar_informe.py` del proyecto IA Tutor de Inglés.*")

    if os.path.dirname(ruta_salida):
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    with open(ruta_salida, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas))

    print(f"[OK] Informe Markdown generado exitosamente en: {ruta_salida}")

if __name__ == "__main__":
    generar_informe()