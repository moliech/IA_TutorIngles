import os
import pandas as pd


def generar_informe(
    ruta_csv="data/lecturas_ingles.csv",
    ruta_salida="informe_lecturas.md"
):
    df = pd.read_csv(ruta_csv)

    # Estadísticas principales
    total_lecturas = len(df)
    promedio_palabras = df["conteo_palabras"].mean()
    promedio_dificultad = df["dificultad_lexica"].mean()
    promedio_tiempo = df["tiempo_lectura_est_min"].mean()

    # Distribución por nivel CEFR
    lecturas_cefr = df["nivel_cefr"].value_counts().sort_index()

    # Distribución por categoría
    lecturas_categoria = df["categoria"].value_counts()

    informe = []

    informe.append("# Informe del Analizador CEFR")
    informe.append("")

    informe.append("## Resumen ejecutivo")
    informe.append("")
    informe.append(f"- **Total de lecturas:** {total_lecturas}")
    informe.append(f"- **Promedio de palabras:** {promedio_palabras:.2f}")
    informe.append(
        f"- **Promedio de dificultad léxica:** {promedio_dificultad:.2f}"
    )
    informe.append(
        f"- **Promedio de tiempo de lectura:** {promedio_tiempo:.2f} minutos"
    )
    informe.append("")

    informe.append("## Distribución de lecturas por nivel CEFR")
    informe.append("")
    informe.append("| Nivel CEFR | Cantidad |")
    informe.append("|---|---:|")

    for nivel, cantidad in lecturas_cefr.items():
        informe.append(f"| {nivel} | {cantidad} |")

    informe.append("")

    informe.append("## Distribución por categoría")
    informe.append("")
    informe.append("| Categoría | Cantidad |")
    informe.append("|---|---:|")

    for categoria, cantidad in lecturas_categoria.items():
        informe.append(f"| {categoria} | {cantidad} |")

    informe.append("")

    # Gráficos
    graficos = [
        (
            "Distribución de palabras por nivel CEFR",
            "graficos/distribucion_cefr_palabras.png"
        ),
        (
            "Relación entre palabras y dificultad",
            "graficos/relacion_dificultad_extension.png"
        ),
        (
            "Distribución de tiempos de lectura por categoría",
            "graficos/distribucion_tiempos_categoria.png"
        )
    ]

    informe.append("## Visualizaciones")
    informe.append("")

    for titulo, ruta in graficos:
        if os.path.exists(ruta):
            informe.append(f"### {titulo}")
            informe.append("")
            informe.append(f"![{titulo}]({ruta})")
            informe.append("")

    informe.append("## Conclusiones")
    informe.append("")
    informe.append(
        "El análisis permite observar la distribución de las lecturas "
        "según el nivel CEFR, la relación entre cantidad de palabras y "
        "dificultad léxica, y la distribución de los tiempos estimados "
        "de lectura según la categoría temática."
    )
    informe.append("")

    with open(ruta_salida, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(informe))

    print(f"[OK] Informe generado: {ruta_salida}")


if __name__ == "__main__":
    generar_informe()