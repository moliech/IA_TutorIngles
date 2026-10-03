import os
import pandas as pd
import matplotlib.pyplot as  plt

df = pd.read_csv("data/lecturas_ingles.csv")

promedio_palabras = df.groupby("nivel_cefr")["conteo_palabras"].mean()

plt.figure(figsize=(8,5))
promedio_palabras.plot(kind="bar")
plt.title("Distribucion de palabras por nivel CEFR")
plt.xlabel("Nivel CEFR")
plt.ylabel("Promedio de palabras")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graficos/distribucion_cefr_palabras.png", dpi=300)
plt.close()

plt.figure(figsize=(8,5))
plt.scatter(df["conteo_palabras"], df["dificultad_lexica"])
plt.title("Relacion entre palabras y dificultad")
plt.xlabel("Cantidad de palabras")
plt.ylabel("Dificultad lexica")
plt.tight_layout()
plt.savefig("graficos/relacion_dificultad_extension.png", dpi=300)
plt.close()

def generar_grafico_distribucion_tiempos(
    df,
    ruta_salida="graficos/distribucion_tiempos_categoria.png"):

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    categorias =df["categoria"].unique()

    datos_por_cat = [
        df[df["categoria"] ==cat]["tiempo_lectura_est_min"]
        for cat in categorias
    ]
   
    plt.figure(figsize=(10,6))

    bp = plt.boxplot(
        datos_por_cat,
        patch_artist=True,
        tick_labels=categorias
    )

    colores = [
         "#2B6CB0",
         "#4299E1",
         "#63B3ED",
         "#90CDF4",
         "#CBD5E0"

    ]

    for patch, color in zip(bp["boxes"], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)


    plt.title(

        "3.er  Grafico PLUS: Distribucion de tiempos de lectura por categoria"
    )


    plt.xlabel("categoria tematica")
    plt.ylabel("tiempo estimado de lectura (min)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)


    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()
    print(f"[OK] Grafico 3 PLUS guardado en: {ruta_salida}")





