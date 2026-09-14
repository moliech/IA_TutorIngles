import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/lecturas_ingles.csv")

promedio_palabras = df.groupby("nivel_cefr")["conteo_palabras"].mean()

plt.figure(figsize=(8,5))
promedio_palabras.plot(kind="bar")
plt.title("Distribucion de palabras por nivel CEFR")
plt.xlabel("Nivel CEFR")
plt.ylabel("Promedio de palabras")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graficos/distribucion_cefr_palabras.png")
plt.close

plt.figure(figsize=(8,5))
plt.scatter(df["conteo_palabras"], df["dificultad_lexica"])
plt.title("Relacion entre palabras y dificultad")
plt.xlabel("Cantidad de palabras")
plt.ylabel("Dificultad lexica")
plt.tight_layout()
plt.savefig("graficos/relacion_dificultad_extension.png")
plt.close()

print("graficos generados correctamente")

