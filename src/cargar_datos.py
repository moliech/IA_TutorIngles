import csv
import numpy as np
import os

def cargar_lecturas(ruta_csv="data/lecturas_ingles.csv"):
    """
    Carga el dataset CSV y retorna:
    - datos_completos: Lista de diccionarios con toda la información.
    - array_numerico: Array 2D de NumPy con [conteo_palabras, dificultad_lexica, tiempo_lectura_est_min].
    """
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo de datos en la ruta: {ruta_csv}")

    datos_completos = []
    matriz_numerica = []

    with open(ruta_csv, mode='r', encoding='utf-8') as archivo:
        lector = csv.DictReader(archivo)
        for fila in lector:
            registro = {
                "id": int(fila["id"]),
                "titulo": fila["titulo"],
                "nivel_cefr": fila["nivel_cefr"],
                "conteo_palabras": float(fila["conteo_palabras"]),
                "dificultad_lexica": float(fila["dificultad_lexica"]),
                "tiempo_lectura_est_min": float(fila["tiempo_lectura_est_min"]),
                "categoria": fila["categoria"]
            }
            datos_completos.append(registro)
            matriz_numerica.append([
                registro["conteo_palabras"],
                registro["dificultad_lexica"],
                registro["tiempo_lectura_est_min"]
            ])

    array_numerico = np.array(matriz_numerica, dtype=float)
    return datos_completos, array_numerico