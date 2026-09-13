import numpy as np

def calcular_estadisticas_numpy(array_numerico):
    """
    Recibe el array 2D de NumPy y calcula métricas descriptivas
    para conteo_palabras (columna 0) y dificultad_lexica (columna 1).
    """
    if array_numerico is None or len(array_numerico) == 0:
        return {}

    palabras = array_numerico[:, 0]
    dificultad = array_numerico[:, 1]
    tiempo = array_numerico[:, 2]

    estadisticas = {
        "palabras": {
            "media": float(np.mean(palabras)),
            "mediana": float(np.median(palabras)),
            "desviacion_estandar": float(np.std(palabras)),
            "minimo": float(np.min(palabras)),
            "maximo": float(np.max(palabras))
        },
        "dificultad": {
            "media": float(np.mean(dificultad)),
            "mediana": float(np.median(dificultad)),
            "desviacion_estandar": float(np.std(dificultad)),
            "minimo": float(np.min(dificultad)),
            "maximo": float(np.max(dificultad))
        },
        "tiempo": {
            "media": float(np.mean(tiempo)),
            "mediana": float(np.median(tiempo)),
            "desviacion_estandar": float(np.std(tiempo)),
            "minimo": float(np.min(tiempo)),
            "maximo": float(np.max(tiempo))
        }
    }

    return estadisticas