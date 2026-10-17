"""
src/neuronas_compuertas.py
Módulo de Redes Neuronales Artificiales: Perceptrón Simple para Compuertas Lógicas.
Implementa AND, OR, NOT, NAND y demuestra experimentalmente el fallo de XOR (Clase 8).
Autor: Heiber Lozano Mercado / IA_TutorIngles
"""

import os
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    """
    Implementación del Perceptrón Simple de Rosenblatt con función escalón.
    """
    def __init__(self, num_entradas: int = 2, tasa_aprendizaje: float = 0.1, epocas_max: int = 50, nombre: str = "Neurona"):
        self.num_entradas = num_entradas
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas_max = epocas_max
        self.nombre = nombre
        self.pesos = np.zeros(num_entradas, dtype=float)
        self.sesgo = 0.0
        self.historial_errores = []

    def activacion_escalon(self, z: np.ndarray | float) -> np.ndarray | int:
        """Función de activación escalón unitario: 1 si z >= 0, sino 0."""
        return np.where(np.round(z, 10) >= 0, 1, 0)

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Calcula la salida de la neurona: Theta(w · x + b)."""
        z = np.dot(X, self.pesos) + self.sesgo
        return self.activacion_escalon(z)

    def entrenar(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Aplica la regla de aprendizaje del perceptrón de Rosenblatt.
        Retorna True si clasificó correctamente todas las entradas; False si alcanzó epocas_max sin converger.
        """
        self.pesos = np.zeros(self.num_entradas, dtype=float)
        self.sesgo = 0.0
        self.historial_errores = []

        for epoca in range(1, self.epocas_max + 1):
            errores_epoca = 0
            for xi, yi in zip(X, y):
                prediccion = self.predecir(xi)
                error = yi - prediccion
                if error != 0:
                    self.pesos = np.round(self.pesos + self.tasa_aprendizaje * error * xi, 10)
                    self.sesgo = float(np.round(self.sesgo + self.tasa_aprendizaje * error, 10))
                    errores_epoca += abs(error)

            self.historial_errores.append(errores_epoca)
            if errores_epoca == 0 and np.array_equal(self.predecir(X), y):
                print(f"[{self.nombre}] Clasificó correctamente todas las entradas en la época {epoca}. Pesos: {self.pesos}, Sesgo: {self.sesgo:.2f}")
                return True

        print(f"[{self.nombre}] NO convergió tras {self.epocas_max} épocas. Error final: {errores_epoca}")
        return False


def obtener_datos_compuertas():
    """Retorna los datasets de verdad de las compuertas lógicas evaluadas."""
    X_bin = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

    datos = {
        "AND":  (X_bin, np.array([0, 0, 0, 1])),
        "OR":   (X_bin, np.array([0, 1, 1, 1])),
        "NAND": (X_bin, np.array([1, 1, 1, 0])),
        "XOR":  (X_bin, np.array([0, 1, 1, 0])),
        "NOT":  (np.array([[0], [1]], dtype=float), np.array([1, 0]))
    }
    return datos


def generar_grafico_fronteras(compuertas_entrenadas: dict, ruta_salida: str = "graficos/fronteras_compuertas.png"):
    """
    Genera un panel 2x2 mostrando las fronteras de decisión lineal de AND, OR, NAND
    y la imposibilidad geométrica de separar XOR con una sola neurona.
    """
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    titulos = ["AND", "OR", "NAND", "XOR"]
    posiciones = [(0, 0), (0, 1), (1, 0), (1, 1)]

    X_bin = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    for titulo, (r, c) in zip(titulos, posiciones):
        ax = axes[r, c]
        p = compuertas_entrenadas[titulo]["perceptron"]
        y = compuertas_entrenadas[titulo]["y"]
        convergio = compuertas_entrenadas[titulo]["convergio"]

        # Graficar puntos
        for xi, yi in zip(X_bin, y):
            color = "#10B981" if yi == 1 else "#EF4444"
            marcador = "o" if yi == 1 else "s"
            ax.scatter(xi[0], xi[1], color=color, s=140, marker=marcador, edgecolors="#0F172A", zorder=4)
            ax.text(xi[0] + 0.05, xi[1] + 0.05, f"({int(xi[0])},{int(xi[1])})={yi}", fontsize=8, fontweight="bold")

        # Graficar frontera de decisión: w1*x1 + w2*x2 + b = 0  => x2 = -(w1*x1 + b) / w2
        x_vals = np.linspace(-0.5, 1.5, 100)
        if abs(p.pesos[1]) > 1e-6:
            y_vals = -(p.pesos[0] * x_vals + p.sesgo) / p.pesos[1]
            estilo = "-" if convergio else "--"
            color_recta = "#2563EB" if convergio else "#F59E0B"
            ax.plot(x_vals, y_vals, linestyle=estilo, color=color_recta, linewidth=2, label="Frontera w·x+b=0")
        else:
            if abs(p.pesos[0]) > 1e-6:
                x_corte = -p.sesgo / p.pesos[0]
                ax.axvline(x=x_corte, color="#2563EB", linestyle="-", linewidth=2, label="Frontera vertical")

        sub_titulo = "Separable (Convergencia OK)" if convergio else "NO Separable Linealmente (Fallo)"
        ax.set_title(f"Compuerta {titulo} — {sub_titulo}", fontsize=11, fontweight="bold")
        ax.set_xlim(-0.2, 1.3)
        ax.set_ylim(-0.2, 1.3)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()
    print(f"[OK] Gráfico comparativo de fronteras de decisión exportado en: {ruta_salida}")


def ejecutar_laboratorio_compuertas():
    """Ejecuta el pipeline completo de entrenamiento de compuertas lógicas."""
    print("=" * 60)
    print("  LABORATORIO CLASE 8: PERCEPTRÓN Y COMPUERTAS LÓGICAS ")
    print("=" * 60)

    datos = obtener_datos_compuertas()
    compuertas_guardadas = {}

    for nombre, (X, y) in datos.items():
        num_in = X.shape[1]
        p = Perceptron(num_entradas=num_in, tasa_aprendizaje=0.1, epocas_max=50, nombre=f"Perceptron-{nombre}")
        convergio = p.entrenar(X, y)
        compuertas_guardadas[nombre] = {"perceptron": p, "X": X, "y": y, "convergio": convergio}

    # Generar visualización 2D
    generar_grafico_fronteras(compuertas_guardadas)
    print("\n[OK] Laboratorio de compuertas lógicas finalizado exitosamente.")
    return compuertas_guardadas


if __name__ == "__main__":
    ejecutar_laboratorio_compuertas()