
## 🚀 Avance 2 — Preparación de Datos y Perceptrón con Descenso del Gradiente

En el **Segundo Corte (Clases 7, 8 y 9)**, el proyecto evolucionó desde la comprensión descriptiva de los datos hacia su preparación formal para Machine Learning y el entrenamiento de un primer modelo neuronal con **descenso del gradiente** implementado desde cero en NumPy.

### 1. Resumen de Limpieza y Preparación de Datos con Pandas (`src/preparacion_datos_pandas.py`)
Se procesó el lote crudo de captura aplicando estandarización de texto en categorías temáticas, eliminación de registros duplicados, filtrado de valores atípicos/imposibles e imputación de valores nulos mediante la mediana por nivel CEFR, exportando el dataset limpio en [`data/proyecto_limpio.csv`](data/proyecto_limpio.csv) y la matriz codificada con `pd.get_dummies` en [`data/lecturas_preparadas_ml.csv`](data/lecturas_preparadas_ml.csv).

| Métrica de Calidad | Antes de la Limpieza (Crudo) | Después de la Limpieza (`proyecto_limpio.csv`) |
| :--- | :---: | :---: |
| **Número de filas** | 46 registros | **40 lecturas validadas** |
| **Número de columnas** | 7 columnas | **8 columnas** (+1 derivada: `velocidad_lectura_ppm`) |
| **Valores nulos totales** | 2 valores nulos (`NaN`) | **0 valores nulos** |
| **Filas duplicadas** | 3 duplicados | **0 duplicados** |
| **Valores atípicos / imposibles** | 3 registros corruptos | **0 registros corruptos** |
| **Categorías temáticas** | 9 etiquetas inconsistentes | **6 categorías estandarizadas** |

### 2. Resultados del Primer Modelo: Perceptrón desde Cero (`src/perceptron_proyecto.py`)
Se implementó un perceptrón con **función de activación sigmoide**, **pérdida logarítmica (Binary Cross-Entropy)** y **descenso del gradiente** en NumPy puro (sin Scikit-Learn) durante $500$ épocas ($\eta = 0.1$):
- **Variable Objetivo Binaria (`0/1`):** `objetivo_avanzado = 0` para lecturas de nivel **Básico (`A1/A2`)** y `objetivo_avanzado = 1` para lecturas de nivel **Intermedio-Avanzado (`B1/B2`)**. La columna `nivel_cefr` se excluye de las entradas para evitar fuga de datos (*data leakage*).
- **Variables de Entrada Estandarizadas (Z-score):** `conteo_palabras` ($\mu = 278.40, \sigma = 138.56$) y `dificultad_lexica` ($\mu = 3.21, \sigma = 1.17$).
- **Métricas Obtenidas:**
  - **Pesos finales:** $w_1 = 2.6936$ (`conteo_palabras`), $w_2 = 2.5407$ (`dificultad_lexica`), sesgo $b = 0.1333$.
  - **Reducción de la Pérdida Logarítmica:** Pérdida inicial $= 0.6931 \longrightarrow$ Pérdida final $= 0.0731$.
  - **Exactitud Final (Accuracy):** **$100.00\%$** ($40/40$ lecturas clasificadas correctamente).

#### Curva de Aprendizaje (Pérdida Logarítmica y Exactitud por Época)
Curva de Aprendizaje Perceptrón

*Archivo exportado:* [`graficos/curva_aprendizaje.png`](./graficos/curva_aprendizaje.png?raw=true)

#### Frontera de Decisión del Perceptrón (Sigmoide + Descenso del Gradiente)
Frontera de Decisión Perceptrón

*Archivo exportado:* [`graficos/frontera_decision.png`](./graficos/frontera_decision.png?raw=true)

---

## 🛠️ Instrucciones para Clonar y Ejecutar el Proyecto

### 1. Clonación del Repositorio
```bash
git clone https://github.com/moliech/IA_TutorIngles.git
cd IA_TutorIngles
```

### 2. Opción A: Ejecución Contenerizada con Docker (Recomendada)
```bash
# Construir y levantar el contenedor en segundo plano:
docker compose up -d --build

# Ingresar a la terminal interactiva del contenedor:
docker exec -it ia-tutor-ingles bash

# Ubicarse en /app/src y ejecutar el perceptrón del proyecto:
cd /app/src
python perceptron_proyecto.py

# Ejecutar el flujo principal completo:
python main.py
```

### 3. Opción B: Ejecución Directa con Python 3
```bash
pip install -r requirements.txt
python3 src/main.py
```