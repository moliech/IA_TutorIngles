# 📚 IA_TutorIngles - Tutor Inteligente de Lectura y Comprensión en Inglés

**Asignatura:** Inteligencia Artificial (Semestre 6)  
**Institución:** Corporación de Estudios Tecnológicos del Norte del Valle (COTECNOVA) — Cartago, Valle del Cauca  
**Docente:** Jhon James Cano Sánchez  
**Equipo de Trabajo:** Jhon Esteban Molina Echavarría, Heiber Lozano Mercado  
**Repositorio GitHub:** [https://github.com/moliech/IA_TutorIngles](https://github.com/moliech/IA_TutorIngles)  

---

## 📌 Resumen Ejecutivo del Proyecto

`IA_TutorIngles` es una plataforma analítica e inteligente diseñada para evaluar la extensión en palabras, la complejidad léxica y los tiempos estimados de lectura en textos en inglés graduados según el **Marco Común Europeo de Referencia para las Lenguas (CEFR: A1 a C2)**.

El objetivo central es diagnosticar cuantitativamente el nivel de los textos para mitigar la dependencia hacia la traducción literal palabra por palabra en los estudiantes, garantizando la asignación adaptativa de lecturas de acuerdo con la capacidad lingüística del usuario.

---

## 🔗 Accesos Directos a Archivos de Interés

- 📄 **Informe Académico Completo (Primer Corte):** [Informe_Corte_1.md](Informe_Corte_1.md) | [informes/Informe_Corte_1.md](informes/Informe_Corte_1.md)
- 🧠 **Informe de Reflexión sobre Neuronas Artificiales (Clase 8):** [informes/reflexion_aprendizaje_neurona.md](informes/reflexion_aprendizaje_neurona.md)
- 📊 **Informe Dinámico Exportado por Código:** [informes/informe_proyecto.md](informes/informe_proyecto.md)
- 📁 **Dataset Base Estandarizado (40 Lecturas):** [data/lecturas_ingles.csv](data/lecturas_ingles.csv)
- 🧹 **Dataset Limpio Exportado (Avance 2):** [data/proyecto_limpio.csv](data/proyecto_limpio.csv)
- 🤖 **Dataset Preparado para Machine Learning (`get_dummies`):** [data/lecturas_preparadas_ml.csv](data/lecturas_preparadas_ml.csv)
- 🎯 **Perceptrón desde Cero con Descenso del Gradiente (Clase 9):** [src/perceptron_proyecto.py](src/perceptron_proyecto.py)
- 🚀 **Orquestador Principal en Python:** [src/main.py](src/main.py)
- 🖼️ **Directorio de Visualizaciones Exportadas:** [graficos/](graficos/)

---

## 📁 Estructura del Repositorio

```text
IA_TutorIngles/
├── Dockerfile                        # Empaquetado oficial en Python 3.12-slim
├── docker-compose.yml                # Configuración del servicio contenedorizado
├── requirements.txt                  # Dependencias del proyecto (numpy, pandas, matplotlib, seaborn, etc.)
├── README.md                         # Documentación ejecutiva del repositorio
├── Informe_Corte_1.md                # Informe académico completo de primer corte
├── data/
│   ├── lecturas_ingles.csv           # Dataset base estandarizado (40 lecturas)
│   ├── proyecto_limpio.csv           # Dataset auditado y limpio exportado con Pandas (Clase 7)
│   └── lecturas_preparadas_ml.csv    # Dataset codificado con One-Hot Encoding listo para ML
├── src/                              # Arquitectura Modular en Python
│   ├── cargar_datos.py               # Módulo 1: Ingesta de CSV y conversión a matrices
│   ├── eda_numpy.py                  # Módulo 2: Estadísticas descriptivas vectorizadas (NumPy)
│   ├── visualizaciones.py            # Módulo 3: Generación de gráficos en Matplotlib
│   ├── preparacion_datos_pandas.py   # Módulo 4: Limpieza Pandas, Seaborn y One-Hot Encoding (Clase 7)
│   ├── generar_informe.py            # Módulo 5: Exportación del informe dinámico en Markdown
│   ├── neuronas_compuertas.py        # Módulo 6: Perceptrón de Rosenblatt para Compuertas Lógicas y XOR (Clase 8)
│   ├── clasificador_neurona_tutor.py # Módulo 7: Clasificador Neuronal con Función Escalón (Clase 8)
│   ├── perceptron_proyecto.py        # Módulo 8: Perceptrón con Sigmoide y Descenso del Gradiente (Clase 9)
│   └── main.py                       # Orquestador central del sistema
├── graficos/                         # Visualizaciones exportadas en 300 DPI (.png)
│   ├── distribucion_cefr_palabras.png
│   ├── relacion_dificultad_extension.png
│   ├── distribucion_tiempos_categoria.png
│   ├── seaborn_distribucion_ppm.png
│   ├── seaborn_palabras_por_cefr.png
│   ├── seaborn_dificultad_vs_tiempo.png
│   ├── fronteras_compuertas.png
│   ├── frontera_lecturas_tutor.png
│   ├── curva_aprendizaje.png
│   └── frontera_decision.png
└── informes/                         # Carpeta de informes generados
    ├── Informe_Corte_1.md
    ├── informe_proyecto.md
    └── reflexion_aprendizaje_neurona.md
```

---

## 📊 Métricas del Análisis Exploratorio (EDA con NumPy)

| Variable Numérica | Media ($\mu$) | Mediana ($Q_2$) | Desviación Estándar ($\sigma$) | Mínimo | Máximo |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Conteo de Palabras** | 278.4 palabras | 240.0 palabras | 134.1 palabras | 85.0 palabras | 520.0 palabras |
| **Dificultad Léxica (1.0 - 5.0)** | 3.21 / 5.0 | 3.10 / 5.0 | 1.12 / 5.0 | 1.20 / 5.0 | 5.00 / 5.0 |
| **Tiempo de Lectura Estimado** | 3.10 min | 2.80 min | 1.50 min | 0.90 min | 6.10 min |

---

## 🛠️ Preparación de Datos y Análisis Avanzado con Seaborn (Clase 7)

Se incorporó el módulo [src/preparacion_datos_pandas.py](src/preparacion_datos_pandas.py) enfocado en la manipulación con **Pandas**, visualización estadística avanzada con **Seaborn** y preparación de variables para Machine Learning:

1. **Manejo e Inspección de Calidad de Datos:** Se inspeccionó el lote crudo con `shape`, `info()`, `isnull().sum()` y `duplicated().sum()`, tratando valores nulos mediante imputación por mediana agrupada, eliminando registros duplicados y atípicos, y normalizando categorías inconsistentes hacia [data/proyecto_limpio.csv](data/proyecto_limpio.csv).
2. **Columna Derivada Creada:** 
   - `velocidad_lectura_ppm`: Mide la tasa de Palabras Por Minuto estimadas para cada lectura ($\text{PPM} = \frac{\text{Conteo de Palabras}}{\text{Tiempo Estimado (min)}}$).
3. **Codificación Categórica (One-Hot Encoding):**
   - Transformación de variables cualitativas `nivel_cefr` y `categoria` mediante `pd.get_dummies()`.
   - Exportación de la matriz codificada resultante a [data/lecturas_preparadas_ml.csv](data/lecturas_preparadas_ml.csv).
4. **Visualizaciones Estadísticas Avanzadas con Seaborn:**
   - [`graficos/seaborn_distribucion_ppm.png`](./graficos/seaborn_distribucion_ppm.png?raw=true): Histograma con estimación de densidad de núcleo (KDE) de la velocidad PPM.
   - [`graficos/seaborn_palabras_por_cefr.png`](./graficos/seaborn_palabras_por_cefr.png?raw=true): Diagrama de cajas (*Boxplot*) de extensión de palabras comparada por Nivel CEFR.
   - [`graficos/seaborn_dificultad_vs_tiempo.png`](./graficos/seaborn_dificultad_vs_tiempo.png?raw=true): Diagrama de dispersión (*Scatterplot*) relacionando dificultad léxica y tiempo estimado por categoría temática.

---

## 🧠 Redes Neuronales Artificiales: Perceptrón y Clasificación Adaptativa (Clase 8)

En la sesión de **Redes Neuronales Artificiales (Clase 8)**, se diseñó e integró una arquitectura basada en el **Perceptrón Simple de Rosenblatt**:

1. **Laboratorio de Compuertas Lógicas ([src/neuronas_compuertas.py](src/neuronas_compuertas.py)):**
   - Implementación desde cero de la clase `Perceptron` con regla de aprendizaje supervisado y función de activación escalón unitario ($\Theta$).
   - Convergencia demostrada sobre funciones linealmente separables (**AND, OR, NOT, NAND**).
   - Demostración empírica de la no convergencia sobre la compuerta **XOR** (problema de separabilidad no lineal formulado por Minsky & Papert, 1969).
2. **Clasificador Neuronal Adaptativo del Tutor ([src/clasificador_neurona_tutor.py](src/clasificador_neurona_tutor.py)):**
   - Entrenamiento de una neurona artificial sobre el corpus de 40 lecturas de `data/lecturas_ingles.csv` utilizando características normalizadas (*Min-Max*): `conteo_palabras` y `dificultad_lexica`.
   - Clasificación automática de lecturas en nivel **Básico (A1/A2 $\to 0$)** vs **Avanzado (B1/B2 $\to 1$)** alcanzando **100.0% de exactitud** en la época 4.
3. **Reflexión Académica y Respuestas de la Guía:**
   - Las respuestas detalladas a las 4 preguntas de reflexión sobre el aprendizaje neuronal (épocas de convergencia, tasa de aprendizaje, comparación AND vs OR y limitación en XOR) se encuentran documentadas en: 👉 **[informes/reflexion_aprendizaje_neurona.md](informes/reflexion_aprendizaje_neurona.md)**.

---

## 🖼️ Visualizaciones Estadísticas y Neuronales Exportadas

El sistema genera automáticamente 10 gráficos analíticos en la carpeta `graficos/`:

### 1. Conteo Promedio de Palabras por Nivel CEFR (Matplotlib)
<img src="./graficos/distribucion_cefr_palabras.png?raw=true" width="700" alt="Distribución CEFR Palabras" />

*Archivo exportado:* [`graficos/distribucion_cefr_palabras.png`](./graficos/distribucion_cefr_palabras.png?raw=true)

---

### 2. Extensión del Texto vs Dificultad Léxica (Matplotlib)
<img src="./graficos/relacion_dificultad_extension.png?raw=true" width="700" alt="Relación Dificultad Extensión" />

*Archivo exportado:* [`graficos/relacion_dificultad_extension.png`](./graficos/relacion_dificultad_extension.png?raw=true)

---

### 3. Tiempos Estimados de Lectura por Categoría (Matplotlib)
<img src="./graficos/distribucion_tiempos_categoria.png?raw=true" width="700" alt="Distribución Tiempos Categoría" />

*Archivo exportado:* [`graficos/distribucion_tiempos_categoria.png`](./graficos/distribucion_tiempos_categoria.png?raw=true)

---

### 4. Distribución de Velocidad de Lectura PPM (Seaborn)
<img src="./graficos/seaborn_distribucion_ppm.png?raw=true" width="700" alt="Seaborn Distribución PPM" />

*Archivo exportado:* [`graficos/seaborn_distribucion_ppm.png`](./graficos/seaborn_distribucion_ppm.png?raw=true)

---

### 5. Extensión de Palabras por Nivel CEFR (Seaborn Boxplot)
<img src="./graficos/seaborn_palabras_por_cefr.png?raw=true" width="700" alt="Seaborn Palabras por CEFR" />

*Archivo exportado:* [`graficos/seaborn_palabras_por_cefr.png`](./graficos/seaborn_palabras_por_cefr.png?raw=true)

---

### 6. Dificultad Léxica vs Tiempo Estimado de Lectura (Seaborn Scatterplot)
<img src="./graficos/seaborn_dificultad_vs_tiempo.png?raw=true" width="700" alt="Seaborn Dificultad vs Tiempo" />

*Archivo exportado:* [`graficos/seaborn_dificultad_vs_tiempo.png`](./graficos/seaborn_dificultad_vs_tiempo.png?raw=true)

---

### 7. Fronteras de Decisión en Compuertas Lógicas y Fallo en XOR (Perceptrón)
<img src="./graficos/fronteras_compuertas.png?raw=true" width="700" alt="Fronteras de Decisión Compuertas Lógicas" />

*Archivo exportado:* [`graficos/fronteras_compuertas.png`](./graficos/fronteras_compuertas.png?raw=true)

---

### 8. Clasificación Adaptativa de Lecturas con Perceptrón Simple
<img src="./graficos/frontera_lecturas_tutor.png?raw=true" width="700" alt="Frontera Neuronal de Lecturas" />

*Archivo exportado:* [`graficos/frontera_lecturas_tutor.png`](./graficos/frontera_lecturas_tutor.png?raw=true)

---

## 🚀 Avance 2 — Preparación de Datos y Perceptrón con Descenso del Gradiente

En el **Segundo Corte (Clases 7, 8 y 9)**, el proyecto evolucionó desde la comprensión descriptiva de los datos hacia su preparación formal para Machine Learning y el entrenamiento de un primer modelo neuronal con **descenso del gradiente** implementado desde cero en NumPy.

### 1. Resumen de Limpieza y Preparación de Datos con Pandas (`src/preparacion_datos_pandas.py`)
Se procesó el lote crudo de captura aplicando estandarización de texto en categorías temáticas, eliminación de registros duplicados, filtrado de valores atípicos/imposibles e imputación de valores nulos mediante la mediana agrupada por nivel CEFR y categoría, exportando el dataset limpio en [`data/proyecto_limpio.csv`](data/proyecto_limpio.csv) y la matriz codificada con `pd.get_dummies` en [`data/lecturas_preparadas_ml.csv`](data/lecturas_preparadas_ml.csv).

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

### 9. Curva de Aprendizaje (Pérdida Logarítmica y Exactitud por Época)
<img src="./graficos/curva_aprendizaje.png?raw=true" width="750" alt="Curva de Aprendizaje Perceptrón" />

*Archivo exportado:* [`graficos/curva_aprendizaje.png`](./graficos/curva_aprendizaje.png?raw=true)

---

### 10. Frontera de Decisión del Perceptrón (Sigmoide + Descenso del Gradiente)
<img src="./graficos/frontera_decision.png?raw=true" width="700" alt="Frontera de Decisión Perceptrón" />

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
# Instalar las librerías necesarias:
pip install -r requirements.txt

# Ejecutar el programa principal:
python3 src/main.py
```
