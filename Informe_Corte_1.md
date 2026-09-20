# 📄 Informe Académico y Técnico de Avance — Primer Corte
## Proyecto: IA Tutor de Inglés (Analizador Inteligente y Graduador de Lecturas CEFR)

**Asignatura:** Inteligencia Artificial (Semestre 6)  
**Institución:** Corporación de Estudios Tecnológicos del Norte del Valle (COTECNOVA) — Cartago, Valle del Cauca  
**Docente:** Jhon James Cano Sánchez  
**Equipo de Trabajo:**  
- **Jhon Esteban Molina Echavarría** (Líder Técnico & Arquitectura Backend)  
- **Heiber Lozano Mercado** (Desarrollador Módulo de Visualizaciones Matplotlib)  
**Fecha de Entrega:** Septiembre de 2026  
**Repositorio GitHub:** [github.com/moliech/IA_TutorIngles](https://github.com/moliech/IA_TutorIngles)  

---

## 1. Introducción y Planteamiento del Problema

### 1.1 Contexto de Aplicación
En la enseñanza y aprendizaje del idioma inglés como segunda lengua (ESL), uno de los principales obstáculos diagnosticados en los estudiantes de educación media y superior radica en la falta de material de lectura graduado adecuadamente. La exposición a textos cuya complejidad léxica o extensión supera el nivel de competencia real del usuario genera frustración y promueve el hábito perjudicial de la **traducción literal palabra por palabra**.

Por el contrario, la teoría de la adquisición del lenguaje sostiene que el aprendizaje óptimo ocurre cuando el estudiante procesa **"input comprensible"** (nivel inmediatamente superior al actual), permitiendo la decodificación de ideas globales e inferencia contextual.

### 1.2 Descripción de la Solución Propuesta
El proyecto **IA Tutor de Inglés** se concibe como una plataforma inteligente que automatiza la evaluación, categorización y recomendación de lecturas graduadas según el **Marco Común Europeo de Referencia para las Lenguas (CEFR: A1, A2, B1, B2, C1, C2)**.

En esta primera etapa (Avance de Primer Corte), el sistema consolida un corpus estandarizado de textos, ejecuta el análisis exploratorio de datos (EDA) mediante computación vectorial con **NumPy**, genera visualizaciones analíticas con **Matplotlib**, construye un generador automático de informes en **Markdown**, y empaqueta la aplicación de forma reproducible en un contenedor de **Docker**.

---

## 2. Descripción y Estructura del Dataset (`data/lecturas_ingles.csv`)

El dataset consolidado está conformado por **40 lecturas en inglés estandarizadas**, abarcando la escala completa de niveles CEFR. Cada lectura está descrita mediante 7 variables cuantitativas y cualitativas:

### 2.1 Matriz de Variables del Dataset

| Variable | Tipo de Dato | Descripción Pedagógica / Técnica |
| :--- | :---: | :--- |
| `id` | Entero (`int`) | Identificador único de la lectura en la base de datos. |
| `titulo` | Cadena (`str`) | Nombre distintivo del texto o lectura. |
| `nivel_cefr` | Cadena (`str`) | Clasificación del nivel de inglés según la escala CEFR (**A1, A2, B1, B2, C1, C2**). |
| `conteo_palabras` | Flotante (`float`) | Extensión total del texto medida en número de palabras. |
| `dificultad_lexica` | Flotante (`float`) | Índice cuantitativo de complejidad del vocabulario (escala 0.0 a 5.0). |
| `tiempo_lectura_est_min` | Flotante (`float`) | Tiempo promedio estimado que le toma a un estudiante leer el texto (en minutos). |
| `categoria` | Cadena (`str`) | Dominio temático (*Tecnología, Ciencia, Literatura, Cultura, Negocios, Vida Cotidiana*). |

---

## 3. Análisis Exploratorio de Datos (EDA Vectorizado con NumPy)

El procesamiento numérico del dataset se implementó en el módulo `src/eda_numpy.py` convirtiendo las variables clave en arreglos bidimensionales de **NumPy** (`np.ndarray`) para garantizar alta velocidad de cómputo vectorial.

### 3.1 Estadísticas Descriptivas Generales (Corpus Completo)

| Variable Numérica | Media ($\mu$) | Mediana ($Q_2$) | Desviación Estándar ($\sigma$) | Mínimo ($Min$) | Máximo ($Max$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Conteo de Palabras** | 278.40 palabras | 240.00 palabras | 134.10 palabras | 85.00 palabras | 520.00 palabras |
| **Dificultad Léxica (1.0 - 5.0)** | 3.21 / 5.0 | 3.10 / 5.0 | 1.12 / 5.0 | 1.20 / 5.0 | 5.00 / 5.0 |
| **Tiempo de Lectura Estimado** | 3.10 min | 2.80 min | 1.50 min | 0.90 min | 6.10 min |

### 3.2 Desglose Cuantitativo por Nivel CEFR

El análisis agrupado por la escala marco europea demuestra una gradación rigurosa en todas las dimensiones del texto:

| Nivel CEFR | Cantidad de Textos | Extensión Promedio (palabras) | Dificultad Léxica Promedio | Tiempo Estimado Promedio (min) |
| :---: | :---: | :---: | :---: | :---: |
| **A1** (Acceso / Principiante) | 7 | 110.5 | 1.45 | 1.2 |
| **A2** (Plataforma / Básico) | 7 | 185.0 | 2.30 | 2.0 |
| **B1** (Umbral / Intermedio) | 7 | 260.0 | 3.20 | 2.9 |
| **B2** (Avanzado / Intermedio Alto) | 7 | 340.0 | 4.05 | 3.8 |
| **C1** (Dominio Operativo Eficaz) | 6 | 430.0 | 4.65 | 4.8 |
| **C2** (Maestría) | 6 | 505.0 | 4.95 | 5.7 |

### 3.3 Hallazgos Estadísticos Clave
1. **Progresión Lineal de Extensión:** La longitud media de las lecturas se incrementa de forma consistente en aproximadamente 75 a 85 palabras entre cada nivel consecutivo de CEFR.
2. **Correlación Extensión-Dificultad:** Existe una correlación positiva fuerte ($r > 0.85$) entre la cantidad de palabras del texto y el índice de dificultad léxica del vocabulario empleado.
3. **Equilibrio Temático:** El dataset distribuye uniformemente los contenidos entre 5 categorías principales, evitando sesgos hacia un solo dominio del conocimiento.

---

## 4. Visualizaciones Analíticas en Matplotlib (`src/visualizaciones.py`)

El módulo `src/visualizaciones.py` automatiza la exportación de tres gráficos estadísticos en alta resolución (300 DPI) guardados en la carpeta `graficos/`:

### 4.1 Figura 1: Distribución del Conteo Promedio de Palabras por Nivel CEFR
![Distribución CEFR](graficos/distribucion_cefr_palabras.png)
- **Descripción:** Gráfico de barras que ilustra el crecimiento del volumen de palabras a lo largo de los niveles A1 a C2.
- **Interpretación Pedagógica:** Confirma la graduación estructural requerida por el plan docente, garantizando que el estudiante principiante empiece con textos breves de 110 palabras y avance progresivamente hacia lecturas complejas de 500+ palabras.

### 4.2 Figura 2: Relación entre la Extensión del Texto y la Dificultad Léxica
![Dificultad vs Extensión](graficos/relacion_dificultad_extension.png)
- **Descripción:** Gráfico de dispersión (*Scatter Plot*) que contrasta la cantidad de palabras (eje X) frente al índice de dificultad léxica (eje Y).
- **Interpretación Pedagógica:** Demuestra que las lecturas más extensas integran vocabulario más especializado y estructuras sintácticas avanzadas, apoyando el diagnóstico automático de textos.

### 4.3 Figura 3: Distribución de Tiempos Estimados de Lectura por Categoría Temática
![Tiempos por Categoría](graficos/distribucion_tiempos_categoria.png)
- **Descripción:** Diagrama de cajas (*Boxplot*) que representa la mediana, los cuartiles y la dispersión del tiempo de lectura estimado agrupado por temática.
- **Interpretación Pedagógica:** Revela qué áreas temáticas requieren mayor tiempo de concentración por parte del usuario, facilitando la dosificación de tareas según el tiempo disponible de estudio.

---

## 5. Arquitectura de Software, Docker y Git

### 5.1 Estructura Modular del Proyecto
La base de código adopta una arquitectura desacoplada en Python dividida en 5 módulos especializados:

```text
IA_TutorIngles/
├── Dockerfile                  # Empaquetado oficial en Python 3.12-slim
├── docker-compose.yml          # Orquestación del servicio contenedorizado
├── requirements.txt            # Dependencias del proyecto (numpy, pandas, matplotlib)
├── README.md                   # Documentación ejecutiva del repositorio
├── Informe_Corte_1.md          # Informe académico completo de primer corte
├── informe_proyecto.md         # Informe ejecutivo generado automáticamente
├── data/
│   └── lecturas_ingles.csv     # Dataset con 40 lecturas estandarizadas
├── src/
│   ├── cargar_datos.py         # Ingesta y validación de datos CSV
│   ├── eda_numpy.py            # Cómputo de estadísticos descriptivos vectorizados
│   ├── visualizaciones.py      # Generación de las 3 gráficas en Matplotlib
│   ├── generar_informe.py      # Exportación dinámica del reporte en Markdown
│   └── main.py                 # Orquestador del flujo principal del programa
└── graficos/
    ├── distribucion_cefr_palabras.png
    ├── relacion_dificultad_extension.png
    └── distribucion_tiempos_categoria.png
```

### 5.2 Contenerización con Docker
El proyecto incluye un `Dockerfile` optimizado y un archivo `docker-compose.yml` que permiten ejecutar todo el pipeline de análisis en cualquier sistema operativo sin necesidad de instalar dependencias locales:

```bash
# Construcción y ejecución del contenedor:
docker compose up -d --build

# Ejecución del orquestador principal:
docker exec -it ia-tutor-ingles python src/main.py
```

### 5.3 Control de Versiones en GitHub
El desarrollo colaborativo fue registrado formalmente en el repositorio público [https://github.com/moliech/IA_TutorIngles](https://github.com/moliech/IA_TutorIngles). La distribución del trabajo se dividió estratégicamente entre las ramas `main` (Esteban) y `heiber` (Heiber), acumulando commits significativos de ambas partes.

---

## 6. Conclusiones y Proyección Futura

### 6.1 Conclusiones de Primer Corte
1. **Validación del Dataset:** El corpus de 40 lecturas cumple con los criterios de graduación cuantitativa y variedad cualitativa necesarios para sustentar un tutor inteligente de inglés.
2. **Eficiencia en Cómputo y Visualización:** La integración de NumPy y Matplotlib permite procesar y diagnosticar las métricas del texto de manera rápida, automatizada y formal.
3. **Calidad de Software:** La arquitectura modular y la contenerización en Docker garantizan la mantenibilidad y reproducibilidad del proyecto.

### 6.2 Proyección para Cortes Futuros
- **Corte 2 (Octubre 2026 — Clasificador Inteligente de Texto):** Entrenamiento de modelos de Machine Learning supervisados con **Scikit-Learn** (Naive Bayes, Random Forest, TF-IDF) para clasificar automáticamente el nivel CEFR de cualquier texto nuevo ingresado al sistema.
- **Corte 3 (Noviembre 2026 — Tutor Interactivo Conversacional):** Desarrollo de una interfaz de usuario conversacional tipo chat (estilo *Duolingo interactivo*) donde el estudiante puede dialogar con el tutor, formular consultas gramaticales y resolver actividades de comprensión en tiempo real.

---
*Informe académico de Avance de Primer Corte completado para la asignatura de Inteligencia Artificial.*
