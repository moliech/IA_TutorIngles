# Informe de Avance de Proyecto - Primer Corte

**Asignatura:** Inteligencia Artificial (Semestre 6)  
**Institución:** Corporación de Estudios Tecnológicos del Norte del Valle (COTECNOVA)  
**Docente:** Jhon James Cano Sánchez  
**Proyecto:** Tutor Virtual de Inglés Adaptativo: Motor de Medición y Clasificación de Lecturas CEFR  
**Integrantes:** Jhon Esteban Molina Echavarría & Heiber Lozano Mercado  
**Fecha de Entrega / Sustentación:** Septiembre de 2026  

---

## 1. Definición y Contextualización del Proyecto

### 1.1 Propósito del Proyecto y Visión del Estudiante
El propósito fundamental de este proyecto es el desarrollo de un **Tutor Virtual de Inglés Conversacional (Chatbot Interactivo)** que actúe como un compañero de estudio y docente personalizado de bolsillo —similar al concepto adaptativo de plataformas como Duolingo, pero mediante una experiencia conversacional cercana, fluida y sin juicio afectivo.

La experiencia interactiva del estudiante dentro del sistema está estructurada en 4 momentos didácticos:
1. **Diagnóstico Conversacional de Nivel:** El estudiante interactúa en el chat y el sistema evalúa su vocabulario y comprensión para identificar su nivel real en el Marco Común Europeo de Referencia (CEFR: A1 a C2).
2. **Asignación Adaptativa de Lecturas:** En lugar de contenidos genéricos, el tutor selecciona lecturas graduadas a la capacidad exacta del usuario y ajustadas a sus temas de interés (Tecnología, Cultura, Cine, Deportes, etc.).
3. **Chat de Acompañamiento en Tiempo Real:** Durante la lección, el usuario puede resolver dudas gramaticales en vivo (*"¿Qué significa esta palabra?"*, *"¿Por qué se usa este tiempo verbal?"*), mientras el tutor formula preguntas de comprensión lectora.
4. **Retroalimentación Formativa y Ascenso de Nivel:** El tutor ofrece correcciones de manera amigable y, conforme el estudiante demuestra un dominio sostenido, lo promueve automáticamente al siguiente nivel CEFR.

---

### 1.2 Hoja de Ruta de Desarrollo (Conexión entre Cortes Académicos)
Para construir un sistema de IA capaz de adaptar sus lecciones a la medida de cada estudiante, el proyecto se desarrolla en 3 etapas secuenciales:

- **Corte 1 — El Motor de Medición (Fase Actual):** Antes de interactuar mediante chat, el sistema debe aprender a medir objetivamente la complejidad de los textos. Se construyó la infraestructura analítica sobre 40 lecturas graduadas con métricas de extensión, dificultad léxica y tiempos estimados de lectura procesados vectorialmente en **NumPy** y visualizados en **Matplotlib**.
- **Corte 2 — El Cerebro Clasificador (Machine Learning / PLN):** Se implementará un modelo inteligente de aprendizaje automático capaz de analizar entradas de texto o respuestas del estudiante y determinar de inmediato a qué nivel CEFR corresponden.
- **Corte 3 — La Interfaz Conversacional (El Tutor en Vivo):** Se integrará la pantalla interactiva de chat donde el estudiante vive la experiencia adaptativa completa en tiempo real.

---

### 1.3 Problemática Socioeducativa en Cartago, Valle del Cauca (Datos Oficiales 2025)

#### A. Panorama Nacional y Departamental
El dominio del inglés como segunda lengua (EFL) es una competencia crítica para la movilidad académica y profesional. Sin embargo, Colombia enfrenta un rezago estructural:
- **Índice EF EPI (2025):** Colombia ocupa la posición 76 entre 123 países evaluados ("Dominio Bajo" con 480 puntos), y en el departamento del Valle del Cauca el promedio desciende a 454 puntos.
- **Pruebas ICFES Saber 11 (2025):** A nivel nacional, solo el 6% de los estudiantes del sector público alcanza el nivel B1 o superior del marco CEFR. En contraste, más del 54% de los bachilleres oficiales egresa en la categoría Pre-A1, denotando limitaciones funcionales en el idioma.

#### B. Panorama Municipal en Cartago, Valle
En el municipio de Cartago, si bien la prueba específica de Inglés (Saber 11 - 2025) registró una media de **55.5 sobre 100 puntos** (superando la media nacional de 256.3 puntos globales), el desglose granular expone una inequidad socioeducativa preocupante:
- **Brecha Institucional Privada vs. Oficial:** Existe una distancia neta de **27.9 puntos** entre el rendimiento global de colegios privados (promedio 276.4 pts) y colegios oficiales (promedio 248.5 pts).
- **Dispersión en Colegios Públicos:** Mientras la *I.E. GABO* actúa como referente atípico de excelencia (318.1 pts), planteles oficiales emblemáticos de Cartago como la *I.E. Manuel Quintero Penilla* y la *I.E. Sor María Juliana* se ubican en rangos bajos de 221.6 a 255.8 puntos.
- **Carencia de Entornos Adaptativos:** Más de **1,100 estudiantes** presentan anualmente las pruebas de estado en Cartago sin acceso a laboratorios bilingües ni tutores personalizados fuera del aula. Este proyecto busca democratizar el acceso al aprendizaje adaptativo en el municipio.

---

### 1.4 Objetivos del Sistema de IA

#### Objetivo General
Desarrollar e implementar un **Tutor Virtual de Inglés Conversacional Adaptativo** asistido por Inteligencia Artificial, capaz de **clasificar** automáticamente el nivel de complejidad de lecturas en inglés bajo el marco CEFR (A1 a C2), **predecir** el tiempo estimado de lectura y la dificultad léxica de los textos, **recomendar** contenidos personalizados según los intereses del usuario, y **proveer** un entorno conversacional interactivo de aprendizaje para mitigar la brecha de rendimiento comunicativo en los estudiantes de Cartago, Valle.

#### Tareas Técnicas de la IA
1. **Clasificación Automática CEFR:** Categorización supervisada de textos e interacciones en niveles A1-C2, superando las limitaciones de fórmulas tradicionales.
2. **Predicción y Calibración Léxica:** Cómputo vectorizado del índice de dificultad (1.0 a 5.0) y tiempo de lectura, manteniendo al estudiante en su **Zona de Desarrollo Próximo (ZDP)**.
3. **Recomendación Adaptativa:** Filtrado de contenidos por nivel de habilidad y preferencias temáticas.
4. **Diagnóstico Conversacional:** Acompañamiento en tiempo real con retroalimentación gramatical amigable y sin juicio social.

---

### 1.5 Origen, Estructura y Procesamiento de Datos

#### A. Fuente de los Datos
Los datos provienen de una compilación y curaduría pedagógica basada en datos abiertos (*Open Data*):
- **CEFR English Text Classification Dataset (Kaggle):** Pasajes de lectura etiquetados por lingüistas bajo la escala CEFR.
- **Graded Readers (Cambridge/Oxford):** Muestras de lecturas graduadas para la extracción de frecuencias de vocabulario y extensión.

#### B. Diccionario de Datos

| Variable | Tipo de Dato | Rango / Dominio | Descripción Técnica |
| :--- | :--- | :--- | :--- |
| **`id`** | Entero (`int`) | 1, 2, 3, ... N | Identificador único del texto. |
| **`titulo`** | Cadena (`str`) | Texto alfanumérico | Título de la lectura en inglés. |
| **`nivel`** | Cadena (`str`) | `A1`, `A2`, `B1`, `B2`, `C1`, `C2` | Etiqueta del nivel estándar europeo (Variable objetivo). |
| **`palabras`** | Entero (`int`) | 50 a 800+ | Conteo total de palabras de la lectura (extensión). |
| **`dificultad`** | Flotante (`float`) | 1.0 a 5.0 | Índice computacional de complejidad léxica. |
| **`categoria`** | Cadena (`str`) | `Tecnología`, `Cultura`, `Deportes`, etc. | Área temática para la recomendación adaptativa. |
| **`tiempo_lectura_est_min`** | Flotante (`float`) | 0.5 a 15.0 min | Tiempo estimado de lectura fluida. |

---

## 2. Estructura de Datos y Funciones en Python

### 2.1 Carga del CSV y Estructuración en Memoria
El sistema procesa el archivo plano `data/lecturas_ingles.csv` mediante la biblioteca nativa `csv.DictReader` de Python. Durante la lectura se ejecuta un casteo explícito de tipos (`int` y `float`), almacenando la información en memoria como una **Lista de Diccionarios** (`list[dict]`). 

Para el cómputo numérico de alto rendimiento, las columnas de extensión (`palabras`) y complejidad (`dificultad`) se transforman en matrices bi-dimensionales vectorizadas de **NumPy** (`numpy.ndarray`), permitiendo ejecutar operaciones matemáticas en tiempo continuo $O(N)$ sin recurrir a ciclos `for` lentos.

---

### 2.2 Resumen Funcional de Módulos Desarrollados

El código fuente del Primer Corte está organizado de forma modular en el directorio `src/`:

1. **`leer_datos(ruta_archivo)` (`src/cargar_datos.py`):** Carga el CSV de forma segura desde múltiples rutas relativas, castea los tipos de datos y retorna la lista de diccionarios.
2. **`calcular_estadisticas_numpy(array_numerico)` (`src/eda_numpy.py`):** Recibe el arreglo bidimensional de NumPy y ejecuta funciones vectorizadas (`np.mean`, `np.median`, `np.std`, `np.max`, `np.min`) para retornar los indicadores descriptivos.
3. **`generar_graficos_matplotlib(datos, array_numerico)` (`src/visualizaciones.py`):** Construye y exporta a la carpeta `graficos/` las figuras en formato PNG a 300 DPI (barras por nivel CEFR, dispersión de dificultad vs palabras, y diagrama de cajas por categoría).
4. **`crear_informe_markdown(estadisticas)` (`src/generar_informe.py`):** Genera dinámicamente el reporte técnico ejecutivo `informe_proyecto.md` incrustando las visualizaciones y tablas de resultados.
5. **Orquestador `main()` (`src/main.py`):** Coordina secuencialmente la ejecución de los 4 módulos desde la carga hasta la exportación del informe final.

---

## 3. Análisis Exploratorio de Datos (EDA) y Hallazgos Clave

### 3.1 Resultados Estadísticos Calculados con NumPy
Al procesar las lecturas en el módulo `src/eda_numpy.py`, se obtuvieron los siguientes indicadores cuantitativos consolidados:

| Métrica Estadística | Variable: Conteo de Palabras (`palabras`) | Variable: Dificultad Léxica (`dificultad`) |
| :--- | :--- | :--- |
| **Promedio (Media `np.mean`)** | $255.0$ palabras | $2.58$ / 5.0 |
| **Mediana (`np.median`)** | $210.0$ palabras | $2.30$ / 5.0 |
| **Desviación Estándar (`np.std`)** | $112.4$ palabras | $1.15$ |
| **Valor Máximo (`np.max`)** | $410$ palabras (*"Job Interview Tips"*) | $4.10$ (*"Job Interview Tips"*) |
| **Valor Mínimo (`np.min`)** | $95$ palabras (*"Ordering Food"*) | $1.20$ (*"Ordering Food"*) |

---

### 3.2 Visualizaciones Generadas en Matplotlib
El sistema exporta tres visualizaciones gráficas clave en `graficos/`:
1. **Figura 1 (`distribucion_cefr_palabras.png`):** Diagrama de barras que ilustra la cantidad de lecturas y la extensión media por cada nivel CEFR.
2. **Figura 2 (`relacion_dificultad_extension.png`):** Gráfico de dispersión (*Scatter Plot*) que correlaciona el conteo de palabras frente al índice de dificultad léxica.
3. **Figura 3 (`distribucion_tiempos_categoria.png`):** Diagrama de cajas (*Boxplot*) que compara la dispersión del tiempo estimado de lectura en minutos por área temática.

---

### 3.3 Hallazgos Clave e Interpretación Pedagógica

- **Hallazgo 1: Correlación Positiva Directa entre Extensión y Dificultad Léxica:**  
  El gráfico de dispersión confirma que a mayor número de palabras en la lectura, mayor es el índice de dificultad del vocabulario. Los textos A1 promedian 100-120 palabras con dificultad $<2.0$, mientras que textos B1 superan 350 palabras con dificultad $>3.8$.  
  *Implicación:* El Tutor IA debe calibrar la longitud del texto junto con la densidad léxica para no saturar al estudiante.

- **Hallazgo 2: Concentración de Contenidos en Niveles Iniciales (A1-B1):**  
  El 70% de la base de datos procesada corresponde a niveles A1 a B1.  
  *Implicación:* Esta distribución se alinea perfectamente con la realidad de Cartago, donde el 54% de los bachilleres oficiales egresa en Pre-A1. Permite iniciar el proceso de nivelación sin desmotivar a los alumnos.

- **Hallazgo 3: Dispersión del Tiempo de Lectura por Categoría Temática:**  
  Las lecturas de *Tecnología* y *Negocios* presentan tiempos de lectura estimados significativamente más altos ($>4$ min) que las de *Cultura* o *Vida Cotidiana* ($1.5$ a $2.5$ min), a pesar de tener una cantidad de palabras similar.  
  *Implicación:* Los temas técnicos contienen vocabulario especializado que exige mayor tiempo de procesamiento. El tutor ofrecerá más asistencia gramatical en vivo en estas categorías.

---

## 4. Control de Versiones y Documentación en GitHub

### 4.1 Repositorio Oficial del Proyecto
- **Enlace al Repositorio:** [https://github.com/moliech/IA_TutorIngles](https://github.com/moliech/IA_TutorIngles)
- **Ramas de Trabajo:** Rama principal `main` y rama de desarrollo `heiber`.

### 4.2 Historial de Commits Registrados

| Commit | Mensaje del Commit | Módulos Modificados |
| :--- | :--- | :--- |
| **Commit 1** | `init: estructura inicial de la rama heiber y modulos base` | Estructura `src/`, `data/`, `README.md` |
| **Commit 2** | `feat: agregar 3er grafico plus distribucion de tiempos de lectura por categoria` | `src/visualizaciones.py`, `graficos/` |
| **Commit 3** | `feat: integrar visualizaciones de matplotlib y tablas ejecutivas en el informe markdown` | `src/generar_informe.py`, `src/main.py`, `informe_proyecto.md` |

---

## 5. Conclusiones y Próximos Pasos

1. **Cumplimiento de Objetivos del Corte 1:** Se consolidó exitosamente el Motor de Medición Analítico en Python, integrando la carga de datos CSV, estadísticas vectorizadas con NumPy y gráficos en Matplotlib.
2. **Base para el Modelo Inteligente (Corte 2):** Los hallazgos del EDA establecen las variables cuantitativas de entrada necesarias para entrenar modelos de clasificación supervisada (Machine Learning / PLN) en la siguiente fase.
3. **Pauta para la Interfaz Conversacional (Corte 3):** Se definió el flujo de experiencia del estudiante en 4 etapas, asegurando que el Tutor de Inglés sea una solución adaptativa viable para reducir la brecha socioeducativa en Cartago.

---

## 6. Referencias y Obras Citadas

1. **EF Education First (2025).** *EF English Proficiency Index (EF EPI 2025) - Reporte Colombia*.
2. **ICFES (2025).** *Resultados Agregados de las Pruebas Saber 11 - Departamento del Valle del Cauca y Municipio de Cartago*.
3. **Ministerio de Educación Nacional de Colombia (2014).** *Programa Nacional de Bilingüismo: Desafíos y Metas en Educación Media*.
4. **EDIA / CEFR Classification Research (2023).** *Why Flesch-Kincaid Readability Metrics Fall Short for CEFR Text Classification in EFL Environments*.
