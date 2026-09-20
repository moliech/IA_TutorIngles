# 📚 IA_TutorIngles - Tutor Inteligente de Lectura y Comprensión en Inglés

**Asignatura:** Inteligencia Artificial (Semestre 6)  
**Institución:** Corporación de Estudios Tecnológicos del Norte del Valle (COTECNOVA)  
**Docente:** Jhon James Cano Sánchez  
**Equipo de Trabajo:**  
- **Jhon Esteban Molina Echavarría** (Líder Técnico & Arquitectura Backend)  
- **Heiber Lozano Mercado** (Desarrollador Módulo de Visualizaciones Matplotlib)  

---

## 📌 Descripción del Proyecto

`IA_TutorIngles` es un sistema inteligente de soporte educativo diseñado para evaluar la extensión en palabras, la dificultad léxica y los tiempos estimados de lectura en textos en inglés clasificados según el Marco Común Europeo de Referencia (**CEFR: A1 a C2**).

El objetivo principal es permitir la nivelación progresiva del estudiante y la dosificación del contenido para fomentar la comprensión lectora activa e interactiva mediante análisis cuantitativo y visualizaciones estadísticas.

---

## 📁 Estructura del Repositorio

```text
IA_TutorIngles/
├── Dockerfile                  # Imagen Docker basada en Python 3.12-slim
├── docker-compose.yml          # Configuración del servicio ia-tutor-ingles
├── requirements.txt            # Dependencias (numpy, pandas, matplotlib, scikit-learn, jupyter)
├── README.md                   # Documentación ejecutiva del proyecto
├── informe_proyecto.md         # Informe Markdown generado automáticamente
├── data/
│   └── lecturas_ingles.csv     # Dataset real con 40 lecturas estandarizadas por CEFR
├── src/                        # Arquitectura Modular en Python
│   ├── cargar_datos.py         # Módulo 1: Carga de CSV y estructuras numéricas
│   ├── eda_numpy.py            # Módulo 2: Estadísticas descriptivas vectorizadas con NumPy
│   ├── visualizaciones.py      # Módulo 3: Generación de gráficos en Matplotlib
│   ├── generar_informe.py      # Módulo 4: Exportación del informe en Markdown
│   └── main.py                 # Orquestador central del sistema
└── graficos/                   # Visualizaciones exportadas en alta resolución (.png)
    ├── distribucion_cefr_palabras.png
    ├── relacion_dificultad_extension.png
    └── distribucion_tiempos_categoria.png
```

---

## 📊 Resultados del Análisis Exploratorio de Datos (EDA con NumPy y Matplotlib)

### 📈 Tabla de Estadísticas Descriptivas (Calculadas con NumPy)

| Variable Numérica | Media (Average) | Mediana (Median) | Desviación Estándar (Std) | Mínimo | Máximo |
|---|:---:|:---:|:---:|:---:|:---:|
| **Conteo de Palabras** | 278.4 palabras | 240.0 palabras | 134.1 palabras | 85.0 palabras | 520.0 palabras |
| **Dificultad Léxica (1.0 - 5.0)** | 3.21 / 5.0 | 3.10 / 5.0 | 1.12 / 5.0 | 1.20 / 5.0 | 5.00 / 5.0 |
| **Tiempo de Lectura Estimado** | 3.10 min | 2.80 min | 1.50 min | 0.90 min | 6.10 min |

---

## 🖼️ Visualizaciones Gráficas Exportadas

El módulo `src/visualizaciones.py` procesa la información y genera automáticamente 3 gráficos estadísticos en la carpeta `graficos/`:

1. **`graficos/distribucion_cefr_palabras.png`**: Gráfico de barras que compara el promedio de palabras por nivel CEFR.
2. **`graficos/relacion_dificultad_extension.png`**: Gráfico de dispersión (*Scatter Plot*) que demuestra la relación entre el número de palabras y la dificultad léxica.
3. **`graficos/distribucion_tiempos_categoria.png`**: Diagrama de cajas (*Boxplot*) que representa la dispersión del tiempo estimado de lectura según la categoría temática.

---

## 🛠️ Instrucciones de Ejecución

### Opción 1: Ejecución con Docker (Recomendada)
```bash
# 1. Levantar el contenedor
docker compose up -d --build

# 2. Ejecutar el orquestador principal
docker exec -it ia-tutor-ingles python src/main.py
```

### Opción 2: Ejecución Local con Python
```bash
python src/main.py
```
