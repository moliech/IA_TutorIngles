# 📚 IA_TutorIngles - Tutor Inteligente de Lectura y Comprensión en Inglés

**Asignatura:** Inteligencia Artificial (Semestre 6)  
**Institución:** Corporación de Estudios Tecnológicos del Norte del Valle (COTECNOVA)  
**Docente:** Jhon James Cano Sánchez  
**Equipo de Trabajo:**  
- **Jhon Esteban Molina Echavarría** (Líder Técnico & Arquitectura Backend)  
- **Heiber Lozano Mercado** (Desarrollador Módulo de Visualizaciones Matplotlib)  

---

## 📌 Descripción del Proyecto

`IA_TutorIngles` es un sistema inteligente de soporte educativo diseñado para evaluar la extensión en palabras y la dificultad léxica de textos en inglés clasificados según el Marco Común Europeo de Referencia (**CEFR: A1, A2, B1, B2**). 

El objetivo principal es permitir la nivelación progresiva del estudiante y la dosificación del contenido para fomentar la comprensión de ideas globales en lugar de la traducción literal palabra por palabra.

---

## 📁 Estructura del Repositorio

```text
IA_TutorIngles/
├── Dockerfile                  # Imagen Docker basada en Python 3.12-slim
├── docker-compose.yml          # Configuración del servicio ia-tutor-ingles
├── requirements.txt            # Dependencias (numpy, pandas, matplotlib, scikit-learn, jupyter)
├── README.md                   # Documentación ejecutiva del proyecto
├── .gitignore                  # Reglas de exclusión de Git
├── .dockerignore               # Reglas de exclusión de Docker
├── data/
│   └── lecturas_ingles.csv     # Dataset real con 40 lecturas estandarizadas por CEFR
├── src/                        # Arquitectura Modular en Python
│   ├── cargar_datos.py         # Módulo 1: Carga de CSV y estructuras numéricas
│   ├── eda_numpy.py            # Módulo 2: Estadísticas descriptivas vectorizadas con NumPy
│   ├── visualizaciones.py      # Módulo 3: Generación de gráficos en Matplotlib
│   ├── generar_informe.py      # Módulo 4: Exportación del informe en Markdown
│   └── main.py                 # Orquestador central del sistema (Versión 2 Integrada)
├── graficos/                   # Visualizaciones exportadas en alta resolución (.png)
│   ├── distribucion_cefr_palabras.png
│   └── relacion_dificultad_extension.png
└── informes/                   # Informes técnicos generados automáticamente
    └── informe_eda.md
```

---

## 📊 Resultados del Análisis Exploratorio de Datos (EDA con NumPy y Matplotlib)

### 📈 Tabla de Estadísticas Descriptivas (Calculadas con NumPy)

| Variable Numérica | Media (Average) | Mediana (Median) | Desviación Estándar (Std) | Mínimo | Máximo |
|---|---|---|---|---|---|
| **Conteo de Palabras** | 276.15 palabras | 235.00 palabras | 136.24 palabras | 88.00 palabras | 520.00 palabras |
| **Dificultad Léxica (1.0 - 5.0)** | 3.23 / 5.0 | 3.15 / 5.0 | 1.15 / 5.0 | 1.20 / 5.0 | 5.00 / 5.0 |
| **Tiempo de Lectura Estimado** | 3.08 min | 2.75 min | 1.54 min | 0.90 min | 6.10 min |

### 🎯 Distribución por Nivel CEFR
- **Nivel A1 (Principiante):** 10 lecturas (Promedio de 113.10 palabras | Dificultad 1.2 - 2.1)
- **Nivel A2 (Principiante Intermedio):** 10 lecturas (Promedio de 199.00 palabras | Dificultad 2.3 - 3.2)
- **Nivel B1 (Intermedio):** 10 lecturas (Promedio de 329.50 palabras | Dificultad 3.4 - 4.1)
- **Nivel B2 (Intermedio Avanzado):** 10 lecturas (Promedio de 463.00 palabras | Dificultad 4.3 - 5.0)

---

## 🖼️ Visualizaciones Gráficas Exportadas

El módulo `src/visualizaciones.py` procesa la información y genera automáticamente 2 gráficos clave en la carpeta `graficos/`:

1. **`graficos/distribucion_cefr_palabras.png`**: Gráfico de barras que compara la cantidad de textos y la progresión del volumen de palabras por nivel CEFR.
2. **`graficos/relacion_dificultad_extension.png`**: Gráfico de dispersión (*Scatter Plot*) que demuestra la correlación directa entre el número de palabras y la complejidad léxica.

---

## 🛠️ Instrucciones de Ejecución

### Opción 1: Ejecución con Docker (Recomendada)
```bash
# 1. Levantar el contenedor
docker compose up -d

# 2. Ejecutar el orquestador principal
docker exec -it ia-tutor-ingles python src/main.py
```

### Opción 2: Ejecución Local con Python
```bash
python src/main.py
```
