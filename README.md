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
- 📊 **Informe Dinámico Exportado por Código:** [informes/informe_proyecto.md](informes/informe_proyecto.md)
- 📁 **Dataset Base Estandarizado (40 Lecturas):** [data/lecturas_ingles.csv](data/lecturas_ingles.csv)
- 🚀 **Orquestador Principal en Python:** [src/main.py](src/main.py)
- 🖼️ **Directorio de Visualizaciones Exportadas:** [graficos/](graficos/)

---

## 📁 Estructura del Repositorio

```text
IA_TutorIngles/
├── Dockerfile                  # Empaquetado oficial en Python 3.12-slim
├── docker-compose.yml          # Configuración del servicio contenedorizado
├── requirements.txt            # Dependencias del proyecto (numpy, pandas, matplotlib)
├── README.md                   # Documentación ejecutiva del repositorio
├── Informe_Corte_1.md          # Informe académico completo de primer corte
├── data/
│   └── lecturas_ingles.csv     # Dataset estandarizado (40 lecturas en inglés)
├── src/                        # Arquitectura Modular en Python
│   ├── cargar_datos.py         # Módulo 1: Ingesta de CSV y conversión a matrices NumPy
│   ├── eda_numpy.py            # Módulo 2: Estadísticas descriptivas vectorizadas
│   ├── visualizaciones.py      # Módulo 3: Generación de gráficos en Matplotlib
│   ├── generar_informe.py      # Módulo 4: Exportación del informe dinámico en Markdown
│   └── main.py                 # Orquestador central del sistema
├── graficos/                   # Visualizaciones exportadas en 300 DPI (.png)
│   ├── distribucion_cefr_palabras.png
│   ├── relacion_dificultad_extension.png
│   └── distribucion_tiempos_categoria.png
└── informes/                   # Carpeta de informes generados
    ├── Informe_Corte_1.md
    └── informe_proyecto.md
```

---

## 📊 Métricas del Análisis Exploratorio (EDA con NumPy)

| Variable Numérica | Media ($\mu$) | Mediana ($Q_2$) | Desviación Estándar ($\sigma$) | Mínimo | Máximo |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Conteo de Palabras** | 278.4 palabras | 240.0 palabras | 134.1 palabras | 85.0 palabras | 520.0 palabras |
| **Dificultad Léxica (1.0 - 5.0)** | 3.21 / 5.0 | 3.10 / 5.0 | 1.12 / 5.0 | 1.20 / 5.0 | 5.00 / 5.0 |
| **Tiempo de Lectura Estimado** | 3.10 min | 2.80 min | 1.50 min | 0.90 min | 6.10 min |

---

## 🖼️ Visualizaciones Estadísticas Exportadas

El sistema genera automáticamente 3 gráficos analíticos en la carpeta `graficos/`:

1. [graficos/distribucion_cefr_palabras.png](graficos/distribucion_cefr_palabras.png): Gráfico de barras con el promedio de palabras por nivel CEFR.
2. [graficos/relacion_dificultad_extension.png](graficos/relacion_dificultad_extension.png): Diagrama de dispersión (*Scatter Plot*) entre extensión y dificultad léxica.
3. [graficos/distribucion_tiempos_categoria.png](graficos/distribucion_tiempos_categoria.png): Diagrama de cajas (*Boxplot*) de tiempos estimados de lectura por categoría.

---

## 🛠️ Instrucciones para Clonar y Ejecutar el Proyecto

### 1. Clonación del Repositorio
```bash
git clone https://github.com/moliech/IA_TutorIngles.git
cd IA_TutorIngles
```

### 2. Opción A: Ejecución Directa con Python 3
```bash
# Instalar las librerías necesarias:
pip install -r requirements.txt

# Ejecutar el programa principal:
python3 src/main.py
```

### 3. Opción B: Ejecución Contenerizada con Docker (Recomendada)
```bash
# Construir y levantar el contenedor en segundo plano:
docker compose up -d --build

# Ejecutar el flujo principal dentro del contenedor:
docker exec -it ia-tutor-ingles python src/main.py
```
