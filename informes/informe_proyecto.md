# 📊 Informe de Análisis Exploratorio de Datos (EDA) — IA Tutor de Inglés

## 1. Resumen Ejecutivo
El presente informe documenta el análisis cuantitativo del corpus de lecturas diseñado para el **Tutor Inteligente de Inglés**. Las métricas extraídas sustentan la graduación pedagógica de los contenidos de acuerdo con el **Marco Común Europeo de Referencia para las Lenguas (CEFR)**.

- **Total de lecturas analizadas:** 40 textos
- **Promedio de extensión:** 278.4 palabras por lectura
- **Índice promedio de dificultad léxica:** 3.21 (escala 0.0 - 1.0)
- **Tiempo estimado de lectura promedio:** 3.2 minutos

## 2. Distribución Cuantitativa por Nivel CEFR
La graduación del contenido garantiza un incremento progresivo en la extensión y complejidad textual a medida que se avanza en la escala del marco europeo:

| Nivel CEFR | Cantidad de Lecturas | Extensión Promedio (palabras) | Dificultad Léxica Promedio | Tiempo Estimado (min) |
| :---: | :---: | :---: | :---: | :---: |
| **A1** | 10 | 112.1 | 1.63 | 1.2 |
| **A2** | 10 | 201.0 | 2.76 | 2.4 |
| **B1** | 10 | 327.0 | 3.76 | 3.8 |
| **B2** | 10 | 473.5 | 4.69 | 5.5 |

## 3. Distribución por Categoría Temática
La diversificación temática del dataset permite ofrecer lecturas contextualizadas según los intereses de los estudiantes:

| Categoría Temática | Cantidad de Textos | Extensión Promedio (palabras) | Tiempo Estimado (min) |
| :--- | :---: | :---: | :---: |
| Ciencia | 9 | 336.1 | 3.9 |
| Cotidiano | 6 | 109.7 | 1.2 |
| Cultura | 6 | 275.8 | 3.2 |
| Estilo de Vida | 6 | 155.5 | 1.8 |
| Negocios | 5 | 357.0 | 4.1 |
| Tecnología | 8 | 385.0 | 4.5 |

## 4. Visualizaciones Analíticas del Sistema

### Figura 1: Distribución del Conteo Promedio de Palabras por Nivel CEFR

![Figura 1: Distribución del Conteo Promedio de Palabras por Nivel CEFR](../graficos/distribucion_cefr_palabras.png?raw=true)

_Muestra cómo la extensión de los textos aumenta de forma progresiva desde el nivel A1 hasta el nivel C2._

### Figura 2: Relación entre la Extensión del Texto y la Dificultad Léxica

![Figura 2: Relación entre la Extensión del Texto y la Dificultad Léxica](../graficos/relacion_dificultad_extension.png?raw=true)

_Evidencia la correlación entre el volumen de palabras y el índice de dificultad léxica del vocabulario._

### Figura 3: Distribución de Tiempos Estimados de Lectura por Categoría Temática

![Figura 3: Distribución de Tiempos Estimados de Lectura por Categoría Temática](../graficos/distribucion_tiempos_categoria.png?raw=true)

_Diagrama de cajas (Boxplot) que compara la dispersión y tiempos promedios requeridos para completar lecturas según su área temática._

## 5. Conclusiones y Diagnóstico Pedagógico
1. **Graduación Adecuada:** El análisis valida que el dataset presenta una progresión coherente tanto en longitud como en sofisticación léxica.
2. **Soporte para Tutoría Personalizada:** La estructura multivariada sienta las bases cuantitativas para la posterior clasificación de textos mediante algoritmos de Machine Learning.

---
*Informe generado automáticamente por el módulo `src/generar_informe.py` del proyecto IA Tutor de Inglés.*