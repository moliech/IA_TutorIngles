# 🧠 Informe Académico: Reflexión sobre el Aprendizaje de Neuronas Artificiales y Compuertas Lógicas

**Asignatura:** Inteligencia Artificial (Semestre 6)  
**Institución:** Corporación de Estudios Tecnológicos del Norte del Valle (COTECNOVA)  
**Docente:** Jhon James Cano Sánchez  
**Equipo:** Jhon Esteban Molina Echavarría, Heiber Lozano Mercado  
**Proyecto:** IA_TutorIngles  

---

## 📌 1. Respuestas a las Preguntas de la Actividad Independiente (Clase 8)

### Pregunta 1. ¿Cuántas épocas necesitó cada compuerta para aprender?
En el experimento ejecutado en `src/neuronas_compuertas.py` (con pesos inicializados en cero $\mathbf{w} = [0, 0]$, sesgo inicial $b = 0.0$ y tasa de aprendizaje $\eta = 0.1$), el número de épocas requeridas por cada modelo fue el siguiente:

| Compuerta / Modelo | Épocas Requeridas | Pesos Finales ($\mathbf{w}$) | Sesgo Final ($b$) | Resultado Experimental |
| :--- | :---: | :---: | :---: | :--- |
| **Compuerta NOT** | **3 épocas** | `[-0.1]` | `0.00` | Clasificó correctamente todas las entradas evaluadas |
| **Compuerta OR** | **4 épocas** | `[0.1, 0.1]` | `-0.10` | Clasificó correctamente todas las entradas evaluadas |
| **Compuerta AND** | **6 épocas** | `[0.2, 0.1]` | `-0.30` | Clasificó correctamente todas las entradas evaluadas |
| **Compuerta NAND** | **6 épocas** | `[-0.2, -0.1]` | `0.20` | Clasificó correctamente todas las entradas evaluadas |
| **Compuerta XOR** | **No converge (50/50 épocas)** | Oscilatorio | Oscilatorio | No clasificó las 4 entradas (Error residual = 4) |
| **Clasificador Tutor (`IA_TutorIngles`)** | **4 épocas** | `[0.0455, 0.0513]` | `-0.05` | Clasificó el 100 % de los datos de entrenamiento (40/40) |

> **Aclaración experimental sobre el Clasificador del Tutor:** Las 4 épocas y el 100 % de exactitud reportados corresponden al experimento realizado y fueron evaluados directamente sobre los datos de entrenamiento (`data/lecturas_ingles.csv`, con $\eta = 0.05$), no sobre un conjunto de prueba independiente.

---

### Pregunta 2. ¿Fue necesario ajustar la tasa de aprendizaje? ¿Por qué?
En las compuertas lógicas se utilizó una tasa de aprendizaje de $0.1$, con la cual se obtuvo convergencia en los experimentos realizados para los modelos que clasificaron correctamente todas las entradas (**NOT, OR, AND y NAND**). No fue necesario modificar este valor en esas pruebas.

Para el clasificador de `IA_TutorIngles` se utilizó una tasa de aprendizaje de $0.05$. Esta decisión buscó realizar ajustes más graduales en los parámetros del modelo a partir de las características normalizadas de las lecturas (`palabras_norm` y `dificultad_norm`). Para determinar si esta tasa es adecuada, es necesario observar cómo evoluciona el error durante el entrenamiento.

---

### Pregunta 3. ¿Qué diferencias encontró entre la compuerta AND y la OR?
La principal diferencia entre **AND** y **OR** está en la condición necesaria para producir una salida igual a $1$. En **AND**, ambas entradas deben ser $1$; en **OR**, basta con que al menos una entrada sea $1$.

Esta diferencia se refleja en los pesos y el sesgo que permiten establecer la frontera de decisión ($w_1 x_1 + w_2 x_2 + b = 0$) de cada compuerta:
- En **AND** ($\mathbf{w} = [0.2, 0.1], b = -0.30$), la combinación de parámetros debe impedir que una sola entrada activa produzca una salida positiva: para $(0,1)$ se obtiene $z = -0.20 < 0$ y para $(1,0)$ se obtiene $z = -0.10 < 0$; solo cuando ambas entradas son $1$ se alcanza $z = 0.2 + 0.1 - 0.30 = 0 \ge 0$ (salida $1$).
- En **OR** ($\mathbf{w} = [0.1, 0.1], b = -0.10$), la combinación de parámetros debe permitir que cualquiera de las dos entradas activas produzca esa salida: tanto $(0,1)$ como $(1,0)$ generan $z = 0.0 \ge 0$ (salida $1$), dejando únicamente a $(0,0)$ con $z = -0.10 < 0$ (salida $0$).

En los experimentos realizados, **OR** necesitó cuatro épocas y **AND** necesitó seis épocas según los resultados registrados. Estos números corresponden a la configuración utilizada y no implica que siempre necesiten la misma cantidad de épocas.

---

### Pregunta 4. ¿Por qué la compuerta XOR no puede ser aprendida por una sola neurona?
La compuerta **XOR** no puede aprenderse correctamente con una sola neurona artificial de tipo perceptrón porque sus clases **no son linealmente separables**. Las entradas que producen salida $1$ son $(0,1)$ y $(1,0)$, mientras que las que producen salida $0$ son $(0,0)$ y $(1,1)$.

Una sola neurona con función de activación escalón establece una frontera lineal ($w_1 x_1 + w_2 x_2 + b = 0$) que no puede separar correctamente estas dos clases dispuestas en esquinas opuestas. Esta limitación del perceptrón simple puede superarse mediante una **red neuronal multicapa (MLP)**, que permite combinar varias neuronas para resolver problemas que una sola neurona no puede clasificar correctamente.

---

## 📌 2. Relación y Aplicación al Proyecto (`IA_TutorIngles`)

En el clasificador adaptativo implementado en `src/clasificador_neurona_tutor.py`, la neurona entrenada sobre las características normalizadas (*Min-Max* en $[0, 1]$) obtuvo los parámetros finales $w_1 = 0.0455$ (conteo de palabras), $w_2 = 0.0513$ (dificultad léxica) y $b = -0.05$:

- **Interpretación de los pesos ($w_1$ y $w_2$):** Ambos pesos son positivos, lo que indica que a medida que aumentan tanto la extensión del texto como su complejidad léxica, mayor es la tendencia a clasificar la lectura en el nivel **Avanzado (B1/B2)**. Aunque el peso de la dificultad léxica es ligeramente mayor, esto no significa necesariamente que sea la variable más importante, ya que también influyen la escala de los datos y su distribución.
- **Interpretación del sesgo ($b = -0.05$):** Actúa como el umbral matemático de decisión del clasificador ($w_1 x_1 + w_2 x_2 \ge 0.05$) que divide las lecturas de nivel **Básico (A1/A2)** de las de nivel **Avanzado (B1/B2)** dentro de los datos evaluados.
