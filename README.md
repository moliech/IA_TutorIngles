# 📚 IA_TutorIngles - Tutor Inteligente de Lectura y Comprensión en Inglés

**Asignatura:** Inteligencia Artificial (Semestre 6)  
**Institución:** Corporación de Estudios Tecnológicos del Norte del Valle (COTECNOVA)  
**Docente:** Jhon James Cano Sánchez  
**Equipo de Trabajo:**  
- **Jhon Esteban Molina Echavarría**  
- **Heiber Lozano Mercado**  

---

## 📌 Estructura del Proyecto

```
IA_TutorIngles/
├── Dockerfile                  # Imagen Docker con Python 3.12-slim
├── docker-compose.yml          # Configuración del servicio ia-tutor-ingles
├── requirements.txt            # Dependencias (numpy, pandas, matplotlib, scikit-learn, jupyter)
├── README.md                   # Documentación ejecutiva del proyecto
├── data/                       # Dataset del proyecto (.csv)
│   └── lecturas_ingles.csv     # Dataset base de lecturas CEFR
├── src/                        # Scripts en Python (Arquitectura Modular)
│   ├── cargar_datos.py         # Módulo 1: Carga y validación de datos
│   ├── eda_numpy.py            # Módulo 2: Análisis estadístico con NumPy
│   ├── visualizaciones.py      # Módulo 3: Generación de gráficos con Matplotlib
│   ├── generar_informe.py      # Módulo 4: Exportación de informe Markdown
│   └── main.py                 # Orquestador central del sistema
├── graficos/                   # Gráficos generados (.png)
└── informes/                   # Informes técnicos en Markdown
```

---

## 🚀 Ejecución del Proyecto

### Opción 1: Ejecución con Docker (Recomendada)
```bash
# 1. Levantar el contenedor
docker compose up -d

# 2. Ejecutar el orquestador principal
docker exec -it ia-tutor-ingles python src/main.py
```

### Opción 2: Ejecución local en Python
```bash
python src/main.py
```
