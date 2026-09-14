# src/main.py (VERSIÓN 2: INTEGRACIÓN COMPLETA - ESTEBAN & HEIBER)
import sys
import os

# Asegurar que la carpeta 'src' esté en el path de Python
DIR_SRC = os.path.dirname(os.path.abspath(__file__))
if DIR_SRC not in sys.path:
    sys.path.insert(0, DIR_SRC)

from cargar_datos import cargar_lecturas
from eda_numpy import calcular_estadisticas_numpy
from generar_informe import crear_informe_markdown

def main():
    print("=" * 60)
    print("🚀 PROYECTO IA_TUTORINGLES - SISTEMA DE ANÁLISIS EXPLORATORIO")
    print("   [VERSIÓN 2: INTEGRACIÓN COMPLETA - ESTEBAN & HEIBER]")
    print("=" * 60)

    # 1. Carga de Datos (Esteban)
    print("\n[1/4] Cargando dataset 'data/lecturas_ingles.csv'...")
    datos_completos, array_numerico = cargar_lecturas()
    print(f" -> Se cargaron exitosamente {len(datos_completos)} registros.")

    # 2. Análisis Estadístico con NumPy (Esteban)
    print("\n[2/4] Calculando estadísticas descriptivas vectorizadas con NumPy...")
    estadisticas = calcular_estadisticas_numpy(array_numerico)
    print(" -> Estadísticas procesadas correctamente con NumPy.")

    # 3. Visualizaciones Estadísticas con Matplotlib (Heiber Lozano)
    print("\n[3/4] Generando visualizaciones gráficas en Matplotlib...")
    try:
        import visualizaciones
        print(" -> Gráficos PNG exportados exitosamente en 'graficos/'.")
    except Exception as e:
        print(f" -> [AVISO] Ocurrió una advertencia en el módulo de visualizaciones: {e}")

    # 4. Exportación de Informe Markdown (Esteban)
    print("\n[4/4] Exportando informe técnico Markdown...")
    crear_informe_markdown(estadisticas)

    print("\n" + "=" * 60)
    print("✅ VERSIÓN 2 INTEGRADA COMPLETADA SATISFACTORIAMENTE")
    print("=" * 60)

if __name__ == "__main__":
    main()
