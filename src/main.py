# src/main.py (VERSIÓN 1: MÓDULOS DE JHON ESTEBAN MOLINA - RAMA MAIN)
import sys
import os

# Asegurar que la carpeta 'src' esté en el path de Python
DIR_SRC = os.path.dirname(os.path.abspath(__file__))
if DIR_SRC not in sys.path:
    sys.path.insert(0, DIR_SRC)

# Importaciones dinámicas a prueba de fallas
try:
    from cargar_datos import cargar_lecturas
    from eda_numpy import calcular_estadisticas_numpy
    from generar_informe import crear_informe_markdown
except ImportError:
    from src.cargar_datos import cargar_lecturas
    from src.eda_numpy import calcular_estadisticas_numpy
    from src.generar_informe import crear_informe_markdown

def main():
    print("=" * 60)
    print("🚀 PROYECTO IA_TUTORINGLES - SISTEMA DE ANÁLISIS EXPLORATORIO")
    print("   [VERSIÓN 1: MÓDULOS DE JHON ESTEBAN MOLINA - RAMA MAIN]")
    print("=" * 60)

    # 1. Carga de Datos
    print("\n[1/3] Cargando dataset 'data/lecturas_ingles.csv'...")
    datos_completos, array_numerico = cargar_lecturas()
    print(f" -> Se cargaron exitosamente {len(datos_completos)} registros.")

    # 2. Análisis Estadístico con NumPy
    print("\n[2/3] Calculando estadísticas descriptivas vectorizadas con NumPy...")
    estadisticas = calcular_estadisticas_numpy(array_numerico)
    print(" -> Estadísticas procesadas correctamente con NumPy.")

    # 3. Exportación de Informe Markdown
    print("\n[3/3] Exportando informe técnico Markdown...")
    crear_informe_markdown(estadisticas)

    print("\n" + "=" * 60)
    print("✅ VERSIÓN 1 COMPLETADA SATISFACTORIAMENTE")
    print("=" * 60)

if __name__ == "__main__":
    main()
