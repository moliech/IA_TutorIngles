"""
Aplicación Principal Streamlit - LexiaAdapt (IA_TutorIngles)
Prototipo Interactivo de Lectura Adaptativa e Inferencia en Vivo
Corte 3 - Inteligencia Artificial
"""

import os
import pandas as pd
import streamlit as st

# Importación condicional del módulo de diseño visual de la rama heiber
try:
    from ui_theme import aplicar_tema_lexia, render_badge
except ImportError:
    def aplicar_tema_lexia():
        pass

    def render_badge(nivel: str, confianza: float = None):
        colores = {
            "A1": "#065F46",
            "A2": "#0F5B54",
            "B1": "#78350F",
            "B2": "#312E81"
        }
        fondos = {
            "A1": "#D1FAE5",
            "A2": "#CCFBF1",
            "B1": "#FEF3C7",
            "B2": "#E0E7FF"
        }
        lvl = str(nivel).upper()
        c = colores.get(lvl, "#0F172A")
        bg = fondos.get(lvl, "#E2E8F0")
        conf_txt = f" • {int(confianza * 100)}% IA" if confianza is not None else ""
        return (
            f'<span style="display:inline-block; padding:4px 12px; border-radius:9999px; '
            f'background-color:{bg}; color:{c}; font-weight:600; font-size:12px; font-family:monospace;">'
            f'MCER {lvl}{conf_txt}</span>'
        )

from usar_modelo import predecir_lectura

st.set_page_config(
    page_title="LexiaAdapt | Tutor Inteligente de Inglés",
    page_icon="📖",
    layout="wide"
)

# Aplicar tokens visuales si el módulo ui_theme está presente
aplicar_tema_lexia()

# Carga del catálogo de lecturas limpias
@st.cache_data
def cargar_catalogo():
    ruta = os.path.join(os.path.dirname(__file__), "..", "data", "proyecto_limpio.csv")
    if not os.path.exists(ruta):
        ruta_alt = os.path.join(os.path.dirname(__file__), "data", "proyecto_limpio.csv")
        if os.path.exists(ruta_alt):
            ruta = ruta_alt
    return pd.read_csv(ruta)

df_lecturas = cargar_catalogo()

# Encabezado institucional
st.markdown("<h1 style='color:#0F2942; margin-bottom:0;'>LexiaAdapt</h1>", unsafe_allow_html=True)
st.caption("Tutor Inteligente de Lectura Adaptativa en Inglés • Cartago, Valle del Cauca")

# Selector de Modo Superior
modo = st.radio(
    "Seleccione el Modo de Operación:",
    ["📖 Modo Estudiante (Catálogo Precargado)", "🧠 Modo Auditoría IA (Simulador de Sustentación)"],
    horizontal=True
)

st.divider()

if "Modo Estudiante" in modo:
    # ------------------ VISTA ESTUDIANTE ------------------
    st.subheader("📚 Biblioteca de Lecturas Graduadas")
    st.write("Seleccione el área temática y el texto para iniciar la lectura adaptada a su nivel:")

    col_filtro1, col_filtro2 = st.columns([1, 2])
    with col_filtro1:
        categorias = ["Todas"] + sorted(df_lecturas["categoria"].dropna().unique().tolist())
        cat_sel = st.selectbox("Filtrar por Área Temática:", categorias)

    df_filtrado = df_lecturas if cat_sel == "Todas" else df_lecturas[df_lecturas["categoria"] == cat_sel]

    with col_filtro2:
        titulos = df_filtrado["titulo"].tolist()
        titulo_sel = st.selectbox("Seleccione la Lectura:", titulos)

    fila = df_filtrado[df_filtrado["titulo"] == titulo_sel].iloc[0]

    st.write("")
    col_izq, col_der = st.columns([2, 1], gap="large")

    with col_izq:
        st.markdown(f"### {fila['titulo']}")
        st.markdown(render_badge(fila["nivel_cefr"]), unsafe_allow_html=True)
        st.write("")

        texto_mostrar = fila.get("texto_ingles", "Cuerpo de lectura precargado en proceso de sincronización.")
        st.markdown(
            f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px; padding:2rem 2.5rem; box-shadow:0 4px 15px -2px rgba(15,41,66,0.05); max-width:72ch;">
                <p style="font-family:'Lexend', sans-serif, -apple-system; font-size:17px; line-height:1.85; color:#0F172A; letter-spacing:0.02em; margin:0;">
                    {texto_mostrar}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_der:
        st.markdown("#### Métricas Lingüísticas")
        st.metric("Conteo de Palabras", f"{fila['conteo_palabras']} palabras")
        st.metric("Dificultad Léxica", f"{fila['dificultad_lexica']} / 5.0")
        st.metric("Tiempo Estimado de Lectura", f"{fila['tiempo_lectura_est_min']} min")
        st.metric("Velocidad de Lectura Promedio", f"{fila['velocidad_lectura_ppm']} PPM")

        st.info("💡 **Recomendación Pedagógica:** Este texto ha sido calibrado por los modelos supervisados de Inteligencia Artificial para evitar sobrecarga cognitiva en lectores del Valle del Cauca.")

else:
    # ------------------ VISTA AUDITORÍA IA ------------------
    st.subheader("🔬 Auditoría de Inferencia y Demostración en Vivo")
    st.write("Ingrese parámetros de una lectura no vista para auditar la respuesta del modelo entrenado:")

    c1, c2 = st.columns(2)
    with c1:
        p_palabras = st.slider("Conteo de Palabras:", min_value=50, max_value=600, value=310, step=10)
    with c2:
        p_dif = st.slider("Dificultad Léxica (1.0 a 5.0):", min_value=1.0, max_value=5.0, value=3.2, step=0.1)

    if st.button("Ejecutar Inferencia (usar_modelo.py)", type="primary"):
        try:
            res = predecir_lectura(p_palabras, p_dif)

            st.success(f"Nivel Predicho: **{res['nivel']}** (Certeza: {res['confianza']*100:.1f}%)")

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Probabilidad Nivel Básico (A1/A2)", f"{res['probabilidad_basico']*100:.2f}%")
                st.progress(res["probabilidad_basico"])
            with col_m2:
                st.metric("Probabilidad Nivel Avanzado (B1/B2)", f"{res['probabilidad_avanzado']*100:.2f}%")
                st.progress(res["probabilidad_avanzado"])

            st.info(f"Tiempo estimado: {res['tiempo_estimado_min']} min a una velocidad promedio de {res['ppm_calculado']} PPM.")
        except FileNotFoundError:
            st.warning("⚠️ **Modelo serializado no encontrado:** Para ejecutar la inferencia en vivo, debe entrenarse primero el modelo en Scikit-Learn ejecutando `python src/modelo_proyecto.py` (correspondiente a la rama `heiber`).")
