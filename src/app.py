"""
Aplicación Principal Streamlit - LexiaAdapt (IA_TutorIngles)
Prototipo Interactivo de Lectura Adaptativa e Inferencia en Vivo
Corte 3 - Inteligencia Artificial
"""

import os
import pandas as pd
import streamlit as st
from ui_theme import aplicar_tema_lexia, render_badge
from usar_modelo import predecir_lectura

st.set_page_config(
    page_title="LexiaAdapt | Tutor Inteligente de Inglés",
    page_icon="📖",
    layout="wide"
)

# Aplicar estilos y tokens visuales ergonómicos
aplicar_tema_lexia()

@st.cache_data
def cargar_catalogo():
    ruta = os.path.join(os.path.dirname(__file__), "..", "data", "proyecto_limpio.csv")
    if not os.path.exists(ruta):
        ruta_alt = os.path.join(os.path.dirname(__file__), "data", "proyecto_limpio.csv")
        if os.path.exists(ruta_alt):
            ruta = ruta_alt
    return pd.read_csv(ruta)

df_lecturas = cargar_catalogo()

st.markdown("<h1 style='color:#0F2942; margin-bottom:0;'>LexiaAdapt</h1>", unsafe_allow_html=True)
st.caption("Tutor Inteligente de Lectura Adaptativa en Inglés • Cartago, Valle del Cauca")

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
            <div class="reading-container">
                <p class="reading-body">{texto_mostrar}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_der:
        st.markdown("#### Métricas del Texto")
        st.metric("Conteo de Palabras", f"{fila['conteo_palabras']} palabras")
        st.metric("Dificultad Léxica", f"{fila['dificultad_lexica']} / 5.0")
        st.metric("Tiempo Estimado", f"{fila['tiempo_lectura_est_min']} min")
        st.metric("Velocidad Promedio", f"{fila['velocidad_lectura_ppm']} PPM")

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
            st.warning("⚠️ **Modelo serializado no encontrado:** Para ejecutar la inferencia en vivo, debe entrenarse primero el modelo ejecutando `python src/modelo_proyecto.py`.")
