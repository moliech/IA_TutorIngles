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
    page_title="lexiaAdapt | Tutor Inteligente de Ingles",
    page_icon="book",
    layout="wide"
)


aplicar_tema_lexia()
@st.cache_data
def cargar_catalogo():
    ruta = os.path.join(os.path.dirname(__file__), "..", "data", "proyecto_limpio.csv")
    return pd.read_csv(ruta)


df_lecturas = cargar_catalogo()

st.markdown("<h1 style='color:#0f2942; margin-bottom:0;'>lexiaAdapt</h1>", unsafe_allow_html=True)
st.caption("Tutor inteligente de lectura Adaptativa en ingles Cartago, Valle del Cauca")

modo = st.radio(
    "Seleccione el modo de Operacion:",
    [" Modo Estudiante (Catalogo Precargado)", " Modo Auditoria IA(Simulador de Sustentacion)"],
    horizontal=True
)

st.divider()

if "Modo Estudiante" in modo:
    st.subheader("Biblioteca de lecturas graduadas")
    st.write("Seleccione el tema de su interes para iniciar la lectura adaptada a su nivel:")

    categorias = ["Todas"] + sorted(df_lecturas["categoria"].unique().tolist())
    cat_sel = st.selectbox("Filtrar por Area Tematica:", categorias)

    df_filtrado = df_lecturas if cat_sel == "Todas" else df_lecturas[df_lecturas["categoria"] == cat_sel]

    titulos = df_filtrado["titulo"].tolist()
    titulo_sel = st.selectbox("seleccione la lectura:", titulos)

    fila = df_filtrado[df_filtrado["titulo"] == titulo_sel].iloc[0]

    col_izq, col_der = st.columns([2, 1])

    with col_izq:
        st.markdown(f"### {fila['titulo']}")
        st.markdown(render_badge(fila['nivel_cefr']), unsafe_allow_html=True)
        st.write("")

        texto_mostrar = fila.get("texto_ingles", "cuerpo de lectura precargado en proceso de sincronizacion.")
        st.markdown(
            f"""
            <div class="reading-container">
                 <p class="reading-body">{texto_mostrar}</p>
            </div>
            """,
            unsafe_allow_html=True
        ) 


    with col_der:
        st.markdown("### Metricas del texto")
        st.metric("Conteo de palabras", f"{fila['conteo_palabras']} palabras")
        st.metric("Dificultad lexica", f"{fila['dificultad_lexica']} / 5.0")
        st.metric("Tiempo Estimado", f"{fila['tiempo_lectura_est_min']} min")
        st.metric("Velocidad Promedio", f"{fila['velocidad_lectura_ppm']} PPM")

else:
    st.subheader("Auditoria de Inferencia y Demostracion en Vivo")
    st.write("Ingrese parametros de una lectura no vista para auditar la respuesta del modelo entrenado:")

    c1, c2 = st.columns(2)
    with c1:
        p_palabras = st.slider("Conteo de Palabras:", min_value=50, max_value=600, value=310, step=10)
    with c2:
        p_dif = st.slider("Dificultad lexica (1.0 a 5.0):", min_value=1.0, max_value=5.0, value=3.2, step=0.1)

    if st.button("Ejecutar Inferencia (Usar_modelo.py)", type="primary"):
        res = predecir_lectura(p_palabras, p_dif)

        st.success(f"Nivel Predicho:  **{res['nivel']}** (Certeza: {res['confianza']*100:.1f}%)")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Probabilidad Nivel Basico (A1/A2)", f"{res['probabilidad_basico']*100:.2f}%")
            st.progress(res["probabilidad_basico"])
        with col_m2:
            st.metric("Probabilidad nivel avanzado (B1/B2)", f"{res['probabilidad_avanzado']*100:.2f}%")
            st.progress(res["probabilidad_avanzado"])
      
        st.info(f"Tiempo estimado: {res['tiempo_estimado_min']}  min a una tasa de {res['ppm_calculado']} PPM.")
