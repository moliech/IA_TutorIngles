"""
src/app.py

Aplicación Principal Streamlit - LexiaAdapt (IA_TutorIngles)
Plataforma de Lectura Adaptativa e Inferencia en Vivo
Corte 3 - Inteligencia Artificial
"""

import os
import re
import pandas as pd
import streamlit as st

# Importación de módulos internos del sistema
from ui_theme import aplicar_tema_lexia, render_badge
from usar_modelo import predecir_lectura
from gemini_tutor import (
    hay_ia_generativa_disponible,
    explicar_prediccion_ia,
    responder_duda_estudiante,
    generar_resumen_espanol,
)

st.set_page_config(
    page_title="LexiaAdapt | Tutor Inteligente de Inglés",
    page_icon="📖",
    layout="wide"
)

# Inyección de estilos y tokens visuales ergonómicos (Google Fonts Lexend)
aplicar_tema_lexia()

# ------------------ CARGA DE DATOS ------------------
@st.cache_data
def cargar_catalogo():
    ruta = os.path.join(os.path.dirname(__file__), "..", "data", "proyecto_limpio.csv")
    if not os.path.exists(ruta):
        ruta = os.path.join(os.path.dirname(__file__), "data", "proyecto_limpio.csv")
    return pd.read_csv(ruta)

df_lecturas = cargar_catalogo()

# ------------------ ANALIZADOR DE TEXTO LIBRE ------------------
def estimar_metricas_texto(texto: str):
    """
    Extrae variables cuantitativas de cualquier texto en inglés
    y calibra la dificultad léxica sobre el corpus oficial del proyecto.
    """
    palabras = re.findall(r"\b[a-zA-Z]+\b", texto)
    n = len(palabras)
    if n < 5:
        return max(1, n), 1.2, 0.5, 85.0

    avg_len = sum(len(w) for w in palabras) / n
    long_w = sum(1 for w in palabras if len(w) >= 7) / n
    ttr = len(set(w.lower() for w in palabras)) / n

    # Regresión calibrada sobre el corpus limpio oficial (R2 = 0.85)
    dif = (0.01258 * n) + (0.51191 * avg_len) - (0.11259 * long_w) - (0.3062 * ttr) - 1.6466
    dificultad = round(max(1.0, min(5.0, dif)), 2)
    tiempo = round(max(0.5, n / 88.0), 1)
    ppm = round(n / tiempo, 1)
    return n, dificultad, tiempo, ppm

# ------------------ ENCABEZADO ------------------
st.markdown("<h1 style='color:#0F2942; margin-bottom:0;'>LexiaAdapt</h1>", unsafe_allow_html=True)
st.caption("Tutor Inteligente de Lectura Adaptativa en Inglés • Cartago, Valle del Cauca")

modo = st.radio(
    "Seleccione el Modo de Operación:",
    ["📖 Modo Estudiante (Lectura & Tutoría)", "🧠 Modo Auditoría IA (Simulador con Texto Libre)"],
    horizontal=True
)

st.divider()

# =========================================================================
# VISTA 1: MODO ESTUDIANTE (LECTURA GUIADA Y ASISTENCIA INTERACTIVA)
# =========================================================================
if "Modo Estudiante" in modo:
    st.subheader("📚 Biblioteca de Lecturas Graduadas")
    st.write("Seleccione el área temática y el texto para iniciar su sesión de lectura:")

    c_f1, c_f2 = st.columns([1, 2])
    with c_f1:
        categorias = ["Todas"] + sorted(df_lecturas["categoria"].dropna().unique().tolist())
        cat_sel = st.selectbox("Área Temática:", categorias)

    df_filtrado = df_lecturas if cat_sel == "Todas" else df_lecturas[df_lecturas["categoria"] == cat_sel]

    with c_f2:
        titulos = df_filtrado["titulo"].tolist()
        titulo_sel = st.selectbox("Seleccione la Lectura:", titulos)

    fila = df_filtrado[df_filtrado["titulo"] == titulo_sel].iloc[0]

    # Controles de accesibilidad del texto
    tam_fuente = st.slider("Tamaño de Fuente para Lectura:", min_value=16, max_value=24, value=18, step=1)

    st.write("")
    col_lectura, col_metricas = st.columns([2, 1], gap="large")

    with col_lectura:
        st.markdown(f"### {fila['titulo']}")
        st.markdown(render_badge(fila["nivel_cefr"]), unsafe_allow_html=True)
        st.write("")

        texto_mostrar = fila.get("texto_ingles", "Texto en proceso de carga.")
        st.markdown(
            f"""
            <div class="reading-container">
                <p class="reading-body" style="font-size:{tam_fuente}px !important;">{texto_mostrar}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")
        # --- APOYO PEDAGÓGICO: RESUMEN GUIADO EN ESPAÑOL ---
        clave_resumen = f"resumen_{fila['id']}"
        with st.expander("💡 ¿No entendiste la idea general? Ver Resumen en Español"):
            if st.button("Generar Resumen Guiado", key="btn_resumen"):
                with st.spinner("El tutor está extrayendo los puntos clave..."):
                    resumen = generar_resumen_espanol(fila["titulo"], texto_mostrar)
                    st.session_state[clave_resumen] = resumen
            
            if clave_resumen in st.session_state:
                st.markdown(st.session_state[clave_resumen])

        # --- ASISTENTE INTERACTIVO / CHAT CON EL TUTOR IA ---
        st.markdown("#### 💬 Pregúntale a tu Tutor IA sobre este texto")
        st.caption("Escribe tus dudas sobre palabras difíciles, ideas principales o gramática de esta lectura.")

        # Botones de preguntas rápidas sugeridas
        b1, b2, b3 = st.columns(3)
        prompt_rapido = None
        if b1.button("🔑 ¿3 palabras clave y significado?", use_container_width=True):
            prompt_rapido = "¿Cuáles son las 3 palabras más importantes o difíciles de este texto y qué significan en español?"
        if b2.button("📝 Explícame la idea en 2 frases", use_container_width=True):
            prompt_rapido = "Explícame de forma muy sencilla en dos frases en español de qué trata este texto."
        if b3.button("❓ Hazme una pregunta de comprensión", use_container_width=True):
            prompt_rapido = "Hazme una pregunta sencilla en inglés sobre este texto para ver si lo comprendí."

        # Estado de sesión para el historial del chat
        clave_chat = f"chat_{fila['id']}"
        if clave_chat not in st.session_state:
            st.session_state[clave_chat] = []

        # Mostrar historial
        for msj in st.session_state[clave_chat]:
            with st.chat_message(msj["rol"]):
                st.write(msj["texto"])

        # Input de chat
        input_usuario = st.chat_input("Escribe tu pregunta sobre la lectura...")
        pregunta_final = prompt_rapido or input_usuario

        if pregunta_final:
            st.session_state[clave_chat].append({"rol": "user", "texto": pregunta_final})
            with st.chat_message("user"):
                st.write(pregunta_final)

            with st.chat_message("assistant"):
                with st.spinner("LexiaTutor está pensando..."):
                    respuesta = responder_duda_estudiante(pregunta_final, fila["titulo"], texto_mostrar)
                    st.write(respuesta)
                    st.session_state[clave_chat].append({"rol": "assistant", "texto": respuesta})

    with col_metricas:
        st.markdown("#### Métricas Lingüísticas")
        st.metric("Conteo de Palabras", f"{fila['conteo_palabras']} palabras")
        st.metric("Dificultad Léxica", f"{fila['dificultad_lexica']} / 5.0")
        st.metric("Tiempo Estimado", f"{fila['tiempo_lectura_est_min']} min")
        st.metric("Velocidad Promedio", f"{fila['velocidad_lectura_ppm']} PPM")

        st.info("🎯 **Calibración CEFR:** Este texto ha sido graduado para que el vocabulario y la sintaxis se ajusten a las capacidades de lectura progresiva de estudiantes del Valle del Cauca.")

# =========================================================================
# VISTA 2: MODO AUDITORÍA IA (SIMULADOR EN VIVO CON TEXTO LIBRE Y SCIKIT-LEARN)
# =========================================================================
else:
    st.subheader("🔬 Auditoría de Inferencia y Demostración en Vivo")
    st.write(
        "Pega cualquier fragmento de lectura en inglés para que la Inteligencia Artificial analice sus "
        "propiedades lingüísticas en tiempo real y clasifique su nivel según el modelo entrenado:"
    )

    tab_texto, tab_sliders = st.tabs(["📝 Analizador de Texto Libre (Recomendado)", "🎛️ Simulador Manual por Parámetros"])

    # ------------------ PESTAÑA 1: CAJA DE TEXTO LIBRE ------------------
    with tab_texto:
        # Estado de sesión para persistir el texto y resultados
        if "texto_auditoria" not in st.session_state:
            st.session_state["texto_auditoria"] = ""
        if "analisis_actual" not in st.session_state:
            st.session_state["analisis_actual"] = None

        # Botones de ejemplos pre-cargados para la demostración rápida ante el evaluador
        c_ej1, c_ej2, c_ej3 = st.columns(3)
        if c_ej1.button("Cargar Ejemplo Básico (A1 - Rutina)", use_container_width=True):
            st.session_state["texto_auditoria"] = (
                "I wake up every day at six in the morning. First, I wash my face and brush my teeth. "
                "Then, I go to the kitchen to prepare breakfast. I usually eat scrambled eggs with bread "
                "and drink black coffee. After eating, I walk to university with my friends."
            )
            st.session_state["analisis_actual"] = None
            st.rerun()

        if c_ej2.button("Cargar Ejemplo Intermedio (B1 - Tecnología)", use_container_width=True):
            st.session_state["texto_auditoria"] = (
                "The unprecedented expansion of social media platforms has profoundly reshaped how modern youth "
                "communicate, consume information, and build personal relationships. Digital networks provide "
                "opportunities for teenagers to express creativity, but compulsive consumption often triggers "
                "social comparisons and heightened anxiety."
            )
            st.session_state["analisis_actual"] = None
            st.rerun()

        if c_ej3.button("Cargar Ejemplo Avanzado (B2 - Inteligencia Artificial)", use_container_width=True):
            st.session_state["texto_auditoria"] = (
                "The integration of artificial intelligence and machine learning algorithms into contemporary "
                "biomedical research is spearheading a monumental paradigm shift. Deep convolutional neural networks "
                "can analyze medical imaging with diagnostic sensitivity matching specialized radiologists, processing "
                "genomic sequencing profiles to detect early-stage malignant tumors before macroscopic symptoms emerge."
            )
            st.session_state["analisis_actual"] = None
            st.rerun()

        texto_ingresado = st.text_area(
            "Pega o escribe aquí el texto en inglés a clasificar:",
            value=st.session_state["texto_auditoria"],
            height=160,
            placeholder="Paste any English reading passage here...",
            key="input_area_texto"
        )
        st.session_state["texto_auditoria"] = texto_ingresado

        if st.button("🚀 Analizar Texto y Predecir Nivel con IA", type="primary", key="btn_analizar_texto"):
            if not texto_ingresado.strip():
                st.warning("Por favor ingrese o pegue un texto para analizar.")
                st.session_state["analisis_actual"] = None
            else:
                # 1. Extracción de métricas lingüísticas en milisegundos
                palabras, dif_calculada, t_est, ppm_calc = estimar_metricas_texto(texto_ingresado)
                try:
                    res = predecir_lectura(palabras, dif_calculada, t_est)
                    st.session_state["analisis_actual"] = {
                        "texto": texto_ingresado,
                        "palabras": palabras,
                        "dificultad": dif_calculada,
                        "tiempo": t_est,
                        "ppm": ppm_calc,
                        "res": res,
                        "explicacion": None
                    }
                except FileNotFoundError:
                    st.error("No se encontró el modelo `modelo_final.joblib`. Ejecute `python src/modelo_proyecto.py` para entrenarlo.")

        # Si hay análisis activo, renderizar métricas, veredicto y explicación
        if st.session_state["analisis_actual"]:
            analisis = st.session_state["analisis_actual"]
            res = analisis["res"]

            st.write("")
            st.markdown("##### 1. Métricas Extraídas por el Procesador Lingüístico:")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Palabras Detectadas", f"{analisis['palabras']}")
            m2.metric("Dificultad Calculada", f"{analisis['dificultad']} / 5.0")
            m3.metric("Tiempo Estimado", f"{analisis['tiempo']} min")
            m4.metric("Tasa Estimada", f"{analisis['ppm']} PPM")

            st.write("")
            st.markdown("##### 2. Veredicto del Modelo Scikit-Learn (Joblib):")
            if res["clase_binaria"] == 1:
                st.success(f"🎯 **Nivel Clasificado: {res['nivel']}** (Certeza: {res['confianza']*100:.1f}%)")
            else:
                st.info(f"🎯 **Nivel Clasificado: {res['nivel']}** (Certeza: {res['confianza']*100:.1f}%)")

            # Barras de distribución de probabilidades
            c_b1, c_b2 = st.columns(2)
            with c_b1:
                st.metric("Probabilidad Nivel Básico (A1/A2)", f"{res['probabilidad_basico']*100:.2f}%")
                st.progress(res["probabilidad_basico"])
            with c_b2:
                st.metric("Probabilidad Nivel Avanzado (B1/B2)", f"{res['probabilidad_avanzado']*100:.2f}%")
                st.progress(res["probabilidad_avanzado"])

            # 3. Valor Agregado: Explicación Pedagógica con IA Generativa (Gemini)
            st.write("")
            st.markdown("##### 3. Explicabilidad Pedagógica con IA Generativa (Valor Agregado):")
            if hay_ia_generativa_disponible():
                if st.button("✨ Generar Justificación Pedagógica con Gemini", key="btn_explicar"):
                    with st.spinner("Gemini está analizando la sintaxis y el vocabulario..."):
                        analisis["explicacion"] = explicar_prediccion_ia(
                            analisis["texto"],
                            res["nivel"],
                            analisis["palabras"],
                            analisis["dificultad"],
                            res["confianza"]
                        )
                
                if analisis.get("explicacion"):
                    st.markdown(
                        f"""
                        <div style="background:#F1EFEA; border-left:4px solid #0F766E; padding:1.2rem; border-radius:0 10px 10px 0; margin-top:1rem;">
                            <strong>Explicación Generada por IA:</strong><br><br>
                            {analisis['explicacion']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.caption("ℹ️ Clave de Gemini no configurada en .env; la explicación por IA generativa está inactiva.")

    # ------------------ PESTAÑA 2: SLIDERS MANUALES ------------------
    with tab_sliders:
        st.caption("Mueva los parámetros numéricos de forma manual para verificar las fronteras matemáticas del clasificador:")
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            s_palabras = st.slider("Conteo de Palabras:", min_value=50, max_value=600, value=320, step=10, key="s_pal")
        with p_c2:
            s_dif = st.slider("Dificultad Léxica (1.0 a 5.0):", min_value=1.0, max_value=5.0, value=3.4, step=0.1, key="s_dif")

        if st.button("Ejecutar Inferencia Paramétrica", key="btn_slider_pred"):
            try:
                res_s = predecir_lectura(s_palabras, s_dif)
                st.success(f"Nivel Predicho: **{res_s['nivel']}** (Certeza: {res_s['confianza']*100:.1f}%)")
                st.progress(res_s["probabilidad_avanzado"] if res_s["clase_binaria"] == 1 else res_s["probabilidad_basico"])
            except FileNotFoundError:
                st.error("Modelo no encontrado.")

    # ------------------ MATRIZ DE CONFUSIÓN OFICIAL ------------------
    st.write("")
    with st.expander("📊 Ver Matriz de Confusión Oficial del Modelo (Requisito 7)"):
        ruta_matriz = os.path.join(os.path.dirname(__file__), "..", "graficos", "matriz_confusion_sklearn.png")
        if os.path.exists(ruta_matriz):
            st.image(ruta_matriz, caption="Matriz de Confusión sobre Conjunto de Prueba no visto (Scikit-Learn)", use_container_width=True)
        else:
            st.info("La matriz de confusión se generará al ejecutar `modelo_proyecto.py`.")
