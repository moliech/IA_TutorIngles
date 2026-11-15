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
from usar_modelo import predecir_lectura, predecir_lote, generar_figura_proyeccion
from gemini_tutor import (
    hay_ia_generativa_disponible,
    explicar_prediccion_ia,
    responder_duda_estudiante,
    generar_resumen_espanol,
    obtener_traduccion_con_cache,
)

st.set_page_config(
    page_title="LexiaAdapt | Tutor Inteligente de Inglés",
    page_icon="📖",
    layout="wide"
)

# Inyección de estilos y tokens visuales ergonómicos y educativos
aplicar_tema_lexia()

# ------------------ CARGA DE DATOS ------------------
@st.cache_data
def cargar_catalogo():
    ruta = os.path.join(os.path.dirname(__file__), "..", "data", "proyecto_limpio.csv")
    if not os.path.exists(ruta):
        ruta = os.path.join(os.path.dirname(__file__), "data", "proyecto_limpio.csv")
    return pd.read_csv(ruta)

df_lecturas = cargar_catalogo()

# ------------------ HELPER PEDAGÓGICO ------------------
def obtener_mision_pedagogica(nivel_cefr: str) -> str:
    """Devuelve una orientación didáctica y motivadora según el nivel del estudiante."""
    if nivel_cefr == "A1":
        return (
            "Identifica las acciones cotidianas y el vocabulario familiar. "
            "No te preocupes por entender cada palabra aislada; concéntrate en captar la idea principal y los personajes."
        )
    elif nivel_cefr == "A2":
        return (
            "Presta atención a los conectores de tiempo (before, after, then) y a las descripciones. "
            "Usa la pestaña en español para cotejar las frases completas si tienes dudas."
        )
    elif nivel_cefr == "B1":
        return (
            "Descubre las causas, efectos y opiniones en el texto. "
            "Anota los adjetivos y términos técnicos nuevos preguntándole a tu tutor interactivo."
        )
    else:
        return (
            "Analiza la estructura argumentativa y los términos especializados. "
            "Intenta responder las preguntas de comprensión del tutor directamente en inglés para reforzar tu fluidez."
        )

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

# ------------------ ENCABEZADO MODERNO Y EDUCATIVO ------------------
st.markdown(
    """<div class="lex-banner">
    <h1>📖 LexiaAdapt</h1>
    <p>Plataforma Inteligente de Lectura Adaptativa e Inferencia Lingüística</p>
    <span class="sub-tag">🎓 Cartago, Valle del Cauca • Inteligencia Artificial 2026-02</span>
</div>""",
    unsafe_allow_html=True
)

modo = st.radio(
    "Seleccione el Modo de Operación:",
    ["📖 Modo Estudiante (Lectura & Tutoría)", "🧠 Modo Auditoría IA (Simulador con Texto Libre)"],
    horizontal=True
)

st.write("")

# =========================================================================
# VISTA 1: MODO ESTUDIANTE (LECTURA GUIADA, TRADUCCIÓN NATIVA Y TUTOR IA)
# =========================================================================
if "Modo Estudiante" in modo:
    st.markdown("<h3 style='font-size:1.6rem; font-weight:800; color:#0F2942; margin-bottom:0.2rem;'>📚 Biblioteca de Lecturas Graduadas</h3>", unsafe_allow_html=True)
    st.markdown("<p class='section-subtitle'>Selecciona el área temática y la lectura que deseas practicar hoy:</p>", unsafe_allow_html=True)

    c_f1, c_f2 = st.columns([1, 2])
    with c_f1:
        categorias = ["Todas"] + sorted(df_lecturas["categoria"].dropna().unique().tolist())
        cat_sel = st.selectbox("Área Temática:", categorias)

    df_filtrado = df_lecturas if cat_sel == "Todas" else df_lecturas[df_lecturas["categoria"] == cat_sel]

    with c_f2:
        titulos = df_filtrado["titulo"].tolist()
        titulo_sel = st.selectbox(
            "Seleccione la Lectura:",
            options=titulos,
            index=None,
            placeholder="👉 Elige una lectura de la lista...",
            key="select_lectura_estudiante"
        )

    # ESTADO INICIAL: PANTALLA LIMPIA DE BIENVENIDA DIDÁCTICA (SI NO HA SELECCIONADO LECTURA)
    if titulo_sel is None:
        st.markdown(
            """<div style="background:#FFFFFF; border:2px dashed #CBD5E1; border-radius:20px; padding:3.5rem 2rem; text-align:center; margin-top:1.5rem;">
    <div style="font-size:3.5rem; margin-bottom:1rem;">📚✨</div>
    <h2 style="color:#0F2942; font-weight:800; margin-bottom:0.6rem;">¡Bienvenido a tu Espacio de Lectura Adaptativa!</h2>
    <p style="font-size:1.15rem; color:#475569; max-width:650px; margin:0 auto 2.2rem auto; line-height:1.6;">
        Elige un área temática y selecciona la lectura que deseas practicar en el menú superior para desplegar el texto en inglés, su traducción y el tutor virtual.
    </p>
    <div style="display:flex; justify-content:center; gap:1.5rem; flex-wrap:wrap;">
        <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:14px; padding:1.2rem 1.5rem; text-align:left; max-width:240px;">
            <span style="font-size:1.6rem;">🇬🇧</span>
            <h4 style="color:#1E3A8A; margin:0.4rem 0 0.2rem 0; font-weight:700;">40 Lecturas Reales</h4>
            <p style="font-size:0.9rem; color:#2563EB; margin:0;">Graduadas por dificultad desde A1 hasta B2.</p>
        </div>
        <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:14px; padding:1.2rem 1.5rem; text-align:left; max-width:240px;">
            <span style="font-size:1.6rem;">🇪🇸</span>
            <h4 style="color:#065F46; margin:0.4rem 0 0.2rem 0; font-weight:700;">Traducción al Español</h4>
            <p style="font-size:0.9rem; color:#10B981; margin:0;">Pestaña bilingüe para cotejo inmediato.</p>
        </div>
        <div style="background:#FAF5FF; border:1px solid #E9D5FF; border-radius:14px; padding:1.2rem 1.5rem; text-align:left; max-width:240px;">
            <span style="font-size:1.6rem;">💬</span>
            <h4 style="color:#581C87; margin:0.4rem 0 0.2rem 0; font-weight:700;">Tutor Virtual IA</h4>
            <p style="font-size:0.9rem; color:#8B5CF6; margin:0;">Resuelve tus preguntas de vocabulario y gramática.</p>
        </div>
    </div>
</div>""",
            unsafe_allow_html=True
        )

    else:
        # LECTURA SELECCIONADA: MOSTRAR TEXTO, TRADUCCIÓN Y TUTOR
        fila = df_filtrado[df_filtrado["titulo"] == titulo_sel].iloc[0]
        texto_ingles = fila.get("texto_ingles", "Texto en proceso de carga.")

        st.write("")
        c_tit1, c_tit2 = st.columns([2, 1])
        with c_tit1:
            st.markdown(f"### {fila['titulo']}")
            st.markdown(render_badge(fila["nivel_cefr"]), unsafe_allow_html=True)
        with c_tit2:
            tam_opcion = st.radio(
                "🔠 Tamaño del texto:",
                ["Normal (18px)", "Grande (22px)", "Muy Grande (28px)"],
                index=1,
                horizontal=True,
                key=f"tam_radio_{fila['id']}"
            )
            tam_px = 18 if "18px" in tam_opcion else (28 if "28px" in tam_opcion else 22)

        st.write("")
        col_lectura, col_metricas = st.columns([2, 1], gap="large")

        with col_lectura:
            # PESTAÑAS NATIVAS: INGLÉS Y ESPAÑOL (SIN TARJETAS NI BOTONES RAROS)
            tab_en, tab_es = st.tabs(["🇬🇧 Lectura en Inglés", "🇪🇸 Traducción al Español"])

            with tab_en:
                st.markdown(
                    f"""<div style="font-family:'Lexend',sans-serif; font-size:{tam_px}px; line-height:1.9; color:#0F172A; background:#FFFFFF; padding:2rem 2.2rem; border-radius:16px; border:2px solid #E2E8F0; border-left:8px solid #2563EB; box-shadow:0 4px 15px rgba(0,0,0,0.04); margin-top:0.8rem;">
    {texto_ingles}
</div>""",
                    unsafe_allow_html=True
                )

            with tab_es:
                clave_trad = f"trad_{fila['id']}"
                if clave_trad not in st.session_state:
                    with st.spinner("Cargando traducción al español..."):
                        st.session_state[clave_trad] = obtener_traduccion_con_cache(
                            int(fila["id"]), fila["titulo"], texto_ingles
                        )
                traduccion = st.session_state[clave_trad]
                st.markdown(
                    f"""<div style="font-family:'Lexend',sans-serif; font-size:{tam_px}px; line-height:1.9; color:#064E3B; background:#F8FCF9; padding:2rem 2.2rem; border-radius:16px; border:2px solid #BBF7D0; border-left:8px solid #10B981; box-shadow:0 4px 15px rgba(0,0,0,0.04); margin-top:0.8rem;">
    {traduccion}
</div>""",
                    unsafe_allow_html=True
                )

            st.write("")
            # --- APOYO PEDAGÓGICO: RESUMEN GUIADO EN ESPAÑOL ---
            clave_resumen = f"resumen_{fila['id']}"
            with st.expander("💡 ¿No entendiste la idea general? Ver Resumen en 3 Puntos Clave"):
                if st.button("Generar Resumen Guiado", key="btn_resumen"):
                    with st.spinner("El tutor está extrayendo los puntos clave..."):
                        resumen = generar_resumen_espanol(fila["titulo"], texto_ingles)
                        st.session_state[clave_resumen] = resumen

                if clave_resumen in st.session_state:
                    st.markdown(st.session_state[clave_resumen])

            st.write("")
            # --- ASISTENTE INTERACTIVO / CHAT CON EL TUTOR IA ---
            st.markdown("#### 💬 Asistente Interactivo (LexiaTutor)")
            st.caption("Resuelve tus dudas sobre vocabulario, pronunciación, gramática o comprensión de este texto:")

            clave_chat = f"chat_{fila['id']}"
            if clave_chat not in st.session_state:
                st.session_state[clave_chat] = []

            # Botones de preguntas rápidas sugeridas
            c_q1, c_q2, c_q3 = st.columns(3)
            prompt_rapido = None
            if c_q1.button("🔑 ¿3 palabras clave y significado?", use_container_width=True, key=f"q1_{fila['id']}"):
                prompt_rapido = "¿Cuáles son las 3 palabras más importantes o difíciles de este texto y qué significan en español?"
            if c_q2.button("📝 Explícame la idea en 2 frases", use_container_width=True, key=f"q2_{fila['id']}"):
                prompt_rapido = "Explícame de forma muy sencilla en dos frases en español de qué trata este texto."
            if c_q3.button("❓ Hazme una pregunta de comprensión", use_container_width=True, key=f"q3_{fila['id']}"):
                prompt_rapido = "Hazme una pregunta sencilla en inglés sobre este texto para ver si lo comprendí."

            # Procesar prompt rápido si fue pulsado
            if prompt_rapido:
                st.session_state[clave_chat].append({"rol": "user", "texto": prompt_rapido})
                with st.spinner("LexiaTutor está pensando..."):
                    resp = responder_duda_estudiante(prompt_rapido, fila["titulo"], texto_ingles)
                    st.session_state[clave_chat].append({"rol": "assistant", "texto": resp})
                st.rerun()

            # Contenedor visual de mensajes del chat con scroll
            chat_container = st.container(height=340)
            with chat_container:
                if not st.session_state[clave_chat]:
                    st.info("👋 ¡Hola! Soy LexiaTutor. Escribe una pregunta abajo o pulsa uno de los botones de arriba para empezar.")
                for msj in st.session_state[clave_chat]:
                    with st.chat_message(msj["rol"]):
                        st.markdown(msj["texto"])

            # Input de chat SIEMPRE posicionado al final (abajo)
            if prompt_input := st.chat_input("Escribe tu pregunta sobre la lectura aquí..."):
                st.session_state[clave_chat].append({"rol": "user", "texto": prompt_input})
                with st.spinner("LexiaTutor está pensando..."):
                    resp = responder_duda_estudiante(prompt_input, fila["titulo"], texto_ingles)
                    st.session_state[clave_chat].append({"rol": "assistant", "texto": resp})
                st.rerun()

        with col_metricas:
            st.markdown("#### 📊 Información de Lectura")

            st.markdown(
                f"""<div class="metric-card-edu">
    <div class="title">⏱️ Tiempo Estimado de Lectura</div>
    <div class="val">~{fila['tiempo_lectura_est_min']} minutos</div>
</div>
<div class="metric-card-edu">
    <div class="title">📄 Extensión del Texto</div>
    <div class="val">{fila['conteo_palabras']} palabras</div>
</div>
<div class="metric-card-edu">
    <div class="title">🎯 Nivel Sugerido</div>
    <div class="val">{fila['nivel_cefr']} ({'Básico' if fila['nivel_cefr'] in ['A1', 'A2'] else 'Intermedio/Avanzado'})</div>
</div>
<div class="metric-card-edu">
    <div class="title">🌟 Complejidad Léxica</div>
    <div class="val">{fila['dificultad_lexica']} / 5.0</div>
</div>""",
                unsafe_allow_html=True
            )

            dif_val = float(fila['dificultad_lexica']) / 5.0
            st.caption("Progreso de Complejidad:")
            st.progress(min(1.0, max(0.1, dif_val)))

            mision_txt = obtener_mision_pedagogica(fila['nivel_cefr'])
            st.markdown(
                f"""<div class="pedagogical-box">
    <h5>🎯 Meta de Aprendizaje</h5>
    <p>{mision_txt}</p>
</div>""",
                unsafe_allow_html=True
            )

# =========================================================================
# VISTA 2: DEMOSTRACIÓN DE INFERENCIA PREDICTIVA EN VIVO (MODELO SERIALIZADO)
# =========================================================================
else:
    st.subheader("🔬 Demostración de Inferencia Predictiva en Vivo")
    st.write(
        "Demostración del sistema en tiempo real: el modelo serializado (`modelo_final.joblib`), "
        "previamente entrenado con el corpus de 40 lecturas, evalúa y clasifica **nuevas lecturas inéditas sin reentrenar**."
    )

    tab_inferencia, tab_diagnostico = st.tabs([
        "📥 Clasificación de Nuevas Lecturas (Inferencia sin Reentrenar)",
        "📊 Diagnóstico y Proyección en la Frontera de Decisión"
    ])

    # ------------------ PESTAÑA 1: INFERENCIA SOBRE NUEVAS LECTURAS ------------------
    with tab_inferencia:
        # 1. Dataset base de referencia colapsable
        with st.expander("📁 Ver Dataset Base Histórico de Entrenamiento (40 Lecturas de Cartago)"):
            st.markdown(
                "El modelo fue entrenado con un corpus equilibrado de **40 lecturas** graduadas según "
                "el Marco Común Europeo de Referencia (MCER):"
            )
            c_b1, c_b2, c_b3, c_b4 = st.columns(4)
            c_b1.metric("Lecturas A1 (Básico)", "10 lecturas", "~112 palabras")
            c_b2.metric("Lecturas A2 (Básico)", "10 lecturas", "~201 palabras")
            c_b3.metric("Lecturas B1 (Avanzado)", "10 lecturas", "~327 palabras")
            c_b4.metric("Lecturas B2 (Avanzado)", "10 lecturas", "~473 palabras")

            st.dataframe(
                df_lecturas[["id", "titulo", "nivel_cefr", "conteo_palabras", "dificultad_lexica", "categoria"]],
                use_container_width=True,
                column_config={
                    "conteo_palabras": "Palabras",
                    "dificultad_lexica": "Dificultad (1-5)",
                    "nivel_cefr": "Nivel MCER"
                }
            )

        st.write("")
        st.markdown("#### ⚡ Evaluación de Nuevas Lecturas Inéditas")
        st.write(
            "Carga un lote de prueba o tu propio archivo CSV para someterlo al modelo entrenado en milisegundos:"
        )

        # Estado de sesión para el lote activo (por defecto carga los 3 casos clave de control)
        if "df_lote_subido" not in st.session_state or st.session_state["df_lote_subido"] is None:
            st.session_state["df_lote_subido"] = pd.DataFrame([
                {
                    "id": 1,
                    "titulo": "Daily Morning Routine",
                    "conteo_palabras": 105,
                    "dificultad_lexica": 1.5,
                    "descripcion": "Rutina diaria breve con vocabulario cotidiano A1 y oraciones simples.",
                    "nivel_esperado": "A1 (Básico)"
                },
                {
                    "id": 2,
                    "titulo": "AI in Biomedical Diagnostics",
                    "conteo_palabras": 480,
                    "dificultad_lexica": 4.4,
                    "descripcion": "Texto académico extenso con terminología médica B2 y sintaxis compleja.",
                    "nivel_esperado": "B2 (Avanzado)"
                },
                {
                    "id": 3,
                    "titulo": "Clean Energy Systems Transition",
                    "conteo_palabras": 278,
                    "dificultad_lexica": 3.2,
                    "descripcion": "Lectura intermedia en la frontera de decisión para evaluar la transición A2/B1.",
                    "nivel_esperado": "B1 (Frontera)"
                },
            ])

        # Botones de selección de lote
        c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 1])
        if c_btn1.button("⚡ 3 Casos de Control Oficial (Básico, Avanzado, Frontera)", use_container_width=True):
            st.session_state["df_lote_subido"] = pd.DataFrame([
                {
                    "id": 1,
                    "titulo": "Daily Morning Routine",
                    "conteo_palabras": 105,
                    "dificultad_lexica": 1.5,
                    "descripcion": "Rutina diaria breve con vocabulario cotidiano A1 y oraciones simples.",
                    "nivel_esperado": "A1 (Básico)"
                },
                {
                    "id": 2,
                    "titulo": "AI in Biomedical Diagnostics",
                    "conteo_palabras": 480,
                    "dificultad_lexica": 4.4,
                    "descripcion": "Texto académico extenso con terminología médica B2 y sintaxis compleja.",
                    "nivel_esperado": "B2 (Avanzado)"
                },
                {
                    "id": 3,
                    "titulo": "Clean Energy Systems Transition",
                    "conteo_palabras": 278,
                    "dificultad_lexica": 3.2,
                    "descripcion": "Lectura intermedia en la frontera de decisión para evaluar la transición A2/B1.",
                    "nivel_esperado": "B1 (Frontera)"
                },
            ])
            st.session_state["explicacion_lote"] = None
            st.rerun()

        if c_btn2.button("📚 Lote de 6 Lecturas Nuevas de Cartago (Inéditas)", use_container_width=True):
            st.session_state["df_lote_subido"] = pd.DataFrame([
                {"id": 101, "titulo": "Meeting a New Friend in the Park", "conteo_palabras": 95, "dificultad_lexica": 1.4, "descripcion": "Conversación informal de saludo en el parque.", "nivel_esperado": "A1 (Básico)"},
                {"id": 102, "titulo": "Ordering Food at a Local Restaurant", "conteo_palabras": 140, "dificultad_lexica": 1.8, "descripcion": "Intercambio cotidiano de menú, precios y compras.", "nivel_esperado": "A2 (Básico)"},
                {"id": 103, "titulo": "Environmental Conservation in Valle", "conteo_palabras": 260, "dificultad_lexica": 3.0, "descripcion": "Párrafos informativos con conectores de causa y efecto.", "nivel_esperado": "B1 (Frontera)"},
                {"id": 104, "titulo": "Machine Learning and Cognitive Networks", "conteo_palabras": 460, "dificultad_lexica": 4.5, "descripcion": "Artículo técnico de inteligencia artificial y redes neuronales.", "nivel_esperado": "B2 (Avanzado)"},
                {"id": 105, "titulo": "Urban Transit Infrastructure in Cartago", "conteo_palabras": 290, "dificultad_lexica": 3.3, "descripcion": "Análisis de transporte regional y movilidad urbana.", "nivel_esperado": "B1 (Avanzado)"},
                {"id": 106, "titulo": "Short Bus Station Directions", "conteo_palabras": 110, "dificultad_lexica": 1.6, "descripcion": "Intercambio breve de instrucciones de ubicación.", "nivel_esperado": "A1 (Básico)"},
            ])
            st.session_state["explicacion_lote"] = None
            st.rerun()

        if c_btn3.button("🔄 Restablecer Casos", use_container_width=True):
            st.session_state["df_lote_subido"] = None
            st.session_state["explicacion_lote"] = None
            st.rerun()

        archivo_csv = st.file_uploader(
            "O arrastra un archivo CSV con nuevas lecturas ('conteo_palabras' y 'dificultad_lexica'):",
            type=["csv"],
            key="uploader_nuevas_lecturas"
        )
        if archivo_csv is not None:
            try:
                df_subido = pd.read_csv(archivo_csv)
                st.session_state["df_lote_subido"] = df_subido
                st.session_state["explicacion_lote"] = None
            except Exception as e:
                st.error(f"Error al leer el archivo CSV: {e}")

        # Ejecutar Inferencia con el modelo serializado (sin reentrenar)
        df_actual = st.session_state.get("df_lote_subido")
        if df_actual is not None and "conteo_palabras" in df_actual.columns and "dificultad_lexica" in df_actual.columns:
            try:
                df_predicho = predecir_lote(df_actual)

                st.write("")
                st.markdown("##### 📊 Resumen Ejecutivo del Lote Clasificado:")
                n_total = len(df_predicho)
                n_basico = int((df_predicho["nivel_predicho"].str.contains("Básico")).sum())
                n_avanzado = n_total - n_basico
                certeza_media = float(df_predicho["certeza_%"].mean())

                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                m_c1.metric("Lecturas Nuevas", f"{n_total}")
                m_c2.metric("Clasificadas Básicas (A1/A2)", f"{n_basico}", f"{n_basico/n_total*100:.0f}% del lote")
                m_c3.metric("Clasificadas Avanzadas (B1/B2)", f"{n_avanzado}", f"{n_avanzado/n_total*100:.0f}% del lote")
                m_c4.metric("Certeza Promedio", f"{certeza_media:.1f}%", "Confianza probabilística")

                st.write("")
                st.markdown("##### 📋 Tabla de Predicciones en Tiempo Real (Joblib sin Reentrenar):")
                cols_mostrar = [c for c in ["id", "titulo", "caso", "conteo_palabras", "dificultad_lexica", "nivel_esperado", "prob_avanzado_%", "nivel_predicho", "certeza_%"] if c in df_predicho.columns]

                st.dataframe(
                    df_predicho[cols_mostrar],
                    use_container_width=True,
                    column_config={
                        "conteo_palabras": "Palabras",
                        "dificultad_lexica": "Dificultad (1-5)",
                        "prob_avanzado_%": st.column_config.ProgressColumn(
                            "Prob. Avanzado",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                        ),
                        "certeza_%": "Certeza",
                        "nivel_predicho": "Veredicto del Modelo",
                    }
                )

                # Botón para descargar el reporte CSV
                csv_bytes = df_predicho.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar Reporte CSV de Inferencia",
                    data=csv_bytes,
                    file_name="reporte_prediccion_lexiaadapt.csv",
                    mime="text/csv",
                )

                # Sección de explicabilidad con Gemini sobre un caso
                st.write("")
                st.markdown("##### ✨ Justificación Pedagógica Ejecutiva con IA Generativa (Gemini):")
                col_id = "titulo" if "titulo" in df_predicho.columns else ("caso" if "caso" in df_predicho.columns else "id")
                opciones_casos = df_predicho[col_id].tolist()
                caso_sel = st.selectbox("Selecciona una lectura evaluada para ver su análisis:", opciones_casos, key="select_caso_gemini")

                fila_sel = df_predicho[df_predicho[col_id] == caso_sel].iloc[0]

                if hay_ia_generativa_disponible():
                    if st.button(f"✨ Justificar con Gemini para '{caso_sel}'", key="btn_justificar_gemini"):
                        with st.spinner("Gemini está formulando la síntesis ejecutiva..."):
                            desc = str(fila_sel.get("descripcion", fila_sel.get("titulo", f"Lectura '{caso_sel}'")))
                            st.session_state["explicacion_lote"] = explicar_prediccion_ia(
                                desc,
                                fila_sel["nivel_predicho"],
                                int(fila_sel["conteo_palabras"]),
                                float(fila_sel["dificultad_lexica"]),
                                float(fila_sel["certeza_%"]) / 100.0
                            )

                    if st.session_state.get("explicacion_lote"):
                        st.markdown(
                            f"""<div style="background:#F8FAFC; border:1px solid #CBD5E1; border-left:5px solid #0F766E; padding:1.2rem 1.5rem; border-radius:10px; margin-top:0.8rem; font-size:0.92rem; line-height:1.7; color:#1E293B;">
    <strong style="color:#0F766E; font-size:1rem;">Justificación Ejecutiva por IA para '{caso_sel}':</strong><br><br>
    {st.session_state['explicacion_lote']}
</div>""",
                            unsafe_allow_html=True
                        )
                else:
                    st.caption("ℹ️ Clave de Gemini no detectada en .env; justificación generativa inactiva.")

            except Exception as e:
                st.error(f"Error durante la inferencia: {e}")

    # ------------------ PESTAÑA 2: DIAGNÓSTICO Y PROYECCIÓN ------------------
    with tab_diagnostico:
        st.markdown("#### 📊 Diagnóstico Técnico y Proyección en la Frontera de Decisión")
        st.write(
            "Visualiza la separación matemática entre niveles y proyecta exactamente dónde cayeron las nuevas lecturas evaluadas:"
        )

        col_diag1, col_diag2 = st.columns([1.1, 0.9], gap="large")

        with col_diag1:
            st.markdown("##### 1. Proyección de Nuevas Lecturas sobre la Frontera Matemática")
            df_lote_act = st.session_state.get("df_lote_subido")
            fig_proyeccion = generar_figura_proyeccion(df_lote_act)
            st.pyplot(fig_proyeccion)

            st.markdown(
                """<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:0.85rem 1rem; margin-top:0.4rem; font-size:0.86rem; color:#334155; line-height:1.6;">
    <strong>Interpretación Visual:</strong> Los círculos azules (Básico A1/A2) y rojos (Avanzado B1/B2) representan el corpus base de 40 lecturas. La línea discontinua marca la frontera de decisión (P = 50%). Las <strong>estrellas doradas</strong> proyectan las nuevas lecturas evaluadas en vivo, demostrando de forma transparente por qué cada una fue clasificada.
</div>""",
                unsafe_allow_html=True
            )

        with col_diag2:
            st.markdown("##### 2. Matriz de Confusión Oficial (Datos de Prueba)")
            ruta_matriz = os.path.join(os.path.dirname(__file__), "..", "graficos", "matriz_confusion_sklearn.png")
            if os.path.exists(ruta_matriz):
                st.image(ruta_matriz, width=380)

            st.markdown(
                """<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:0.85rem 1rem; margin-top:0.4rem; font-size:0.86rem; color:#334155; line-height:1.6;">
    <strong>Evaluación en Test (8 lecturas):</strong> Exactitud del 100%, F1-Score = 1.0, cero falsos positivos y cero falsos negativos.<br><br>
    <strong>Pesos del Pipeline ($w_1 x_1 + w_2 x_2 + b = 0$):</strong><br>
    • $w_1 = +1.64$ (Palabras): Mayor extensión incrementa la probabilidad de nivel avanzado.<br>
    • $w_2 = +1.50$ (Dificultad léxica): Mayor complejidad sintáctica exige mayor competencia lectora.<br>
    • $b = +0.08$ (Sesgo balanceado del umbral).
</div>""",
                unsafe_allow_html=True
            )
