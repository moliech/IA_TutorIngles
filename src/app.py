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
                        f"""<div style="background:#F1EFEA; border-left:4px solid #0F766E; padding:1.2rem; border-radius:0 10px 10px 0; margin-top:1rem;">
    <strong>Explicación Generada por IA:</strong><br><br>
    {analisis['explicacion']}
</div>""",
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
