"""
src/app.py

Aplicación Principal Streamlit - LexiaAdapt (IA_TutorIngles)
Plataforma de Lectura Adaptativa e Inferencia en Vivo
Sistema Inteligente de Comprensión Lectora
"""

import os
import re
import importlib
import pandas as pd
import streamlit as st

# Importación y recarga en caliente de módulos internos del sistema
import ui_theme
import usar_modelo
import gemini_tutor

importlib.reload(ui_theme)
importlib.reload(usar_modelo)
importlib.reload(gemini_tutor)

from ui_theme import aplicar_tema_lexia, render_badge
from usar_modelo import predecir_lectura, predecir_lote, generar_figura_proyeccion
from gemini_tutor import (
    hay_ia_generativa_disponible,
    explicar_prediccion_ia,
    responder_duda_estudiante,
    generar_resumen_espanol,
    obtener_traduccion_con_cache,
    traducir_texto_espanol,
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

# ------------------ EXTRACTOR LINGÜÍSTICO Y CALIBRADOR DE DIFICULTAD ------------------
def calcular_metricas_lectura(texto: str):
    """
    Calcula el conteo exacto de palabras y calibra la dificultad léxica (1.0 a 5.0)
    a partir de indicadores lingüísticos positivos independientes de la extensión
    (longitud media de palabra, proporción de léxico complejo >=9 letras y extensión oracional).
    """
    palabras = re.findall(r"\b[a-zA-Z]+\b", str(texto))
    n = len(palabras)
    if n == 0:
        return 0, 1.0

    avg_len = sum(len(w) for w in palabras) / n
    pct_long = sum(1 for w in palabras if len(w) >= 9) / n
    oraciones = [s.strip() for s in re.split(r"[.!?]+", str(texto)) if s.strip()]
    avg_sent_len = min(35.0, n / len(oraciones)) if oraciones else min(35.0, float(n))

    raw_dif = -0.2177 + (0.0782 * avg_len) + (3.3717 * pct_long) + (0.0868 * avg_sent_len)
    dif = round(float(min(5.0, max(1.0, raw_dif))), 2)
    return n, dif

def preparar_lote_entradas(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza el DataFrame de entrada. Si contiene la lectura en texto,
    el sistema calcula automáticamente el conteo de palabras y la dificultad léxica.
    """
    df = df_in.copy()
    cols_map = {str(c).strip().lower(): c for c in df.columns}

    # Detectar columna con el texto de la lectura
    col_lectura = None
    for k in ["lectura", "texto", "texto_ingles", "text", "content", "reading"]:
        if k in cols_map:
            col_lectura = cols_map[k]
            break

    # Si no tiene columnas numéricas directas pero sí texto, se calculan automáticamente
    tiene_metricas = "conteo_palabras" in cols_map and "dificultad_lexica" in cols_map
    if col_lectura and not tiene_metricas:
        p_list = []
        d_list = []
        for txt in df[col_lectura].astype(str):
            p, d = calcular_metricas_lectura(txt)
            p_list.append(p)
            d_list.append(d)
        df["conteo_palabras"] = p_list
        df["dificultad_lexica"] = d_list

    # Estandarizar nombres canónicos para la visualización
    for orig in list(df.columns):
        low = str(orig).strip().lower()
        if low in ["titulo", "title"]:
            df = df.rename(columns={orig: "titulo"})
        elif low in ["categoria", "category"]:
            df = df.rename(columns={orig: "categoria"})
        elif low in ["id", "codigo"]:
            df = df.rename(columns={orig: "id"})

    return df

# ------------------ ENCABEZADO MODERNO Y COMERCIAL ------------------
st.markdown(
    """<div class="lex-banner">
    <h1>📖 LexiaAdapt</h1>
    <p>Plataforma Inteligente de Lectura Adaptativa e Inferencia Lingüística</p>
    <span class="sub-tag">✨ EdTech AI Platform • Clasificación Adaptativa MCER &amp; Tutoría Bilingüe</span>
</div>""",
    unsafe_allow_html=True
)

modo = st.radio(
    "Módulo de la Plataforma:",
    ["🧠 Clasificador Predictivo e Inferencia IA", "📖 Biblioteca Curada (Catálogo Base)"],
    horizontal=True
)

st.write("")

# =========================================================================
# VISTA 2 (SECUNDARIA): BIBLIOTECA CURADA (LECTURA GUIADA, TRADUCCIÓN Y TUTOR)
# =========================================================================
if "Biblioteca" in modo:
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
                    st.info("👋 ¡Hola! Soy LexiaTutor, tu asistente académico. Escribe una consulta abajo o selecciona una opción rápida para comenzar.")
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
# VISTA 1 (PRINCIPAL): MOTOR DE CLASIFICACIÓN PREDICTIVA Y AUDITORÍA IA
# =========================================================================
else:
    st.subheader("🧠 Motor de Clasificación Lingüística y Evaluación MCER")
    st.write(
        "Evalúa lotes de lecturas inéditas en tiempo real. El motor calcula métricas léxicas automáticamente, "
        "determina el nivel de competencia (A1–B2) y ofrece análisis pedagógico asistido por IA generativa."
    )

    tab_inferencia, tab_diagnostico = st.tabs([
        "📥 Evaluación de Nuevas Lecturas",
        "📊 Analítica del Modelo y Frontera de Decisión"
    ])

    # ------------------ PESTAÑA 1: INFERENCIA SOBRE NUEVAS LECTURAS ------------------
    with tab_inferencia:
        st.markdown("#### ⚡ Análisis de Lotes y Documentos")
        st.write(
            "Selecciona un lote de muestra o carga tu propio archivo plano para clasificar lecturas automáticamente:"
        )

        # Estado de sesión para el lote activo (inicia en blanco por defecto)
        if "df_lote_subido" not in st.session_state:
            st.session_state["df_lote_subido"] = None
        if "uploader_key_version" not in st.session_state:
            st.session_state["uploader_key_version"] = 0

        # Botones de selección de lote con lecturas reales (el sistema calcula palabras y dificultad)
        c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 1])
        if c_btn1.button("⚡ Lote Rápido (3 Niveles)", use_container_width=True):
            st.session_state["uploader_key_version"] += 1
            st.session_state["df_cargado_pendiente"] = pd.DataFrame([
                {
                    "titulo": "Daily Morning Routine",
                    "categoria": "Cotidiano",
                    "lectura": (
                        "I wake up every day at six in the morning. First, I wash my face and brush my teeth. "
                        "Then, I go to the kitchen to prepare breakfast. I usually eat scrambled eggs with fresh bread "
                        "and drink a cup of black coffee. After eating, I walk to the university campus with my friends. "
                        "We study computer science and practice English vocabulary in the morning classes. In the afternoon, "
                        "I return home, help my parents with household chores, complete my homework, and read a short story. "
                        "At night, I listen to calm music and go to sleep early to rest well."
                    )
                },
                {
                    "titulo": "Clean Energy Systems Transition",
                    "categoria": "Ciencia",
                    "lectura": (
                        "The gradual structural transition toward renewable electricity and clean energy infrastructure represents "
                        "one of the most decisive challenges for sustainable economic progress across Latin America. Replacing fossil "
                        "fuel combustion with photovoltaic solar power plants, small hydroelectric generation stations, and modern wind "
                        "turbine networks dramatically curtails atmospheric greenhouse emissions while upgrading aging municipal distribution "
                        "grids. However, implementing comprehensive national decarbonization agendas requires sustained capital investment in "
                        "decentralized battery storage systems, automated electrical substations, and digital smart-grid monitoring software. "
                        "Regional public authorities, private engineering enterprises, and technical universities must collaborate actively to "
                        "train skilled specialists capable of maintaining complex clean energy equipment over multiple decades. Across fertile "
                        "agricultural valleys, integrating solar photovoltaic panels directly into automated crop irrigation systems optimizes "
                        "groundwater consumption, reduces monthly utility expenditures for small rural farmers, and preserves long-term ecological "
                        "stability along delicate river watersheds. Furthermore, progressive municipal regulations encouraging rooftop solar "
                        "installations on schools, public hospitals, and commercial buildings allow local communities to generate clean energy "
                        "close to consumption points while minimizing power transmission losses. Ultimately, coherent public policy must carefully "
                        "align industrial productivity goals with strict biodiversity conservation commitments to foster resilient regional development."
                    )
                },
                {
                    "titulo": "AI in Biomedical Diagnostics",
                    "categoria": "Tecnología",
                    "lectura": (
                        "The systematic integration of deep convolutional neural network architectures and multi-parametric predictive algorithms "
                        "is spearheading a monumental transformation across contemporary clinical oncology and diagnostic pathology worldwide. "
                        "Modern biomedical infrastructures increasingly deploy automated diagnostic pipelines capable of scrutinizing high-resolution "
                        "digitized histology tissue biopsies alongside longitudinal patient genomic sequencing profiles with diagnostic accuracy "
                        "matching specialized board-certified medical pathologists. By systematically identifying subtle microscopic morphological "
                        "anomalies across multi-gigapixel whole-slide microscopic scans, artificial intelligence algorithms detect pre-symptomatic "
                        "malignant cellular transformations long before macroscopic pathology becomes clinically detectable through conventional "
                        "physical examinations or routine radiographic screening. Furthermore, automated clinical decision-support software substantially "
                        "accelerates diagnostic workflow velocity in high-volume hospital pathology laboratories, enabling multidisciplinary healthcare "
                        "teams to formulate highly personalized pharmacological regimens tailored precisely to individual patient molecular biomarkers "
                        "and genetic mutations. In critical oncological interventions, early algorithmic identification fundamentally alters longitudinal "
                        "clinical prognosis, substantially mitigating the necessity for invasive exploratory surgical biopsies and severely toxic "
                        "systemic cytotoxic chemotherapy protocols. Beyond diagnostic imaging, sophisticated natural language processing frameworks "
                        "continuously synthesize vast repositories of peer-reviewed medical literature and anonymized electronic health records to "
                        "uncover unexpected epidemiological correlations and therapeutic drug interactions. Nevertheless, the widespread deployment "
                        "of autonomous predictive systems within regional healthcare networks requires rigorous clinical validation, continuous "
                        "algorithmic auditing, and transparent statistical calibration to prevent demographic biases arising from unrepresentative "
                        "training cohorts. Ethical governance committees and biomedical regulatory agencies emphasize that machine learning software "
                        "must always operate strictly as an assistive tool empowering clinical physicians rather than substituting human diagnostic "
                        "accountability and empathetic doctor-patient communication. Continued interdisciplinary collaboration among clinical oncologists, "
                        "computational biostatisticians, bioinformatics specialists, and software engineers remains indispensable to refine model "
                        "interpretability, protect sensitive patient confidentiality under stringent cybersecurity protocols, and democratize universal "
                        "access to precision diagnostic medicine across diverse healthcare environments globally."
                    )
                },
            ])
            st.session_state["df_lote_subido"] = None
            st.session_state["explicaciones_por_caso"] = {}
            st.session_state["ultimo_archivo_cargado"] = None
            st.rerun()

        if c_btn2.button("📚 Lote Integral (6 Lecturas)", use_container_width=True):
            st.session_state["uploader_key_version"] += 1
            st.session_state["df_cargado_pendiente"] = pd.DataFrame([
                {
                    "titulo": "Meeting a New Friend in the Park",
                    "categoria": "Cotidiano",
                    "lectura": (
                        "Hello! My name is Carlos and I am nineteen years old. Every morning I walk through the quiet park near my house. "
                        "The park has tall green trees, pretty yellow flowers, and a nice blue lake. Today I met a friendly boy named Mateo. "
                        "He is also a student and likes playing soccer on weekends. We talked about our favorite food, music, and English classes. "
                        "Now we practice speaking English together every afternoon."
                    )
                },
                {
                    "titulo": "Ordering Food at a Local Restaurant",
                    "categoria": "Cotidiano",
                    "lectura": (
                        "Good afternoon! Today is a sunny Saturday, so my family and I decided to have lunch at a cozy neighborhood restaurant near the river. "
                        "When we entered, a polite waiter smiled and guided us to a quiet wooden table near the open window. "
                        "We looked carefully at the printed menu for several minutes because every traditional dish looked delicious and fresh. "
                        "Finally, my parents ordered grilled chicken soup, white rice, fresh green salad, and roasted potatoes with garlic. "
                        "For dessert, my younger sister asked for sweet vanilla ice cream with sliced strawberries and cold mango juice. "
                        "The food arrived quickly, tasted wonderful, and we left a generous tip before walking happily back home together."
                    )
                },
                {
                    "titulo": "Environmental Conservation in River Valleys",
                    "categoria": "Ciencia",
                    "lectura": (
                        "Preserving biological diversity across fertile river valleys requires implementing sustainable agricultural methods alongside "
                        "responsible regional watershed governance. Over recent decades, unregulated industrial expansion and extensive chemical "
                        "fertilizer runoff have threatened freshwater ecosystems and reduced native forest corridors along important mountain rivers. "
                        "Fortunately, local farming cooperatives are increasingly adopting organic composting practices, protective riparian tree barriers, "
                        "and efficient drip irrigation infrastructure to maintain crop productivity while restoring delicate ecological balance. "
                        "Scientific monitoring programs conducted by regional environmental institutions demonstrate that protecting natural wetlands "
                        "filters agricultural sediments, prevents seasonal flooding in nearby towns, and sustains habitats for migratory bird species. "
                        "Furthermore, community education workshops in rural schools teach young students how to classify recyclable materials, conserve "
                        "household water supplies, and measure local water quality indicators. When agricultural producers, municipal leaders, and "
                        "university researchers collaborate around shared sustainability targets, regional river basins recover vitality while "
                        "supporting resilient rural economies over multiple generations."
                    )
                },
                {
                    "titulo": "Machine Learning and Cognitive Networks",
                    "categoria": "Tecnología",
                    "lectura": (
                        "Contemporary artificial intelligence architectures leverage deep multi-layered neural topologies and attention-based "
                        "transformer mechanisms to extract high-dimensional semantic representations from vast unstructured sensory and textual corpora. "
                        "Unlike classical rule-based expert systems that relied strictly upon manually programmed symbolic heuristics, modern connectionist "
                        "models autonomously discover intricate statistical regularities across billions of training parameters through iterative "
                        "stochastic gradient optimization. These computational breakthroughs empower automated cognitive reasoning, real-time cross-lingual "
                        "neural translation, precision speech recognition, and high-resolution diagnostic computer vision across global enterprise platforms. "
                        "In parallel, self-supervised pretraining paradigms enable foundational neural networks to generalize rapidly onto specialized "
                        "downstream domains—such as molecular protein folding prediction, autonomous robotic navigation, and personalized educational "
                        "tutoring—using remarkably modest quantities of manually annotated training samples. Nevertheless, scaling massive deep learning "
                        "models introduces substantial engineering and ethical challenges involving exponential computational energy consumption, hardware "
                        "semiconductor scarcity, algorithmic opacity, and the unintended amplification of historical socioeconomic biases embedded inside "
                        "internet datasets. Consequently, contemporary machine learning researchers prioritize developing parameter-efficient quantization "
                        "techniques, interpretable explainable artificial intelligence frameworks, and rigorous safety alignment benchmarks to verify "
                        "that autonomous cognitive systems operate reliably, equitably, and transparently across real-world human institutions."
                    )
                },
                {
                    "titulo": "Urban Transit Infrastructure in Intermediate Cities",
                    "categoria": "Ciencia",
                    "lectura": (
                        "Modernizing urban public transit in intermediate municipalities involves strategic planning of dedicated bus corridors, "
                        "non-motorized bicycle pathways, and synchronized traffic signaling to alleviate vehicular congestion and lower carbon emissions "
                        "across urban residential sectors."
                    )
                },
                {
                    "titulo": "Short Bus Station Directions",
                    "categoria": "Cotidiano",
                    "lectura": (
                        "Excuse me, could you please tell me how to get to the central bus terminal? Sure! First, walk straight along this main avenue "
                        "for three short blocks until you see a large bakery. Then, turn left at the traffic light and walk past the public library. "
                        "The blue ticket office is right across the street from the green supermarket. Have a very safe trip!"
                    )
                },
            ])
            st.session_state["df_lote_subido"] = None
            st.session_state["explicaciones_por_caso"] = {}
            st.session_state["ultimo_archivo_cargado"] = None
            st.rerun()

        if c_btn3.button("🔄 Restablecer Espacio de Trabajo", use_container_width=True):
            st.session_state["uploader_key_version"] += 1
            st.session_state["df_cargado_pendiente"] = None
            st.session_state["df_lote_subido"] = None
            st.session_state["explicaciones_por_caso"] = {}
            st.session_state["ultimo_archivo_cargado"] = None
            st.rerun()

        # Cargador de archivo plano con soporte nativo para separador '|'
        st.write("")
        c_up1, c_up2 = st.columns([2, 1], gap="medium")
        with c_up1:
            u_key = f"uploader_lecturas_v{st.session_state['uploader_key_version']}"
            archivo_subido = st.file_uploader(
                "Cargar archivo de lecturas (delimitado por | o CSV):",
                type=["txt", "csv"],
                key=u_key
            )
            if archivo_subido is not None:
                file_sig = (archivo_subido.name, archivo_subido.size)
                if st.session_state.get("ultimo_archivo_cargado") != file_sig:
                    try:
                        import io
                        contenido_raw = archivo_subido.getvalue().decode("utf-8", errors="replace")
                        try:
                            df_subido = pd.read_csv(io.StringIO(contenido_raw), sep="|")
                            if len(df_subido.columns) < 2:
                                df_subido = pd.read_csv(io.StringIO(contenido_raw), sep=None, engine="python")
                        except Exception:
                            df_subido = pd.read_csv(io.StringIO(contenido_raw), sep=None, engine="python")

                        st.session_state["df_cargado_pendiente"] = df_subido
                        st.session_state["df_lote_subido"] = None
                        st.session_state["explicaciones_por_caso"] = {}
                        st.session_state["ultimo_archivo_cargado"] = file_sig
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al leer y procesar el archivo plano: {e}")

        with c_up2:
            st.caption("📋 Estructura requerida (separador |):")
            st.markdown(
                """<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:0.65rem 0.85rem; font-size:0.83rem; color:#475569; line-height:1.5;">
    <strong>Columnas del Archivo Plano:</strong><br>
    <code>titulo | categoria | lectura</code><br><br>
    <em>El sistema calcula automáticamente el conteo de palabras, la dificultad léxica y clasifica el nivel.</em>
</div>""",
                unsafe_allow_html=True
            )
            ruta_plantilla = os.path.join(os.path.dirname(__file__), "..", "data", "ejemplos_demostracion", "lote_1_variado_niveles.txt")
            if os.path.exists(ruta_plantilla):
                with open(ruta_plantilla, "r", encoding="utf-8") as fp:
                    plantilla_plana = fp.read()
            else:
                plantilla_plana = (
                    "titulo|categoria|lectura\n"
                    "My Morning Walk in the Park|Cotidiano|I wake up early in the morning and walk through the park. The air is fresh and the streets are very quiet. I like to say good morning to the friendly neighbors and buy fresh warm bread at the local bakery with my family.\n"
                    "Ordering Coffee at a Local Shop|Cotidiano|Good afternoon! I would like to order a cup of black coffee and a small pastry, please. How much does it cost? Can I pay with credit card or cash? Thank you very much for your great service."
                )
            st.download_button(
                "📥 Descargar Plantilla Plana (|)",
                data=plantilla_plana,
                file_name="plantilla_nuevas_lecturas.txt",
                mime="text/plain",
                use_container_width=True
            )

        # Botón de acción explícita para procesar el lote cargado
        df_pendiente = st.session_state.get("df_cargado_pendiente")
        df_actual = st.session_state.get("df_lote_subido")

        if df_pendiente is not None and df_actual is None:
            st.write("")
            st.info(
                f"📁 **Lote preparado:** {len(df_pendiente)} lecturas listas para evaluación. "
                "Presiona el botón para extraer variables lingüísticas y clasificar:"
            )
            if st.button("🚀 Ejecutar Clasificación con IA", type="primary", use_container_width=True, key="btn_procesar_lote_ia"):
                with st.spinner("Procesando textos, extrayendo métricas y clasificando..."):
                    df_prep = preparar_lote_entradas(df_pendiente)
                    st.session_state["df_lote_subido"] = predecir_lote(df_prep)
                    st.session_state["df_cargado_pendiente"] = None
                    st.session_state["explicaciones_por_caso"] = {}
                    st.rerun()

        elif df_actual is None and df_pendiente is None:
            st.write("")
            st.info(
                "💡 **Motor de Evaluación Listo:**\n\n"
                "Selecciona un lote de muestra (`⚡ Lote Rápido` o `📚 Lote Integral`) "
                "o carga un archivo plano (`titulo|categoria|lectura`) para iniciar el análisis."
            )

        # Resultados de inferencia cuando el lote ha sido procesado
        if df_actual is not None and "conteo_palabras" in df_actual.columns and "dificultad_lexica" in df_actual.columns:
            try:
                df_predicho = df_actual

                st.write("")
                st.markdown("##### 📊 Resumen Ejecutivo del Lote Clasificado:")
                n_total = len(df_predicho)
                n_basico = int((df_predicho["nivel_predicho"].str.contains("Básico")).sum())
                n_avanzado = n_total - n_basico
                certeza_media = float(df_predicho["certeza_%"].mean())

                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                m_c1.metric("Lecturas Evaluadas", f"{n_total}")
                m_c2.metric("Nivel Básico (A1/A2)", f"{n_basico}", f"{n_basico/n_total*100:.0f}% del lote")
                m_c3.metric("Nivel Avanzado (B1/B2)", f"{n_avanzado}", f"{n_avanzado/n_total*100:.0f}% del lote")
                m_c4.metric("Certeza Promedio", f"{certeza_media:.1f}%", "Confianza probabilística")

                st.write("")
                st.markdown("##### 📋 Resultados de Clasificación Lingüística:")
                cols_mostrar = [c for c in ["id", "titulo", "categoria", "conteo_palabras", "dificultad_lexica", "prob_avanzado_%", "nivel_predicho", "subnivel_estimado", "certeza_%"] if c in df_predicho.columns]

                st.dataframe(
                    df_predicho[cols_mostrar],
                    use_container_width=True,
                    column_config={
                        "titulo": "Título de la Lectura",
                        "categoria": "Categoría",
                        "conteo_palabras": "Palabras",
                        "dificultad_lexica": "Dificultad (1–5)",
                        "prob_avanzado_%": st.column_config.ProgressColumn(
                            "Prob. Avanzado",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                        ),
                        "nivel_predicho": "Clasificación General",
                        "subnivel_estimado": "Nivel MCER Estimado",
                        "certeza_%": "Certeza",
                    }
                )

                # Botón para descargar el reporte CSV
                csv_bytes = df_predicho.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Exportar Reporte de Resultados (.csv)",
                    data=csv_bytes,
                    file_name="reporte_prediccion_lexiaadapt.csv",
                    mime="text/csv",
                )

                # Sección de explicabilidad con Gemini sobre un caso
                st.write("")
                st.markdown("##### ✨ Análisis Ejecutivo y Recomendación Pedagógica (IA):")
                col_id = "titulo" if "titulo" in df_predicho.columns else ("caso" if "caso" in df_predicho.columns else "id")
                opciones_casos = df_predicho[col_id].tolist()
                caso_sel = st.selectbox("Selecciona una lectura evaluada para inspeccionar su diagnóstico:", opciones_casos, key="select_caso_gemini")

                fila_sel = df_predicho[df_predicho[col_id] == caso_sel].iloc[0]

                if "explicaciones_por_caso" not in st.session_state:
                    st.session_state["explicaciones_por_caso"] = {}

                if hay_ia_generativa_disponible():
                    cj_b1, cj_b2 = st.columns([1, 1])
                    with cj_b1:
                        if st.button(f"✨ Generar Análisis IA para '{caso_sel}'", key=f"btn_justificar_{hash(caso_sel)}"):
                            with st.spinner(f"Gemini está formulando la síntesis ejecutiva para '{caso_sel}'..."):
                                desc = str(fila_sel.get("lectura", fila_sel.get("descripcion", fila_sel.get("titulo", f"Lectura '{caso_sel}'"))))
                                st.session_state["explicaciones_por_caso"][caso_sel] = explicar_prediccion_ia(
                                    desc[:300],
                                    fila_sel["nivel_predicho"],
                                    int(fila_sel["conteo_palabras"]),
                                    float(fila_sel["dificultad_lexica"]),
                                    float(fila_sel["certeza_%"]) / 100.0
                                )
                                st.rerun()

                    with cj_b2:
                        if caso_sel in st.session_state["explicaciones_por_caso"]:
                            if st.button("🔄 Regenerar Análisis", key=f"btn_regen_{hash(caso_sel)}"):
                                with st.spinner("Regenerando síntesis ejecutiva..."):
                                    desc = str(fila_sel.get("lectura", fila_sel.get("descripcion", fila_sel.get("titulo", f"Lectura '{caso_sel}'"))))
                                    st.session_state["explicaciones_por_caso"][caso_sel] = explicar_prediccion_ia(
                                        desc[:300],
                                        fila_sel["nivel_predicho"],
                                        int(fila_sel["conteo_palabras"]),
                                        float(fila_sel["dificultad_lexica"]),
                                        float(fila_sel["certeza_%"]) / 100.0
                                    )
                                    st.rerun()

                    if caso_sel in st.session_state["explicaciones_por_caso"]:
                        st.markdown(
                            f"""<div style="background:#F8FAFC; border:1px solid #CBD5E1; border-left:5px solid #0F766E; padding:1rem 1.4rem; border-radius:10px; margin-top:0.8rem; font-size:0.93rem; line-height:1.55; color:#1E293B;">
<div style="color:#0F766E; font-size:1.02rem; font-weight:700; margin-bottom:0.55rem;">Diagnóstico Pedagógico por IA — '{caso_sel}':</div>
{st.session_state['explicaciones_por_caso'][caso_sel]}
</div>""",
                            unsafe_allow_html=True
                        )
                    else:
                        st.caption(f"💡 Haz clic en el botón superior para generar el diagnóstico pedagógico de '{caso_sel}'.")
                else:
                    st.caption("ℹ️ Clave de API generativa no configurada; módulo de diagnóstico IA inactivo.")

                # =====================================================================
                # VISTA PEDAGÓGICA INTEGRADA: LECTURA BILINGÜE Y TUTOR DEL CASO EVALUADO
                # =====================================================================
                st.write("")
                st.markdown("---")
                st.markdown(f"#### 📖 Lector Interactivo y Asistencia Bilingüe: '{caso_sel}'")
                st.caption(f"Nivel asignado por la IA: **{fila_sel['subnivel_estimado']}** | Categoría: **{fila_sel.get('categoria', 'General')}**")

                texto_eval_en = str(fila_sel.get("lectura", fila_sel.get("texto_ingles", fila_sel.get("descripcion", "Texto en proceso de carga."))))

                col_lec_eval, col_met_eval = st.columns([2, 1], gap="large")

                with col_lec_eval:
                    tab_en_eval, tab_es_eval = st.tabs(["🇬🇧 Lectura en Inglés", "🇪🇸 Traducción al Español"])

                    with tab_en_eval:
                        st.markdown(
                            f"""<div style="font-family:'Lexend',sans-serif; font-size:19px; line-height:1.9; color:#0F172A; background:#FFFFFF; padding:1.8rem 2rem; border-radius:16px; border:2px solid #E2E8F0; border-left:8px solid #2563EB; box-shadow:0 4px 15px rgba(0,0,0,0.04); margin-top:0.6rem;">
    {texto_eval_en}
</div>""",
                            unsafe_allow_html=True
                        )

                    with tab_es_eval:
                        clave_trad_eval = f"trad_eval_{hash(texto_eval_en)}"
                        if clave_trad_eval not in st.session_state:
                            with st.spinner("Generando traducción al español con IA..."):
                                res_t = traducir_texto_espanol(caso_sel, texto_eval_en)
                                # Solo almacenar en cache si fue exitosa
                                if res_t and not res_t.startswith("⚠️") and not res_t.startswith("Error") and "alta demanda" not in res_t:
                                    st.session_state[clave_trad_eval] = res_t
                                else:
                                    trad_eval_txt = res_t
                        else:
                            trad_eval_txt = st.session_state[clave_trad_eval]

                        if clave_trad_eval in st.session_state:
                            st.markdown(
                                f"""<div style="font-family:'Lexend',sans-serif; font-size:19px; line-height:1.9; color:#064E3B; background:#F8FCF9; padding:1.8rem 2rem; border-radius:16px; border:2px solid #BBF7D0; border-left:8px solid #10B981; box-shadow:0 4px 15px rgba(0,0,0,0.04); margin-top:0.6rem;">
    {st.session_state[clave_trad_eval]}
</div>""",
                                unsafe_allow_html=True
                            )
                        else:
                            st.warning(f"⚠️ {trad_eval_txt}")
                            if st.button("🔄 Reintentar Traducción con IA", key=f"btn_reint_{hash(caso_sel)}"):
                                st.session_state.pop(clave_trad_eval, None)
                                st.rerun()

                    st.write("")
                    # Resumen guiado en 3 puntos clave
                    clave_res_eval = f"resumen_eval_{hash(texto_eval_en)}"
                    with st.expander("💡 ¿No entendiste la idea general? Ver Resumen en 3 Puntos Clave"):
                        if st.button("Generar Resumen Guiado", key=f"btn_res_eval_{hash(caso_sel)}"):
                            with st.spinner("El tutor está extrayendo los puntos clave..."):
                                st.session_state[clave_res_eval] = generar_resumen_espanol(caso_sel, texto_eval_en)
                        if clave_res_eval in st.session_state:
                            st.markdown(st.session_state[clave_res_eval])

                    st.write("")
                    # Asistente interactivo / Chat con LexiaTutor
                    st.markdown("#### 💬 Asistente Interactivo (LexiaTutor)")
                    st.caption("Resuelve tus dudas sobre esta lectura evaluada:")
                    clave_chat_eval = f"chat_eval_{hash(texto_eval_en)}"
                    if clave_chat_eval not in st.session_state:
                        st.session_state[clave_chat_eval] = []

                    # Contenedor de chat con scroll
                    chat_cont_eval = st.container(height=280)
                    with chat_cont_eval:
                        if not st.session_state[clave_chat_eval]:
                            st.info("👋 ¡Hola! Soy LexiaTutor, tu asistente académico. Puedes consultarme sobre vocabulario, gramática o el sentido de cualquier pasaje de este texto.")
                        for msj in st.session_state[clave_chat_eval]:
                            with st.chat_message(msj["rol"]):
                                st.markdown(msj["texto"])

                    if prompt_chat := st.chat_input("Escribe tu pregunta sobre esta lectura aquí...", key=f"input_chat_eval_{hash(caso_sel)}"):
                        st.session_state[clave_chat_eval].append({"rol": "user", "texto": prompt_chat})
                        with st.spinner("LexiaTutor está pensando..."):
                            resp_tutor = responder_duda_estudiante(prompt_chat, caso_sel, texto_eval_en)
                            st.session_state[clave_chat_eval].append({"rol": "assistant", "texto": resp_tutor})
                        st.rerun()

                with col_met_eval:
                    st.markdown("#### 📊 Información de Lectura")
                    t_est = round(max(0.5, fila_sel["conteo_palabras"] / 90.0), 1)
                    st.markdown(
                        f"""<div class="metric-card-edu">
    <div class="title">⏱️ TIEMPO ESTIMADO DE LECTURA</div>
    <div class="val">~{t_est} minutos</div>
</div>
<div class="metric-card-edu">
    <div class="title">📄 EXTENSIÓN DEL TEXTO</div>
    <div class="val">{fila_sel['conteo_palabras']} palabras</div>
</div>
<div class="metric-card-edu">
    <div class="title">🎯 NIVEL SUGERIDO</div>
    <div class="val">{fila_sel['subnivel_estimado']}</div>
</div>
<div class="metric-card-edu">
    <div class="title">🌟 COMPLEJIDAD LÉXICA</div>
    <div class="val">{fila_sel['dificultad_lexica']} / 5.0</div>
</div>""",
                        unsafe_allow_html=True
                    )

                    dif_p = float(fila_sel["dificultad_lexica"]) / 5.0
                    st.caption("Progreso de Complejidad:")
                    st.progress(min(1.0, max(0.1, dif_p)))

                    mision_txt = obtener_mision_pedagogica(fila_sel["subnivel_estimado"][:2])
                    st.markdown(
                        f"""<div class="pedagogical-box">
    <h5>🎯 Meta de Aprendizaje</h5>
    <p>{mision_txt}</p>
</div>""",
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"Error durante la inferencia: {e}")

    # ------------------ PESTAÑA 2: ANALÍTICA DEL MODELO Y PROYECCIÓN ------------------
    with tab_diagnostico:
        st.markdown("#### 📊 Analítica Predictiva y Frontera de Decisión")
        st.write(
            "Visualiza la separación matemática entre niveles lingüísticos, el catálogo base de referencia y la proyección de los documentos evaluados:"
        )

        # Dataset base histórico de referencia colapsable
        with st.expander("📁 Explorar Catálogo Base de Referencia (40 Lecturas Graduadas)", expanded=False):
            st.markdown(
                "El motor predictivo se encuentra calibrado sobre un corpus equilibrado de **40 lecturas** graduadas según "
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
        col_diag1, col_diag2 = st.columns([1.1, 0.9], gap="large")

        with col_diag1:
            st.markdown("##### 1. Mapa de Proyección sobre la Frontera de Decisión")
            df_lote_act = st.session_state.get("df_lote_subido")
            fig_proyeccion = generar_figura_proyeccion(df_lote_act)
            st.pyplot(fig_proyeccion)

            st.markdown(
                """<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:0.85rem 1rem; margin-top:0.4rem; font-size:0.86rem; color:#334155; line-height:1.6;">
    <strong>Interpretación Visual:</strong> Los círculos azules (Básico A1/A2) y rojos (Avanzado B1/B2) representan el corpus base de referencia. La línea discontinua marca la frontera de decisión (P = 50%). Las <strong>estrellas doradas</strong> proyectan los nuevos documentos evaluados en tiempo real.
</div>""",
                unsafe_allow_html=True
            )

        with col_diag2:
            st.markdown("##### 2. Matriz de Rendimiento en Validación")
            ruta_matriz = os.path.join(os.path.dirname(__file__), "..", "graficos", "matriz_confusion_sklearn.png")
            if os.path.exists(ruta_matriz):
                st.image(ruta_matriz, width=380)

            st.markdown(
                """<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:0.85rem 1rem; margin-top:0.4rem; font-size:0.86rem; color:#334155; line-height:1.6;">
    <strong>Métricas de Validación (8 lecturas de prueba):</strong> Exactitud del 100%, F1-Score = 1.0, cero falsos positivos y cero falsos negativos.<br><br>
    <strong>Parámetros del Clasificador Lineal ($w_1 x_1 + w_2 x_2 + b = 0$):</strong><br>
    • $w_1 = +1.64$ (Extensión): Mayor cantidad de palabras incrementa la probabilidad de nivel avanzado.<br>
    • $w_2 = +1.50$ (Dificultad léxica): Mayor complejidad sintáctica exige mayor competencia lectora.<br>
    • $b = +0.08$ (Sesgo balanceado del umbral).
</div>""",
                unsafe_allow_html=True
            )

# ------------------ FOOTER CORPORATIVO MINIMALISTA ------------------
st.markdown(
    """<div style="margin-top:3rem; padding:1.4rem 1rem 0.6rem 1rem; border-top:1px solid #E2E8F0; text-align:center; color:#64748B; font-size:0.84rem; line-height:1.7;">
    <div style="color:#0F2942; font-weight:700; font-size:0.9rem; margin-bottom:0.2rem;">
        📖 LexiaAdapt © 2026 · Plataforma EdTech de Evaluación Lingüística
    </div>
    <div>
        Ingeniería &amp; IA: <strong style="color:#334155;">Jhon Esteban Molina</strong> · <strong style="color:#334155;">Heiber Lozano</strong>
        &nbsp;|&nbsp; Respaldo Institucional: <strong style="color:#0F2942;">COTECNOVA</strong>
        &nbsp;|&nbsp; <a href="https://github.com/moliech/IA_TutorIngles" target="_blank" rel="noopener noreferrer" style="color:#2563EB; text-decoration:none; font-weight:600;">Repositorio en GitHub ↗</a>
    </div>
</div>""",
    unsafe_allow_html=True
)




