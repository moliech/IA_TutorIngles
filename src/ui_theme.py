"""
Modulo de Estilos y Ergonomia Visual - LexiaAdapt
Inyeccion de Tokens Visuales, Tipografias Educativas y Badges Accesibles WCAG AAA
"""

import streamlit as st

def aplicar_tema_lexia():
    """Inyecta tipografias y reglas de diseño editorial y educativo en Streamlit."""
    custom_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Lexend:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

        :root {
            --brand-primary: #1E3A8A;
            --brand-accent: #0D9488;
            --brand-indigo: #4F46E5;
            --bg-page: #F8FAFC;
            --bg-card: #FFFFFF;
            --text-main: #0F172A;
            --text-muted: #334155;
            --border-soft: #E2E8F0;

            --cefr-a1-txt: #065F46; --cefr-a1-bg: #D1FAE5; --cefr-a1-br: #34D399;
            --cefr-a2-txt: #115E59; --cefr-a2-bg: #CCFBF1; --cefr-a2-br: #2DD4BF;
            --cefr-b1-txt: #92400E; --cefr-b1-bg: #FEF3C7; --cefr-b1-br: #FBBF24;
            --cefr-b2-txt: #3730A3; --cefr-b2-bg: #E0E7FF; --cefr-b2-br: #818CF8;
        }

        .stApp {
            background-color: var(--bg-page);
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: var(--text-main);
        }

        /* Banner Principal de la Plataforma */
        .lex-banner {
            background: linear-gradient(135deg, #0F2942 0%, #1E3A8A 55%, #0F766E 100%);
            border-radius: 18px;
            padding: 1.8rem 2.4rem;
            color: #FFFFFF;
            box-shadow: 0 10px 25px -5px rgba(15, 41, 66, 0.2);
            margin-bottom: 1.5rem;
        }
        .lex-banner h1 {
            color: #FFFFFF !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 2.3rem !important;
            font-weight: 800 !important;
            margin: 0 !important;
            letter-spacing: -0.02em;
        }
        .lex-banner p {
            color: #F1F5F9 !important;
            font-size: 1.15rem !important;
            font-weight: 500 !important;
            margin: 0.4rem 0 0.5rem 0 !important;
            line-height: 1.5;
        }
        .lex-banner .sub-tag {
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.35);
            border-radius: 999px;
            padding: 0.25rem 0.9rem;
            font-size: 0.88rem;
            font-weight: 600;
            color: #FFFFFF;
        }

        /* Subtitulos con alta legibilidad */
        .section-subtitle {
            font-size: 1.15rem !important;
            font-weight: 600 !important;
            color: #1E293B !important;
            margin: 0.3rem 0 1.2rem 0 !important;
            line-height: 1.5 !important;
        }

        /* Tarjetas Didacticas de Metricas */
        .metric-card-edu {
            background-color: #FFFFFF;
            border: 1px solid var(--border-soft);
            border-radius: 14px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            margin-bottom: 0.85rem;
        }
        .metric-card-edu .title {
            font-size: 0.85rem;
            font-weight: 700;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .metric-card-edu .val {
            font-size: 1.35rem;
            font-weight: 800;
            color: #0F2942;
            margin-top: 0.25rem;
        }

        /* Badges CEFR */
        .cefr-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.88rem;
            font-weight: 700;
        }

        .cefr-a1 { background-color: var(--cefr-a1-bg); color: var(--cefr-a1-txt); border: 1.5px solid var(--cefr-a1-br); }
        .cefr-a2 { background-color: var(--cefr-a2-bg); color: var(--cefr-a2-txt); border: 1.5px solid var(--cefr-a2-br); }
        .cefr-b1 { background-color: var(--cefr-b1-bg); color: var(--cefr-b1-txt); border: 1.5px solid var(--cefr-b1-br); }
        .cefr-b2 { background-color: var(--cefr-b2-bg); color: var(--cefr-b2-txt); border: 1.5px solid var(--cefr-b2-br); }

        /* Guia Pedagogica */
        .pedagogical-box {
            background: linear-gradient(135deg, #EFF6FF 0%, #F0FDFA 100%);
            border: 1.5px solid #BFDBFE;
            border-radius: 14px;
            padding: 1.25rem;
            margin-top: 1rem;
        }
        .pedagogical-box h5 {
            color: #1E3A8A;
            font-weight: 800;
            margin: 0 0 0.5rem 0;
            font-size: 1.05rem;
        }
        .pedagogical-box p {
            color: #1E293B;
            font-size: 0.95rem;
            line-height: 1.6;
            margin: 0;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def render_badge(nivel: str, confianza: float = None) -> str:
    """Retorna etiqueta HTML con color y simbolo CEFR correspondiente."""
    glyphs = {"A1": "◉", "A2": "◈", "B1": "▲", "B2": "❖"}
    lvl = str(nivel).upper()
    g = glyphs.get(lvl, "◉")
    conf_txt = f" • {int(confianza * 100)}% Certeza IA" if confianza is not None else ""
    return f'<span class="cefr-badge cefr-{lvl.lower()}">{g} NIVEL MCER {lvl}{conf_txt}</span>'
