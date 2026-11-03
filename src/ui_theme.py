"""
Modulo de Estilos y Ergonomia Visual - LexiaAdapt
Inyeccion de Tokens Visuales y Badges Accesibles WCAG AAA
"""

import streamlit as st

def aplicar_tema_lexia():
    """Inyecta tipografias y reglas de diseño editorial en Streamlit."""
    custom_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Lexend:wght@300;400;500;600&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

        :root {
            --lex-navy: #0F2942;
            --lex-blue: #1E3A8A;
            --lex-teal: #0F766E;
            --lex-paper: #FBF9F5;
            --lex-card: #FFFFFF;
            --lex-text: #0F172A;
            --lex-muted: #475569;
            --lex-border: #E2E8F0;

            --cefr-a1-txt: #065F46; --cefr-a1-bg: #D1FAE5; --cefr-a1-br: #6EE7B7;
            --cefr-a2-txt: #0F5B54; --cefr-a2-bg: #CCFBF1; --cefr-a2-br: #5EEAD4;
            --cefr-b1-txt: #78350F; --cefr-b1-bg: #FEF3C7; --cefr-b1-br: #FCD34D;
            --cefr-b2-txt: #312E81; --cefr-b2-bg: #E0E7FF; --cefr-b2-br: #A5B4FC;
        }

        .stApp {
            background-color: var(--lex-paper);
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: var(--lex-text);
        }

        .reading-container {
            background-color: var(--lex-card);
            border: 1px solid var(--lex-border);
            border-radius: 14px;
            padding: 2.25rem;
            box-shadow: 0 4px 15px -2px rgba(15, 41, 66, 0.05);
            max-width: 72ch;
            margin: 0 auto;
        }

        .reading-body {
            font-family: 'Lexend', sans-serif;
            font-size: 18px;
            line-height: 1.85;
            letter-spacing: 0.025em;
            color: var(--lex-text);
        }

        .cefr-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            font-weight: 600;
        }

        .cefr-a1 { background-color: var(--cefr-a1-bg); color: var(--cefr-a1-txt); border: 1px solid var(--cefr-a1-br); }
        .cefr-a2 { background-color: var(--cefr-a2-bg); color: var(--cefr-a2-txt); border: 1px solid var(--cefr-a2-br); }
        .cefr-b1 { background-color: var(--cefr-b1-bg); color: var(--cefr-b1-txt); border: 1px solid var(--cefr-b1-br); }
        .cefr-b2 { background-color: var(--cefr-b2-bg); color: var(--cefr-b2-txt); border: 1px solid var(--cefr-b2-br); }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def render_badge(nivel: str, confianza: float = None) -> str:
    """Retorna etiqueta HTML con color y simbolo CEFR correspondiente."""
    glyphs = {"A1": "◉", "A2": "◈", "B1": "▲", "B2": "❖"}
    lvl = str(nivel).upper()
    g = glyphs.get(lvl, "◉")
    conf_txt = f" • {int(confianza * 100)}% IA" if confianza is not None else ""
    return f'<span class="cefr-badge cefr-{lvl.lower()}">{g} MCER {lvl}{conf_txt}</span>'
