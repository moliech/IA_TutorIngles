"""
Modulo de Estilos y Ergonomia Visual - LexiaAdapt
Inyeccion de tokens Visuales Y Badges Accesibles WCAG AAA
"""

import streamlit as st

def aplicar_tema_lexia():
    """Inyectar tipografias y reglas de diseño editorial en Streamlit."""
    custom_css = """

    """

    st.markdown(custom_css, unsafe_allow_html=True)

def render_badge(nivel: str, confianza: float = None)-> str:
    """Retoma etiqueta HTML con color y simbolo CEFR correspondiente."""
    glyphs = {"A1": chr(0x25c9), "A2": chr(0x25c8), "B1": chr(0x25B2), "B2": chr(0x2756)}
    lvl = nivel.upper()
    g = glyphs.get(lvl, chr(0x25c9))
    conf_txt = f" {chr(0x2022)} {int(confianza *100)}% IA" if confianza is not None else ""
    return f'{g} MCER {lvl}{conf_txt}'
