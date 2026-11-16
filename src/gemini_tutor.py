"""
src/gemini_tutor.py

Módulo de Integración con IA Generativa (Google Gemini API)
Proporciona explicaciones pedagógicas de predicciones y asistencia interactiva a estudiantes.
"""

import os
import re
import json
import urllib.request
import urllib.error

# Modelos rápidos verificados con soporte oficial generateContent
MODELOS_CANDIDATOS = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
]

def obtener_gemini_api_key() -> str:
    """Busca y retorna la clave de API desde el entorno o archivo .env."""
    key = os.environ.get("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()

    posibles_rutas = [
        os.path.join(os.path.dirname(__file__), "..", ".env"),
        os.path.join(os.path.dirname(__file__), ".env"),
        "/app/.env",
        ".env"
    ]
    for ruta in posibles_rutas:
        if os.path.exists(ruta):
            try:
                with open(ruta, "r", encoding="utf-8") as f:
                    for linea in f:
                        linea = linea.strip()
                        if linea.startswith("GEMINI_API_KEY="):
                            k = linea.split("=", 1)[1].strip().strip('"').strip("'")
                            if k:
                                return k
            except Exception:
                pass
    return None

def hay_ia_generativa_disponible() -> bool:
    """Verifica si existe una clave de API configurada."""
    k = obtener_gemini_api_key()
    return bool(k and len(k) > 10)

def consultar_gemini(prompt: str, system_instruction: str = None) -> str:
    """Ejecuta una consulta al modelo Gemini vía REST API con fallback entre modelos."""
    key = obtener_gemini_api_key()
    if not key:
        return "⚠️ Clave de API no detectada en archivo .env."

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    if system_instruction:
        payload["system_instruction"] = {
            "parts": [{"text": system_instruction}]
        }

    data_bytes = json.dumps(payload).encode("utf-8")

    for modelo in MODELOS_CANDIDATOS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={key}"
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return res["candidates"][0]["content"]["parts"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            if e.code in [404, 429, 503]:
                continue
            return f"Error HTTP {e.code} al consultar Gemini API."
        except Exception:
            continue

    return "No fue posible conectar con el servicio de IA generativa en este momento (alta demanda)."

def explicar_prediccion_ia(titulo_o_desc: str, nivel_predicho: str, palabras: int, dificultad: float, confianza: float) -> str:
    """Genera una justificación ejecutiva, directa y pedagógica en exactamente 3 viñetas con formato HTML compacto."""
    system_prompt = (
        "Eres un docente universitario experto en lingüística aplicada y evaluación pedagógica de lectura en inglés como lengua extranjera. "
        "Tu estilo es conciso, académico, claro y didáctico."
    )
    prompt = (
        f"El modelo de Machine Learning serializado (modelo_final.joblib) clasificó la lectura '{titulo_o_desc}' "
        f"como '{nivel_predicho}' con una certeza del {confianza*100:.1f}%.\n"
        f"Métricas del texto: {palabras} palabras | Dificultad léxica: {dificultad:.2f} / 5.0.\n\n"
        "Redacta tu evaluación en exactamente 3 líneas (máximo 25 palabras por línea), conservando OBLIGATORIAMENTE "
        "los 3 encabezados exactos al inicio de cada línea:\n"
        "🎯 Diagnóstico de Inferencia: <evalúa si el nivel asignado concuerda con el texto y la certeza>\n"
        "⚖️ Fundamento Cuantitativo: <explica cómo las palabras y la dificultad ubican la lectura frente a la frontera>\n"
        "💡 Recomendación Pedagógica: <orientación didáctica práctica para trabajar esta lectura en clase>"
    )
    res = consultar_gemini(prompt, system_instruction=system_prompt)
    if res and not res.startswith("⚠️") and not res.startswith("Error"):
        etiquetas_base = [
            ("🎯 Diagnóstico de Inferencia:", "🎯"),
            ("⚖️ Fundamento Cuantitativo:", "⚖️"),
            ("💡 Recomendación Pedagógica:", "💡"),
        ]
        lineas_limpias = [ln.strip().lstrip("•-*").strip() for ln in res.split("\n") if ln.strip()]
        bloques_html = []
        for idx, linea in enumerate(lineas_limpias[:3]):
            # Quitar negritas markdown si el modelo las incluyó
            linea = re.sub(r"\*\*(.*?)\*\*", r"\1", linea)
            # Asegurar que el encabezado en negrita esté presente
            etiq_esperada, icono = etiquetas_base[idx] if idx < len(etiquetas_base) else ("", "")
            if etiq_esperada and ":" in linea:
                partes = linea.split(":", 1)
                cabeza = partes[0].strip()
                cuerpo = partes[1].strip()
                if any(k in cabeza.lower() for k in ["diagnóstico", "diagnostico", "fundamento", "recomendación", "recomendacion"]):
                    linea_fmt = f"• <strong>{etiq_esperada}</strong> {cuerpo}"
                else:
                    linea_fmt = f"• <strong>{etiq_esperada}</strong> {linea.lstrip(icono).strip()}"
            elif etiq_esperada:
                linea_fmt = f"• <strong>{etiq_esperada}</strong> {linea.lstrip(icono).strip()}"
            else:
                linea_fmt = f"• {linea}"
            bloques_html.append(f'<div style="margin-bottom:0.4rem;">{linea_fmt}</div>')
        return "\n".join(bloques_html)
    return res

def responder_duda_estudiante(pregunta: str, titulo: str, texto: str) -> str:
    """Actúa como tutor pedagógico respondiendo dudas del estudiante sobre la lectura."""
    system_prompt = (
        "Eres 'LexiaTutor', un tutor virtual amigable, claro y paciente de inglés para estudiantes. "
        "Respondes en español con explicaciones sencillas, ejemplos claros y tono motivador, "
        "enfocado en ayudar al estudiante a comprender la lectura."
    )
    prompt = (
        f"El estudiante está leyendo el texto titulado '{titulo}':\n"
        f"\"\"\"{texto}\"\"\"\n\n"
        f"Pregunta del estudiante: \"{pregunta}\"\n\n"
        "Responde de forma concisa (máximo 120 palabras), clara y pedagógica para resolver su duda."
    )
    return consultar_gemini(prompt, system_instruction=system_prompt)

def generar_resumen_espanol(titulo: str, texto: str) -> str:
    """Genera un resumen estructurado en 3 viñetas en español de la lectura."""
    system_prompt = "Eres un profesor de inglés que resume textos en español para lectores principiantes e intermedios."
    prompt = (
        f"Lee este texto en inglés titulado '{titulo}':\n"
        f"\"\"\"{texto}\"\"\"\n\n"
        "Entrega un resumen en español con exactamente 3 viñetas claras que expliquen de qué trata la lectura, "
        "para que el estudiante entienda el contexto general antes de leer en inglés."
    )
    return consultar_gemini(prompt, system_instruction=system_prompt)

def traducir_texto_espanol(titulo: str, texto: str) -> str:
    """Traduce fielmente el texto de lectura en inglés al español para apoyo pedagógico del estudiante."""
    system_prompt = (
        "Eres un traductor profesional bilingüe especializado en pedagogía de lectura. "
        "Tu única labor es traducir el texto en inglés al español latinoamericano con total naturalidad y fidelidad. "
        "Prohibido incluir preámbulos, saludos, notas del traductor o frases como 'Aquí tienes la traducción'. "
        "Comienza directamente con la primera palabra del texto traducido."
    )
    prompt = (
        f"Traduce directamente este texto en inglés al español:\n\n"
        f"\"\"\"{texto}\"\"\""
    )
    res = consultar_gemini(prompt, system_instruction=system_prompt)
    if res and not res.startswith("⚠️") and not res.startswith("Error"):
        # Limpiar posibles encabezados automáticos del modelo
        res = re.sub(r"^(?:Aquí tienes|A continuación|Traducción|Esta es)[^\n]*\n+", "", res, flags=re.IGNORECASE).strip()
        if res.startswith('"') and res.endswith('"'):
            res = res[1:-1].strip()
    return res

def obtener_traduccion_con_cache(id_lectura: int, titulo: str, texto_ingles: str) -> str:
    """Busca la traducción en disco (data/traducciones_espanol.json) o la genera con Gemini y la cachea."""
    ruta_json = os.path.join(os.path.dirname(__file__), "..", "data", "traducciones_espanol.json")
    traducciones = {}
    if os.path.exists(ruta_json):
        try:
            with open(ruta_json, "r", encoding="utf-8") as f:
                traducciones = json.load(f)
        except Exception:
            traducciones = {}

    id_str = str(id_lectura)
    if id_str in traducciones and traducciones[id_str].strip():
        return traducciones[id_str]

    if not hay_ia_generativa_disponible():
        return "⚠️ Traducción no disponible (servicio de IA generativa no configurado)."

    trad = traducir_texto_espanol(titulo, texto_ingles)
    if trad and not trad.startswith("Error") and not trad.startswith("⚠️"):
        traducciones[id_str] = trad
        try:
            os.makedirs(os.path.dirname(ruta_json), exist_ok=True)
            with open(ruta_json, "w", encoding="utf-8") as f:
                json.dump(traducciones, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return trad
