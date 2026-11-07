"""
src/gemini_tutor.py

Módulo de Integración con IA Generativa (Google Gemini API)
Valor Agregado (Opcional) - Requisito de los Lineamientos del Proyecto Final
Proporciona explicaciones pedagógicas de predicciones y asistencia interactiva a estudiantes.
"""

import os
import json
import urllib.request
import urllib.error

MODELOS_CANDIDATOS = [
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest"
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
            with urllib.request.urlopen(req, timeout=14) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return res["candidates"][0]["content"]["parts"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 404 or e.code == 503:
                continue
            return f"Error HTTP {e.code} al consultar Gemini API."
        except Exception as e:
            continue

    return "No fue posible conectar con el servicio de IA generativa en este momento (alta demanda)."

def explicar_prediccion_ia(texto: str, nivel_predicho: str, palabras: int, dificultad: float, confianza: float) -> str:
    """Genera una explicación pedagógica en lenguaje natural sobre por qué el texto tiene ese nivel."""
    system_prompt = (
        "Eres un pedagogo experto en lingüística aplicada y enseñanza del inglés como segunda lengua "
        "para estudiantes del Valle del Cauca, Colombia. Tu rol es explicar de forma clara, motivadora "
        "y científica por qué un modelo de Machine Learning clasificó un texto en un determinado nivel CEFR."
    )
    prompt = (
        f"El modelo de Machine Learning clasificó el siguiente texto como '{nivel_predicho}' "
        f"con una confianza del {confianza*100:.1f}%. Métricas calculadas: {palabras} palabras "
        f"y dificultad léxica de {dificultad:.2f}/5.0.\n\n"
        f"Texto analizado:\n\"\"\"{texto[:500]}\"\"\"\n\n"
        "Redacta una explicación de 2 a 3 párrafos cortos en español explicando:\n"
        "1. Las razones lingüísticas (longitud, conectores, vocabulario, sintaxis) que justifican este nivel.\n"
        "2. Qué tipo de reto cognitivo representa para un estudiante de Cartago.\n"
        "3. Una recomendación pedagógica puntual para trabajar esta lectura."
    )
    return consultar_gemini(prompt, system_instruction=system_prompt)

def responder_duda_estudiante(pregunta: str, titulo: str, texto: str) -> str:
    """Actúa como tutor pedagógico respondiendo dudas del estudiante sobre la lectura."""
    system_prompt = (
        "Eres 'LexiaTutor', un tutor virtual amigable, claro y paciente de inglés para estudiantes de "
        "secundaria y educación técnica de Cartago, Valle del Cauca. Respondes en español con explicaciones "
        "sencillas, ejemplos claros y tono motivador, enfocado en ayudar al estudiante a comprender la lectura."
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
