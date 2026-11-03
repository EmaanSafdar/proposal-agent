import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-3.5-flash-lite"
_client = None


def client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def generate_structured(system: str, prompt: str, schema, retries: int = 4):
    """One LLM call that must return JSON matching a Pydantic schema."""
    for attempt in range(retries):
        try:
            res = client().models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.1,
                ),
            )
            if res.parsed is None:
                raise ValueError("Model returned no parsable JSON")
            return res.parsed
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))  # backoff for 429 / transient errors
def generate_text(system: str, prompt: str, temperature: float = 0.3, retries: int = 4) -> str:
    """One LLM call returning plain text."""
    for attempt in range(retries):
        try:
            res = client().models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system, temperature=temperature),
            )
            if not res.text:
                raise ValueError("Empty response")
            return res.text.strip()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))