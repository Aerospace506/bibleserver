from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import logging
import asyncio
import re
import httpx
import struct
from time import time
from urllib.parse import quote_plus
import uvicorn
from backend.config import OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BibleTranscriber")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=OPENAI_API_KEY)
BIBLE_API_URL = "https://bible-api.com/{}?translation={}"

# Validated translations from bible-api.com documentation
SUPPORTED_TRANSLATIONS = ["kjv", "bbe", "web", "almeida", "asv", "darby", "ylt"]

SUPPORTED_LANGUAGES = ["en", "es", "pt", "ro", "la", "sr", "de"]



VERSE_PATTERN = r"""
(?ixu)
\b
(  # Book (Group 1)
    (?:[1-3]?(?:st|nd|rd|th)?\s?[A-Za-zÀ-ÿ]+) |
    (?:First|Second|Third)\s+[A-Za-zÀ-ÿ]+
)
\s+
(  # Chapter (Group 2)
    \d+
)
\s*[:\.\s-]*\s*
(  # Verse (Group 3)
    \d+(?:\s*(?:-|to|through)\s*\d+)?
)
""".strip()


class AudioProcessor:
    def __init__(self):
        self.buffer = bytearray()
        self.sample_rate = 16000
        self.channels = 1
        self.sample_width = 2

    def add_pcm_chunk(self, chunk: bytes):
        self.buffer.extend(chunk)

    def create_wav(self):
        data_size = len(self.buffer)
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF', 36 + data_size, b'WAVE', b'fmt ',
            16, 1, 1, 16000, 32000, 2, 16,
            b'data', data_size
        )
        return header + self.buffer


async def fetch_bible_verse(reference: str, translation: str) -> dict:
    try:
        cleaned = re.sub(r'\b(chapter|ch|verses?|vs?|vv?|pt|verse|v)\b', '', reference, flags=re.I)
        cleaned = re.sub(r'[^\w\sÀ-ÿ:-]', '', cleaned, flags=re.UNICODE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned = re.sub(r'(?<=\d)(?=[A-Za-zÀ-ÿ])', ' ', cleaned, flags=re.UNICODE)
        cleaned = re.sub(r'(?<=[A-Za-zÀ-ÿ])(?=\d)', ' ', cleaned, flags=re.UNICODE)

        logger.info(f"Fetching: {cleaned} ({translation})")
        async with httpx.AsyncClient() as client:
            encoded_ref = quote_plus(cleaned)
            url = BIBLE_API_URL.format(encoded_ref, translation)
            response = await client.get(url)

            if response.status_code == 404:
                return {"error": "Translation or verse not found"}

            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.error(f"Bible API error: {data['error']}")
                return {"error": data["error"]}

            return data

    except httpx.HTTPStatusError as e:
        logger.error(f"Bible API HTTP error: {e.response.text}")
        return {"error": f"API Error: {e.response.status_code}"}
    except Exception as e:
        logger.error(f"Bible API failure: {str(e)}")
        return {"error": "Verse fetch failed"}


async def get_ai_verse_reference(context: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "You are a Bible reference detector. Return ONLY the reference in standard format."
            }, {
                "role": "user",
                "content": f"Extract Bible reference from: {context}"
            }],
            temperature=0.1,
            max_tokens=20
        )
        content = response.choices[0].message.content
        content = re.sub(r'\b(verses?|vs?|vv?)\b', ':', content)
        match = re.search(VERSE_PATTERN, content, re.X | re.UNICODE)
        if match:
            return f"{match.group(1).strip()} {match.group(2).strip()}:{match.group(3).strip()}"
        return None
    except Exception as e:
        logger.error(f"OpenAI error: {str(e)}")
        return None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")

    translation = "kjv"
    language = "en"
    audio_processor = AudioProcessor()
    context_history = []
    connection_open = True
    sent_verses = {}

    try:
        settings = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        translation = settings.get("translation", "kjv")
        language = settings.get("language", "en")

        if translation not in SUPPORTED_TRANSLATIONS:
            raise WebSocketDisconnect(code=1008, reason="Invalid translation")
        if language not in SUPPORTED_LANGUAGES:
            raise WebSocketDisconnect(code=1008, reason="Invalid language")

        logger.info(f"Valid settings: {translation}/{language}")

    except (asyncio.TimeoutError, WebSocketDisconnect) as e:
        logger.warning(f"Settings error: {str(e)}")
        await websocket.close(code=1008)
        return
    except Exception as e:
        logger.error(f"Connection error: {str(e)}")
        await websocket.close(code=1011)
        return

    try:
        while connection_open:
            try:
                pcm_data = await asyncio.wait_for(websocket.receive_bytes(), timeout=2.0)
                audio_processor.add_pcm_chunk(pcm_data)

                if len(audio_processor.buffer) >= 16000 * 2 * 3:
                    wav_data = audio_processor.create_wav()

                    try:
                        whisper_lang = language if language in ["en", "es", "pt"] else "en"
                        transcription = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=("audio.wav", wav_data, "audio/wav"),
                            temperature=0.2,
                            prompt="Religious context with Bible references",
                            language=whisper_lang
                        )
                    except Exception as e:
                        logger.error(f"Transcription error: {str(e)}")
                        continue

                    clean_text = re.sub(r'[^\w\s.:-À-ÿ]', '', transcription.text, flags=re.UNICODE)
                    logger.info(f"Transcription: {clean_text}")
                    context_history = [clean_text] + context_history[:2]

                    responses = []
                    try:
                        matches = re.finditer(VERSE_PATTERN, clean_text, re.X | re.UNICODE)
                        for match in matches:
                            ref = f"{match.group(1).strip()} {match.group(2).strip()}:{match.group(3).strip()}"
                            verse_data = await fetch_bible_verse(ref, translation)
                            if "text" in verse_data:
                                ref_key = verse_data["reference"].lower()
                                if ref_key not in sent_verses or (time() - sent_verses[ref_key] > 300):
                                    responses.append({
                                        "type": "verse",
                                        "source": "direct",
                                        "reference": verse_data["reference"],
                                        "text": verse_data["text"],
                                        "translation": translation
                                    })
                                    sent_verses[ref_key] = time()
                            elif "error" in verse_data:
                                responses.append({
                                    "type": "error",
                                    "message": verse_data["error"]
                                })
                    except Exception as e:
                        logger.error(f"Processing error: {str(e)}")

                    if not responses:
                        ai_ref = await get_ai_verse_reference(" ".join(context_history))
                        if ai_ref:
                            verse_data = await fetch_bible_verse(ai_ref, translation)
                            if "text" in verse_data:
                                ref_key = verse_data["reference"].lower()
                                if ref_key not in sent_verses or (time() - sent_verses[ref_key] > 300):
                                    responses.append({
                                        "type": "verse",
                                        "source": "ai",
                                        "reference": verse_data["reference"],
                                        "text": verse_data["text"],
                                        "translation": translation
                                    })
                                    sent_verses[ref_key] = time()

                    if responses:
                        await websocket.send_json(responses)

                    audio_processor = AudioProcessor()

            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                connection_open = False
                break

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        if connection_open:
            await websocket.close()
        logger.info("Connection closed")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)