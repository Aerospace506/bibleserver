from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # Add this line
from openai import OpenAI
import logging
import asyncio
import re
import httpx
import struct
from time import time
import os

from config import OPENAI_API_KEY

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
BIBLE_API_URL = "https://bible-api.com/{}?translation=kjv"

VERSE_PATTERN = r"""
(?ix)
\b
(  # Book (Group 1)
    (?:[1-3]?(?:st|nd|rd|th)?\s?[A-Za-z]+) |
    (?:First|Second|Third)\s+[A-Za-z]+
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


async def fetch_bible_verse(reference: str) -> dict:
    try:
        cleaned = re.sub(r'\b(chapter|ch|verses?|vs?|vv?|pt|verse|v)\b', '', reference, flags=re.I)
        cleaned = re.sub(r'[^a-zA-Z0-9\s:-]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', cleaned)
        cleaned = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', cleaned)

        logger.info(f"Cleaned reference: {cleaned}")
        async with httpx.AsyncClient() as client:
            encoded_ref = cleaned.replace(" ", "%20").replace(":", "%3A")
            response = await client.get(BIBLE_API_URL.format(encoded_ref))
            return response.json()
    except Exception as e:
        logger.error(f"Bible API error: {str(e)}")
        return {"error": "Could not fetch verse"}


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
        match = re.search(VERSE_PATTERN, content, re.X)
        if match:
            book = match.group(1).strip()
            chapter = match.group(2).strip()
            verse = match.group(3).strip()
            return f"{book} {chapter}:{verse}"
        return None
    except Exception as e:
        logger.error(f"OpenAI error: {str(e)}")
        return None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")

    audio_processor = AudioProcessor()
    context_history = []
    connection_open = True
    sent_verses = {}

    try:
        while connection_open:
            try:
                pcm_data = await asyncio.wait_for(websocket.receive_bytes(), timeout=2.0)
                audio_processor.add_pcm_chunk(pcm_data)

                if len(audio_processor.buffer) >= 16000 * 2 * 3:
                    wav_data = audio_processor.create_wav()

                    try:
                        transcription = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=("audio.wav", wav_data, "audio/wav"),
                            temperature=0.2,
                            prompt="Religious context with Bible references",
                            language="en"
                        )
                    except Exception as e:
                        logger.error(f"Transcription error: {str(e)}")
                        continue

                    clean_text = re.sub(r'[^\w\s.:-]', '', transcription.text)
                    logger.info(f"Raw transcription: {clean_text}")
                    context_history.append(clean_text)

                    if len(context_history) > 3:
                        context_history.pop(0)

                    responses = []
                    try:
                        matches = re.finditer(VERSE_PATTERN, clean_text, re.X)
                        for match in matches:
                            book = match.group(1)
                            chapter = match.group(2)
                            verse = match.group(3)
                            ref = f"{book} {chapter}:{verse}"
                            verse_data = await fetch_bible_verse(ref)
                            if "text" in verse_data:
                                ref_key = verse_data["reference"].lower()
                                if ref_key not in sent_verses or (time() - sent_verses[ref_key] > 300):
                                    responses.append({
                                        "type": "verse",
                                        "source": "direct",
                                        "reference": verse_data["reference"],
                                        "text": verse_data["text"]
                                    })
                                    sent_verses[ref_key] = time()
                    except Exception as e:
                        logger.error(f"Regex error: {str(e)}")

                    if not responses:
                        ai_ref = await get_ai_verse_reference(" ".join(context_history))
                        if ai_ref:
                            verse_data = await fetch_bible_verse(ai_ref)
                            if "text" in verse_data:
                                ref_key = verse_data["reference"].lower()
                                if ref_key not in sent_verses or (time() - sent_verses[ref_key] > 300):
                                    responses.append({
                                        "type": "verse",
                                        "source": "ai",
                                        "reference": verse_data["reference"],
                                        "text": verse_data["text"]
                                    })
                                    sent_verses[ref_key] = time()

                    if responses:
                        try:
                            await websocket.send_json(responses)
                        except (RuntimeError, WebSocketDisconnect):
                            connection_open = False
                            break

                    audio_processor = AudioProcessor()

            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                connection_open = False
                break

    except Exception as e:
        logger.error(f"Connection error: {str(e)}")
    finally:
        if connection_open:
            try:
                await websocket.close(code=1000)
            except RuntimeError:
                pass
        logger.info("Connection closed")

# Mount static files PROPERLY
"""app.mount(
    "/",
    StaticFiles(
        directory=os.path.join(os.path.dirname(__file__), "static"),
        html=True
    ),
    name="static"
)"""

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8008)