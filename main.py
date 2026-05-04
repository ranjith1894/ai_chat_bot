import os

import tempfile
from fastapi import FastAPI, UploadFile, File,  Form
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional

from fastapi.middleware.cors import CORSMiddleware

from groq import Groq

from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse


load_dotenv()


app = FastAPI(title="Ranjith Rephrase Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)



class ChatRequest(BaseModel):
    message: str
    mode: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str

@app.get("/")
def health():
    return {"status": "running"}

def get_mode_instruction(mode: Optional[str]) -> str:
    if not mode:
        return "Rewrite in a natural, human-like way."

    mode_map = {
        "polite": "Rewrite politely.",
        "professional": "Rewrite in a professional tone.",
        "short": "Rewrite in a very short and concise way.",
        "friendly": "Rewrite in a friendly tone.",
        "hr": "Rewrite as a strong HR/interview response.",
        "grammar": "Fix grammar and make it correct."
    }

    return mode_map.get(mode.lower(), "Rewrite in a natural, human-like way.")
    
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    mode_instruction = get_mode_instruction(req.mode)

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
You are an English writing assistant.

Task:
Rewrite the message clearly and naturally.

Rules:
- Keep meaning same
- Do not add extra details
- Keep it short

{mode_instruction}
"""),
        ("human", "{message}")
    ])

    chain = prompt | llm

    result = chain.invoke({
        "message": req.message
    })

    return {"reply": result.content.strip()}



@app.post("/whatsapp")
async def whatsapp_webhook(Body: str = Form(...)):
    prompt_req = ChatRequest(message=Body)

    result = chat(prompt_req)

    twilio_response = MessagingResponse()
    twilio_response.message(result["reply"])

    return PlainTextResponse(str(twilio_response), media_type="application/xml")


@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...)):
    try:
        print("API HIT ✅")

        # read file
        content = await file.read()

        if not content:
            return {"error": "Empty file"}

        # save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp:
            temp.write(content)
            temp_path = temp.name

        print("File saved:", temp_path)

        # TRANSCRIBE
        with open(temp_path, "rb") as audio:
            transcription = groq_client.audio.transcriptions.create(
                file=audio,
                model="whisper-large-v3-turbo"
            )

        user_text = transcription.text
        print("Transcribed:", user_text)

        # ANSWER (not rephrase)
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Keep answers short."},
                {"role": "user", "content": user_text}
            ],
            temperature=0.5
        )

        reply = response.choices[0].message.content

        return {
            "transcribed_text": user_text,
            "reply": reply
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}