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

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))



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


INTERVIEW_SYSTEM_PROMPT = """
You are Ranjith's interview support assistant.

Context about Ranjith:
- Senior Python Backend Engineer
- 8+ years Python experience
- Strong in FastAPI, Django, REST APIs, microservices
- Worked in telecom domain at Nokia
- Experience with AWS: EC2, Lambda, RDS/Aurora PostgreSQL
- Good with SQL/PostgreSQL, Redis basics, Docker, CI/CD
- Preparing for Senior Python Backend / AWS / System Design interviews

Your job:
- Answer interview questions clearly and confidently
- Prefer practical senior-engineer style answers
- Keep answers short unless asked for detailed explanation
- If audio transcription has mistakes, infer the likely technical question
- If unclear, give the most likely answer and mention the assumption
- Use examples from Python backend, FastAPI, microservices, AWS, PostgreSQL
- Avoid fake project details
- Make the answer sound natural for an interview

Answer format:
1. Direct answer
2. Small practical example
3. Interview-ready closing line
"""


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
                model="whisper-large-v3-turbo",
                response_format="json",
                language="en",
                prompt="Technical interview audio about Python, FastAPI, Django, AWS, system design, microservices, PostgreSQL, Redis, Docker, CI/CD."
            )

        user_text = transcription.text
        print("Transcribed:", user_text)

        # ANSWER (not rephrase)
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": INTERVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ],
            temperature=0.4
        )

        reply = response.choices[0].message.content

        return {
            "transcribed_text": user_text,
            "reply": reply
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}