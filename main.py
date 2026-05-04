import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional

from fastapi.middleware.cors import CORSMiddleware

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
