import os
import json
import re
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(title="EduGenie AI", version="2.0.0")

# Mount static and template directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Gemini Client
api_key = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=api_key) if api_key else None
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Helper function to sanitize JSON strings
def clean_json_block(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text.strip()

class QueryRequest(BaseModel):
    prompt: str

def call_gemini(system_instruction: str, user_prompt: str) -> str:
    if not client:
        raise HTTPException(status_code=500, detail="Gemini API Key is missing.")
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            )
        )
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/explain")
async def explain_concept(req: QueryRequest):
    system_instruction = "Explain the provided topic in simple, beginner-friendly language using analogies."
    return {"status": "success", "result": call_gemini(system_instruction, req.prompt)}

@app.post("/qa")
async def answer_question(req: QueryRequest):
    system_instruction = "You are EduGenie. Answer the user's question with factual precision."
    return {"status": "success", "result": call_gemini(system_instruction, req.prompt)}

@app.post("/quiz")
async def generate_quiz(req: QueryRequest):
    system_instruction = (
        "Generate exactly 3 multiple-choice questions (MCQs) with 4 options. "
        "Return strictly a raw JSON array: "
        "[{\"id\": 1, \"question\": \"...\", \"options\": [\"A\", \"B\", \"C\", \"D\"], \"correct_answer\": \"A\", \"explanation\": \"...\"}]"
    )
    raw_response = call_gemini(system_instruction, req.prompt)
    try:
        return {"status": "success", "quiz": json.loads(clean_json_block(raw_response))}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse JSON.")

@app.post("/summarize")
async def summarize_text(req: QueryRequest):
    system_instruction = "Condense the provided passage into key bullet points and a short summary."
    return {"status": "success", "result": call_gemini(system_instruction, req.prompt)}

@app.post("/learn/recommendations")
async def learning_path(req: QueryRequest):
    system_instruction = "Provide a structured learning path from Beginner to Advanced for the specified topic with timelines."
    return {"status": "success", "result": call_gemini(system_instruction, req.prompt)}