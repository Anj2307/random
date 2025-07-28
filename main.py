# main.py

from fastapi import FastAPI, Header, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import os
import uvicorn

from faiss_pipeline import RAGRequestSchema, get_answers_from_url_and_questions

load_dotenv()

app = FastAPI()

# Mount static files and template engine
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

HACKRX_AUTH_KEY = os.getenv("HACKRX_AUTH_KEY")

# Dependency to check Authorization token
def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format.")
    token = authorization.split("Bearer ")[1]
    if token != HACKRX_AUTH_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key.")

@app.get("/", response_class=HTMLResponse)
def render_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/hackrx/run")
def hackrx_query(payload: RAGRequestSchema, _: str = Depends(verify_token)):
    try:
        results = get_answers_from_url_and_questions(str(payload.documents), payload.questions)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
