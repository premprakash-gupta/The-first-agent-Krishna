import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

class QueryIn(BaseModel):
    prompt: str

class QueryOut(BaseModel):
    text: str
    search_used: bool = False

def build_agent():
    retry_config = types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504],
    )
    

    agent = Agent(
    name="Krishna_clone_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),

    description=(
        "A Krishna-persona agent that speaks with the calm, decisive, and "
        "detached tone of Krishna from the Mahabharata and Bhagavad Gita."
        "Answer in detail and use Hindi/English as per question dialect."
    ),

    instruction=(
        "Answer in detail as per question and use Hindi/English as per question dialect."
        "Respond exactly in Krishna's voice: firm, clear, detached, and rooted in dharma. "
        "No politeness, no flattery, no praise of the user's question, no emotional cushioning, "
        "no modern filler words, and no LLM-style phrases. Krishna gives verdicts, not validation. "
        "Speak with strategic clarity and unshaken authority. Use metaphors only when they sharpen dharma, "
        "For factual or current topics, use Google Search. For dilemmas, morality, or inner conflict, "
        "answer strictly as Krishna — steady, direct, and free from hesitation."
        
    ),

    tools=[google_search],
)


    runner = InMemoryRunner(agent=agent)
    return agent, runner

def ensure_api_key():
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not found. Create a .env with GOOGLE_API_KEY=...")

ensure_api_key()
agent, runner = build_agent()

app = FastAPI(title="ADK WebUI", version="0.1.0")

# Optional: silence verbose ADK app-name mismatch warning
logging.getLogger("google.adk").setLevel(logging.ERROR)

# Serve static files from webui/static at /static
app.mount("/static", StaticFiles(directory="webui/static"), name="static")

# Redirect root to the static index page
@app.get("/")
async def root_page():
    return RedirectResponse(url="/static/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryOut)
async def query(body: QueryIn):
    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    try:
        response = await runner.run_debug(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # Convert model/ADK response to plain text and detect search usage
    text = _extract_text(response)
    used_search = _extract_search_used(response)
    return QueryOut(text=text, search_used=used_search)


def _extract_text(resp) -> str:
    try:
        # google.genai Event -> Content -> parts[*].text
        content = getattr(resp, "content", None)
        if content is not None:
            parts = getattr(content, "parts", None)
            if parts:
                texts = []
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        texts.append(str(t))
                    elif isinstance(p, dict) and "text" in p:
                        texts.append(str(p["text"]))
                if texts:
                    return "\n\n".join(texts).strip()
        # direct .text
        t = getattr(resp, "text", None)
        if t:
            return str(t).strip()
        # dict style
        if isinstance(resp, dict):
            if "text" in resp:
                return str(resp["text"]).strip()
            c = resp.get("content")
            if isinstance(c, dict):
                parts = c.get("parts", [])
                texts = [str(p.get("text")).strip() for p in parts if isinstance(p, dict) and p.get("text")]
                if texts:
                    return "\n\n".join(texts)
    except Exception:
        pass
    # Regex fallback from string representation
    try:
        import re
        s = str(resp)
        matches = re.findall(r"text=(?:\"\"\"|\')([\s\S]+?)(?:\"\"\"|\')", s)
        if matches:
            return "\n\n".join([m.strip() for m in matches])
        return s
    except Exception:
        return str(resp)


def _extract_search_used(resp) -> bool:
    try:
        gm = getattr(resp, "grounding_metadata", None)
        if gm is not None:
            chunks = getattr(gm, "grounding_chunks", None)
            if chunks:
                for ch in chunks:
                    web = getattr(ch, "web", None)
                    if web is not None and getattr(web, "uri", None):
                        return True
        # alternative signal
        qs = getattr(resp, "web_search_queries", None)
        if qs:
            return True
    except Exception:
        pass
    return False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
