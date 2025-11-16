# Web UI for ADK Agent

## Structure
```
webui/
  app.py              # FastAPI backend exposing /query and /health
  gradio_app.py       # Gradio interface (prompt -> response)
  static/index.html   # Simple fetch() front-end for FastAPI
  requirements.txt    # Web-specific deps (FastAPI + Gradio)
```

## Prerequisites
1. Python 3.12 virtual environment already exists at `adk_dev_env/`.
2. `.env` file at project root with `GOOGLE_API_KEY=...`.

## Install Dependencies (inside the existing venv)
Activate (Windows CMD):
```
adk_dev_env\Scripts\activate.bat
```
Install:
```
pip install -r webui\requirements.txt
```

## Run FastAPI Backend
```
python webui\app.py
```
Open: `http://localhost:8000/static/index.html`

## Run Gradio Interface
```
python webui\gradio_app.py
```
Open: `http://localhost:7860` (default) for the Gradio Blocks UI.

## Endpoint Contract (FastAPI)
- `POST /query` body: `{ "prompt": "text" }` → `{ "text": "model output" }`
- `GET /health` → `{ "status": "ok" }`

## Modify Agent Behavior
- FastAPI: edit `build_agent()` in `app.py`.
- Gradio: edit agent instantiation in `gradio_app.py`.

## Common Issues
- Missing API key → RuntimeError at startup.
- Blank prompt → validation returns message instead of call.
- 429/5xx errors retried automatically via `HttpRetryOptions`.

## Next Ideas
- Add conversation history (list of turns) in Gradio.
- Streaming token updates using callback hooks.
- Shared agent builder in `sample-agent/agent.py` for DRY configuration.
