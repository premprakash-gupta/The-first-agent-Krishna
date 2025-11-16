import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types
import gradio as gr

# Load API key
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY missing. Create .env with GOOGLE_API_KEY=...")

# Build agent & runner
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
        "Respond exactly in Krishna's voice: firm, clear, detached, and rooted in dharma. "
        "No politeness, no flattery, no praise of the user's question, no emotional cushioning, "
        "no modern filler words, and no LLM-style phrases. Krishna gives verdicts, not validation. "
        "Speak with strategic clarity and unshaken authority. Use metaphors only when they sharpen dharma, "
        
        "For factual or current topics, use Google Search. For dilemmas, morality, or inner conflict, "
        "answer strictly as Krishna — steady, direct, and free from hesitation."
        "Answer in detail as per question and use Hindi/English as per question dialect."
    ),

    tools=[google_search],
)

runner = InMemoryRunner(agent=agent)

async def ask_agent(prompt: str):
    prompt = (prompt or "").strip()
    if not prompt:
        return "Prompt cannot be empty"
    response = await runner.run_debug(prompt)
    return _extract_text(response)


def _extract_text(resp) -> str:
    """Extract plain text from ADK/GenAI response objects so Gradio shows only the answer."""
    try:
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
        t = getattr(resp, "text", None)
        if t:
            return str(t).strip()
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
    try:
        import re
        s = str(resp)
        matches = re.findall(r"text=(?:\"\"\"|\')([\s\S]+?)(?:\"\"\"|\')", s)
        if matches:
            return "\n\n".join([m.strip() for m in matches])
        return s
    except Exception:
        return str(resp)

with gr.Blocks(title="ADK Gradio Assistant") as demo:
    gr.Markdown("# ADK Gradio Assistant\nEnter a prompt to query the Gemini-based agent.")
    inp = gr.Textbox(label="Prompt", placeholder="Ask something...", lines=3)
    out = gr.Markdown(label="Response")
    btn = gr.Button("Send")

    btn.click(fn=ask_agent, inputs=inp, outputs=out)

if __name__ == "__main__":
    # Note: queue() enables concurrency; share=True exposes a public link (optional)
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)
