"""HTTP API around the agent.

Run:   uvicorn api:app --reload
Try:   curl -X POST localhost:8000/investigate -H 'Content-Type: application/json' \
            -d '{"question": "Why was player 1042'"'"'s withdrawal declined?"}'
Docs:  http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import Report, investigate

app = FastAPI(title="iGaming Ops Agent")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Investigation(BaseModel):
    answer: str
    report: Report
    tool_calls: list[dict]


@app.post("/investigate", response_model=Investigation)
def post_investigate(q: Question):  # plain def: FastAPI runs it in a thread, so slow LLM calls don't block
    try:
        return investigate(q.question)
    except RuntimeError as e:  # Ollama unreachable
        raise HTTPException(503, str(e))


@app.get("/health")
def health():
    return {"ok": True}
