"""HTTP API around the agent.

Run:   uvicorn api:app --reload
Try:   curl -X POST localhost:8000/investigate -H 'Content-Type: application/json' \
            -d '{"question": "Why was player 1042'"'"'s withdrawal declined?"}'
Docs:  http://localhost:8000/docs
UI:    http://localhost:8000
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import actions
from agent import MODEL, Report, investigate

app = FastAPI(title="iGaming Ops Agent")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Investigation(BaseModel):
    answer: str
    report: Report
    tool_calls: list[dict]
    actions: list[dict]  # proposed by the agent, pending until a person decides


class Decision(BaseModel):
    approve: bool
    # ponytail: the client says who it is. Behind real auth (SSO), take this from the session instead,
    # or the audit log records whatever name was typed.
    by: str = Field(min_length=2, max_length=100)


@app.post("/investigate", response_model=Investigation)
def post_investigate(q: Question):  # plain def: FastAPI runs it in a thread, so slow LLM calls don't block
    try:
        r = investigate(q.question)
    except RuntimeError as e:  # Ollama unreachable
        raise HTTPException(503, str(e))
    r["actions"] = [actions.propose(**p, question=q.question, proposed_by=f"agent:{MODEL}")
                    for p in r.pop("proposals")]
    return r


@app.get("/actions")
def get_actions():
    """The audit log: every proposed action, who decided it and when."""
    return actions.log()


@app.post("/actions/{action_id}/decision")
def post_decision(action_id: int, d: Decision):
    try:
        return actions.decide(action_id, d.approve, d.by)
    except LookupError as e:
        raise HTTPException(409, str(e))


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
def health():
    return {"ok": True}
