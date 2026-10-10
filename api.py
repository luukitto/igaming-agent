"""HTTP API around the agent.

Run:   uvicorn api:app --reload
Try:   curl -X POST localhost:8000/investigate -H 'Content-Type: application/json' \
            -d '{"question": "Why was player 1042'"'"'s withdrawal declined?"}'
Docs:  http://localhost:8000/docs
UI:    http://localhost:8000
"""
import os
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

import actions
from agent import MODEL, Report, investigate

app = FastAPI(title="iGaming Ops Agent")

# Approvers log in with HTTP Basic (the browser shows its own login prompt).
# APPROVERS="j.smith:secret1,a.lee:secret2". Unset means nobody can approve.
# ponytail: shared passwords in an env var; behind SSO (OIDC), read the user from its token instead.
APPROVERS = {name: pw for name, _, pw in (u.strip().partition(":") for u in os.environ.get("APPROVERS", "").split(","))
             if name and pw}
basic = HTTPBasic(realm="approvers")


def approver(c: HTTPBasicCredentials = Depends(basic)) -> str:
    """The logged-in approver's name. It goes in the audit log, so it must come from the login, not the request body."""
    # compare_digest even for unknown users, so response time doesn't reveal which names exist
    ok = secrets.compare_digest(c.password.encode(), APPROVERS.get(c.username, "").encode())
    if not (ok and c.username in APPROVERS):
        raise HTTPException(401, "wrong username or password", headers={"WWW-Authenticate": 'Basic realm="approvers"'})
    return c.username


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Investigation(BaseModel):
    answer: str
    report: Report
    tool_calls: list[dict]
    actions: list[dict]  # proposed by the agent, pending until a person decides


class Decision(BaseModel):
    approve: bool


@app.post("/investigate", response_model=Investigation)
def post_investigate(q: Question):  # plain def: FastAPI runs it in a thread, so slow LLM calls don't block
    try:
        r = investigate(q.question)
    except RuntimeError as e:  # LLM unreachable or API error (bad key, rate limit)
        raise HTTPException(503, str(e))
    r["actions"] = [actions.propose(**p, question=q.question, proposed_by=f"agent:{MODEL}")
                    for p in r.pop("proposals")]
    return r


@app.get("/actions")
def get_actions():
    """The audit log: every proposed action, who decided it and when."""
    return actions.log()


@app.post("/actions/{action_id}/decision")
def post_decision(action_id: int, d: Decision, by: str = Depends(approver)):
    try:
        return actions.decide(action_id, d.approve, by)
    except LookupError as e:
        raise HTTPException(409, str(e))


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/favicon.ico", include_in_schema=False)  # browsers ask for this path on their own
def favicon():
    return FileResponse(Path(__file__).parent / "favicon.png")


@app.get("/health")
def health():
    return {"ok": True}
