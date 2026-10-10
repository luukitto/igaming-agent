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
from agent import DB, MODEL, TODAY, Report, investigate, top_risk_players

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


@app.get("/stats")
def stats():
    """Numbers for the team dashboard. Plain SQL, no LLM, so it loads instantly."""
    one = lambda sql: dict(DB.execute(sql, {"today": TODAY}).fetchone())
    rows = lambda sql: [dict(r) for r in DB.execute(sql, {"today": TODAY})]
    week = "created_at >= date(:today, '-7 days')"
    return {
        "as_of": TODAY,
        "cases": one("SELECT COUNT(*) total, SUM(status = 'pending') pending, SUM(status = 'approved') approved, "
                     "SUM(status = 'rejected') rejected, MIN(CASE WHEN status = 'pending' THEN proposed_at END) oldest_pending "
                     "FROM actions"),
        "pending_by_action": rows("SELECT action, COUNT(*) n FROM actions WHERE status = 'pending' GROUP BY action ORDER BY n DESC"),
        "players": one("SELECT COUNT(*) total, SUM(kyc_status != 'verified') kyc_unfinished, "
                       "SUM(account_status = 'self_excluded') self_excluded, SUM(account_status = 'blocked') blocked FROM players"),
        "withdrawals": rows(f"SELECT status, COUNT(*) n FROM transactions WHERE type = 'withdrawal' AND {week} GROUP BY status ORDER BY n DESC"),
        "declines": rows(f"SELECT decline_reason reason, COUNT(*) n FROM transactions WHERE status = 'declined' AND {week} "
                         "GROUP BY reason ORDER BY n DESC"),
        "at_risk": top_risk_players(7, 8),
    }


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    return FileResponse(Path(__file__).parent / "dashboard.html")


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/favicon.ico", include_in_schema=False)  # browsers ask for this path on their own
def favicon():
    return FileResponse(Path(__file__).parent / "favicon.ico")


@app.get("/health")
def health():
    return {"ok": True}
