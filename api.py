"""HTTP API around the agent.

Run:   uvicorn api:app --reload
Try:   curl -X POST localhost:8000/investigate -H 'Content-Type: application/json' \
            -d '{"question": "Why was player 1042'"'"'s withdrawal declined?"}'
Docs:  http://localhost:8000/docs
UI:    http://localhost:8000
"""
import os
import secrets
from datetime import date, timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

import actions
from agent import DB, MODEL, TODAY, Report, investigate, rg_scores

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


# Every dashboard query sees only the chosen country's rows (or all, when country is None).
def _scoped(table):
    return f"(SELECT x.* FROM {table} x JOIN players p ON p.id = x.player_id WHERE :country IS NULL OR p.country = :country)"


B, T, A = _scoped("bets"), _scoped("transactions"), _scoped("actions")


@app.get("/stats")
def stats(days: int = Query(30, ge=1, le=366), country: str | None = Query(None, max_length=2)):
    """Numbers for the owner's dashboard over the last `days` days, compared with the `days` before.
    Plain SQL, no LLM, so it loads instantly."""
    end = date.fromisoformat(TODAY) + timedelta(days=1)  # exclusive, so TODAY is included
    win = lambda a, b: {"country": country, "start": str(a), "end": str(b)}
    cur, prev = win(end - timedelta(days), end), win(end - timedelta(2 * days), end - timedelta(days))
    one = lambda sql, p=cur: dict(DB.execute(sql, p).fetchone())
    rows = lambda sql, p=cur: [dict(r) for r in DB.execute(sql, p)]
    in_window = "created_at >= :start AND created_at < :end"

    def kpis(p):
        return one(f"""SELECT
            (SELECT COALESCE(SUM(stake), 0) FROM {B} WHERE {in_window}) turnover,
            (SELECT COALESCE(SUM(stake - payout), 0) FROM {B} WHERE {in_window}) ggr,
            (SELECT COUNT(DISTINCT player_id) FROM {B} WHERE {in_window}) active_players,
            (SELECT COALESCE(SUM(amount), 0) FROM {T} WHERE {in_window} AND type = 'deposit' AND status = 'completed') deposits,
            (SELECT COALESCE(SUM(amount), 0) FROM {T} WHERE {in_window} AND type = 'withdrawal' AND status = 'completed') withdrawals,
            (SELECT COUNT(*) FROM players WHERE registered_at >= :start AND registered_at < :end
               AND (:country IS NULL OR country = :country)) new_players""", p)

    daily = {r["day"]: r for r in rows(f"""SELECT substr(created_at, 1, 10) day, SUM(stake) turnover, SUM(stake - payout) ggr,
        COUNT(DISTINCT player_id) players FROM {B} WHERE {in_window} GROUP BY day""")}
    for r in rows(f"""SELECT substr(created_at, 1, 10) day, SUM(amount) deposits FROM {T}
                      WHERE {in_window} AND type = 'deposit' AND status = 'completed' GROUP BY day"""):
        daily.setdefault(r["day"], {}).update(r)
    days_list = [str(end - timedelta(i)) for i in range(days, 0, -1)]
    zero = {"turnover": 0, "ggr": 0, "players": 0, "deposits": 0}

    risk = rg_scores(7)  # the risk rules are tuned for a one-week window, whatever period is shown
    players = rows(f"""SELECT p.id, p.username, p.country, p.registered_at, p.kyc_status, p.account_status,
            p.weekly_deposit_limit, COALESCE(t.deposits, 0) deposits, COALESCE(t.withdrawals, 0) withdrawals,
            COALESCE(b.bets, 0) bets, COALESCE(b.turnover, 0) turnover, COALESCE(b.ggr, 0) ggr,
            (SELECT MAX(created_at) FROM bets WHERE player_id = p.id) last_bet
        FROM players p
        LEFT JOIN (SELECT player_id, SUM(CASE WHEN type = 'deposit' AND status = 'completed' THEN amount END) deposits,
                          SUM(CASE WHEN type = 'withdrawal' AND status = 'completed' THEN amount END) withdrawals
                   FROM transactions WHERE {in_window} GROUP BY player_id) t ON t.player_id = p.id
        LEFT JOIN (SELECT player_id, COUNT(*) bets, SUM(stake) turnover, SUM(stake - payout) ggr
                   FROM bets WHERE {in_window} GROUP BY player_id) b ON b.player_id = p.id
        WHERE :country IS NULL OR p.country = :country
        ORDER BY p.id""")
    for pl in players:
        r = risk.get(pl["id"], {})
        pl |= {"risk_score": r.get("risk_score", 0), "risk_reasons": r.get("reasons", [])}

    return {
        "as_of": TODAY, "start": cur["start"], "days": days, "country": country,
        "countries": [r["country"] for r in DB.execute("SELECT DISTINCT country FROM players ORDER BY country")],
        "kpis": kpis(cur), "prev": kpis(prev),
        "daily": [{"day": d} | zero | daily.get(d, {}) for d in days_list],
        "games": rows(f"""SELECT g.name, g.provider, g.category, g.rtp, COUNT(*) bets, COUNT(DISTINCT b.player_id) players,
            SUM(b.stake) turnover, SUM(b.stake - b.payout) ggr FROM {B} b JOIN games g ON g.id = b.game_id
            WHERE b.created_at >= :start AND b.created_at < :end GROUP BY g.id ORDER BY ggr DESC"""),
        "declines": rows(f"""SELECT decline_reason reason, COUNT(*) n, SUM(amount) amount FROM {T}
            WHERE {in_window} AND status = 'declined' GROUP BY reason ORDER BY n DESC"""),
        "pending": one(f"SELECT COUNT(*) n, MIN(proposed_at) oldest FROM {A} WHERE status = 'pending'"),
        "players": players,
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
