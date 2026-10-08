"""An LLM agent that investigates player issues on a casino platform.

An "agent" is just: an LLM + a loop + tools.
  1. Send the question to the LLM, along with descriptions of the tools it may use.
  2. The LLM either answers, or replies "call get_player(1042)" (a tool call).
  3. We run that Python function and send the result back to the LLM.
  4. Repeat until the LLM answers with no tool calls.
  5. Finally, turn the free-text answer into a typed Report (structured output).
The LLM never touches the database; it can only ask us to run the functions below.

Run:  python agent.py "Why was player 1042's withdrawal declined?"
"""
import json
import math
import os
import re
import sqlite3
import sys
import urllib.request
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

HERE = Path(__file__).parent
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("MODEL", "qwen3:4b")
TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", 900))  # seconds per LLM call; one call takes ~6 min on an old Intel CPU
EMBED_MODEL = "nomic-embed-text"
TODAY = "2026-10-01"  # matches seed.py, so "last week" means the same thing every run
# Read-only connection: even a buggy tool can't modify data. check_same_thread=False
# because FastAPI runs requests in worker threads (reads only, so sharing is safe).
DB = sqlite3.connect(f"file:{HERE / 'casino.db'}?mode=ro", uri=True, check_same_thread=False)
DB.row_factory = sqlite3.Row  # rows behave like dicts: dict(row)


def post(path, payload):
    req = urllib.request.Request(OLLAMA + path, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.load(r)
    except TimeoutError as e:  # a slow read raises this directly, not wrapped in URLError
        raise RuntimeError(f"Ollama took over {TIMEOUT}s for one call (slow on CPU?). Raise OLLAMA_TIMEOUT.") from e
    except OSError as e:  # URLError, connection refused or reset mid-call
        raise RuntimeError(f"Can't reach Ollama at {OLLAMA}. Start it with `ollama serve`.") from e


# --- Tools: plain Python functions the LLM is allowed to call ----------------
# Values always go through ? placeholders, never f-strings: the LLM chooses
# these arguments, so treat them like user input (SQL injection).
def get_player(player_id: int) -> dict:
    row = DB.execute("SELECT * FROM players WHERE id = ?", (player_id,)).fetchone()
    return dict(row) if row else {"error": f"no player with id {player_id}"}


def get_transactions(player_id: int, status: str | None = None) -> list[dict]:
    """The player's deposits/withdrawals/bonuses, newest first, optionally only one status."""
    sql = "SELECT * FROM transactions WHERE player_id = ?"
    params: list = [player_id]
    if status:
        sql += " AND status = ?"
        params.append(status)
    return [dict(r) for r in DB.execute(sql + " ORDER BY created_at DESC", params)]


def get_betting_summary(player_id: int, days: int = 30) -> dict:
    """Totals over the last `days` days. The tool does the arithmetic, because
    small LLMs are bad at adding up 100 numbers; the LLM only interprets."""
    since = f"date('{TODAY}', '-{int(days)} days')"  # int() makes the f-string safe
    row = DB.execute(f"""
        SELECT COUNT(*) AS bets, ROUND(COALESCE(SUM(stake), 0), 2) AS total_staked,
               ROUND(COALESCE(SUM(payout), 0), 2) AS total_payout,
               ROUND(COALESCE(SUM(payout - stake), 0), 2) AS net_result,
               ROUND(AVG(CAST(strftime('%H', created_at) AS INT) < 5), 2) AS share_of_bets_midnight_to_5am
        FROM bets WHERE player_id = ? AND created_at >= {since}""", (player_id,)).fetchone()
    bonus = DB.execute("SELECT amount, created_at FROM transactions WHERE player_id = ? AND type = 'bonus' "
                       "AND status = 'completed' ORDER BY created_at DESC LIMIT 1", (player_id,)).fetchone()
    summary = dict(row) | {"period_days": days}
    if bonus:
        wagered = DB.execute("SELECT ROUND(COALESCE(SUM(stake), 0), 2) FROM bets "
                             "WHERE player_id = ? AND created_at >= ?", (player_id, bonus["created_at"])).fetchone()[0]
        summary |= {"last_bonus_amount": bonus["amount"], "last_bonus_at": bonus["created_at"],
                    "wagered_since_last_bonus": wagered}
    return summary


# --- Responsible gambling: risk signals and a score to rank players by -------
RG_SQL_BETS = """
WITH b AS (
  SELECT player_id, id, created_at, stake,
         julianday(created_at) - julianday(LAG(created_at) OVER w) AS gap_days,
         LAG(payout < stake) OVER w AS prev_lost
  FROM bets WHERE created_at >= date(:today, :window) AND (:pid IS NULL OR player_id = :pid)
  WINDOW w AS (PARTITION BY player_id ORDER BY created_at, id)),
s AS (  -- a session ends after 30 minutes without a bet
  SELECT *, SUM(gap_days IS NULL OR gap_days > 30 / 1440.0)
              OVER (PARTITION BY player_id ORDER BY created_at, id) AS session FROM b),
sessions AS (
  SELECT player_id, (julianday(MAX(created_at)) - julianday(MIN(created_at))) * 1440 AS minutes
  FROM s GROUP BY player_id, session)
SELECT player_id, COUNT(*) AS bets,
       ROUND(AVG(CAST(strftime('%H', created_at) AS INT) < 5), 2) AS night_share,
       ROUND(AVG(CASE WHEN prev_lost THEN stake END) / AVG(CASE WHEN NOT prev_lost THEN stake END), 2)
         AS stake_after_loss_vs_after_win,
       (SELECT ROUND(MAX(minutes)) FROM sessions x WHERE x.player_id = s.player_id) AS longest_session_minutes
FROM s GROUP BY player_id"""

RG_SQL_MONEY = """
SELECT player_id,
  SUM(CASE WHEN type = 'deposit' AND status = 'completed' AND created_at >= date(:today, :window)
           THEN amount ELSE 0 END) AS deposits,
  ROUND(SUM(CASE WHEN type = 'deposit' AND status = 'completed' AND created_at < date(:today, :window)
                 THEN amount ELSE 0 END) * :days / 28.0, 2) AS usual_deposits_previous_4_weeks,
  SUM(decline_reason IS 'deposit_limit_reached' AND created_at >= date(:today, :window)) AS deposit_limit_hits,
  SUM(type = 'withdrawal' AND status = 'cancelled' AND created_at >= date(:today, :window) AND EXISTS (
        SELECT 1 FROM bets b WHERE b.player_id = t.player_id AND b.created_at > t.created_at
          AND julianday(b.created_at) - julianday(t.created_at) <= 1)) AS cancelled_withdrawals_gambled
FROM transactions t
WHERE created_at >= date(:today, :window, '-28 days') AND (:pid IS NULL OR player_id = :pid)
GROUP BY player_id"""

# ponytail: hand-picked thresholds and weights (sum 100); calibrate them on cases the RG team labels
RG_RULES = [
    (lambda s: s["deposits"] >= 200 and s["deposits"] >= 3 * s["usual_deposits_previous_4_weeks"], 25,
     "deposits at least 3x the usual amount"),
    (lambda s: s["bets"] >= 10 and s["night_share"] >= 0.5, 20, "most bets placed between midnight and 5 am"),
    (lambda s: s["bets"] >= 30 and (s["stake_after_loss_vs_after_win"] or 0) >= 2, 20,
     "raises stakes after losses (loss chasing)"),
    (lambda s: s["cancelled_withdrawals_gambled"] > 0, 15, "cancelled a withdrawal and gambled the money"),
    (lambda s: (s["longest_session_minutes"] or 0) >= 180, 10, "sessions of 3+ hours"),
    (lambda s: s["deposit_limit_hits"] > 0, 10, "hit a deposit limit"),
]
RG_EMPTY = {"bets": 0, "night_share": None, "stake_after_loss_vs_after_win": None, "longest_session_minutes": None,
            "deposits": 0, "usual_deposits_previous_4_weeks": 0, "deposit_limit_hits": 0,
            "cancelled_withdrawals_gambled": 0}


def rg_scores(days=7, player_id=None) -> dict[int, dict]:
    """Signals for every player with activity in the last `days` days, plus a 0-100 risk score."""
    params = {"today": TODAY, "window": f"-{int(days)} days", "days": int(days), "pid": player_id}
    players = {}
    for row in [*DB.execute(RG_SQL_BETS, params), *DB.execute(RG_SQL_MONEY, params)]:
        players.setdefault(row["player_id"], dict(RG_EMPTY)).update(dict(row))
    for s in players.values():
        hits = [(points, why) for rule, points, why in RG_RULES if rule(s)]
        s |= {"risk_score": sum(p for p, _ in hits), "reasons": [why for _, why in hits]}
    return players


def get_rg_signals(player_id: int, days: int = 7) -> dict:
    if "error" in (player := get_player(player_id)):
        return player
    s = rg_scores(days, player_id).get(player_id, RG_EMPTY | {"player_id": player_id, "risk_score": 0, "reasons": []})
    return s | {"account_status": player["account_status"], "period_days": days}


def top_risk_players(days: int = 7, limit: int = 10) -> list[dict]:
    ranked = sorted(rg_scores(days).values(), key=lambda s: -s["risk_score"])
    return [{k: s[k] for k in ("player_id", "risk_score", "reasons")} for s in ranked[:int(limit)] if s["risk_score"]]


# --- Write actions: the agent only proposes, a person approves (see actions.py) -
ACTIONS = {
    "request_kyc_documents": "ask the player to upload a government ID and proof of address",
    "flag_for_rg_review": "send the account to the responsible gambling team for a safer-gambling interaction",
    "apply_deposit_limit": "set a weekly deposit limit (needs weekly_deposit_limit); can only lower an existing limit",
    "block_account": "block the account, for fraud or abuse (not for self-excluded players, they stay excluded)",
}


def propose_action(player_id: int, action: str, reason: str, weekly_deposit_limit: float | None = None) -> dict:
    """Changes nothing. The proposal goes back to the API, which saves it as pending
    until a person approves or rejects it."""
    if action not in ACTIONS:
        return {"error": f"unknown action {action!r}, available: {list(ACTIONS)}"}
    if "error" in (player := get_player(player_id)):
        return player
    params = {}
    if action == "apply_deposit_limit":
        if not isinstance(weekly_deposit_limit, (int, float)) or weekly_deposit_limit <= 0:
            return {"error": "apply_deposit_limit needs a positive weekly_deposit_limit"}
        params = {"limit": float(weekly_deposit_limit)}
    return {"proposal": {"player_id": player_id, "action": action, "params": params, "reason": reason},
            "status": "waiting for human approval. Tell the user it is proposed, not done."}


# --- RAG over policies.md: one chunk per "## " section ------------------------
def embed(texts):
    return post("/api/embed", {"model": EMBED_MODEL, "input": texts})["embeddings"]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))


@cache  # embed the policies once per process, on first use
def policy_index():
    sections = ["## " + s.strip() for s in (HERE / "policies.md").read_text().split("\n## ")[1:]]
    # ponytail: 6 sections, brute-force cosine in Python; use a vector DB (see pdf_chatbot) past ~1000 chunks
    return list(zip(sections, embed(["search_document: " + s for s in sections])))


def search_policy(query: str, k: int = 2) -> list[str]:
    q = embed(["search_query: " + query])[0]
    return [s for s, _ in sorted(policy_index(), key=lambda p: -cosine(q, p[1]))[:k]]


TOOLS = {"get_player": get_player, "get_transactions": get_transactions,
         "get_betting_summary": get_betting_summary, "get_rg_signals": get_rg_signals,
         "top_risk_players": top_risk_players, "search_policy": search_policy, "propose_action": propose_action}

# The LLM never sees the Python code above, only these descriptions (JSON Schema).
# The description is how it decides WHEN to call a tool, so write it for the model.
def spec(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


PLAYER_ID = {"player_id": {"type": "integer"}}
TOOL_SPECS = [
    spec("get_player", "Look up a player's profile: country, registration date, "
         "KYC (identity verification) status and account status (active, self_excluded, blocked).",
         PLAYER_ID, ["player_id"]),
    spec("get_transactions", "List a player's deposits, withdrawals and bonuses, newest first, "
         "including decline reasons for declined ones.",
         PLAYER_ID | {"status": {"type": "string", "enum": ["completed", "pending", "declined", "cancelled"],
                                 "description": "Only return transactions with this status."}},
         ["player_id"]),
    spec("get_betting_summary", "Betting totals for a player over the last N days: number of bets, "
         "total staked, total payout, net result, share of bets placed between midnight and 5 am, "
         "and how much was wagered since the player's last bonus.",
         PLAYER_ID | {"days": {"type": "integer", "description": "Look-back window, default 30."}},
         ["player_id"]),
    spec("get_rg_signals", "Responsible gambling risk signals for one player over the last N days: deposits "
         "vs the usual amount, share of night play, loss chasing (stake after a loss vs after a win), longest "
         "session, cancelled withdrawals that were gambled, deposit limit hits, and a 0-100 risk score with reasons.",
         PLAYER_ID | {"days": {"type": "integer", "description": "Look-back window, default 7."}}, ["player_id"]),
    spec("top_risk_players", "Rank all players by responsible gambling risk score over the last N days. "
         "Use it for questions like 'which players are most at risk this week'.",
         {"days": {"type": "integer", "description": "Look-back window, default 7."},
          "limit": {"type": "integer", "description": "How many players, default 10."}}, []),
    spec("search_policy", "Search the platform's policies (KYC, bonus wagering, responsible gambling, "
         "deposit limits, self-exclusion, withdrawals). Use it to explain a decline reason or decide what to do.",
         {"query": {"type": "string"}}, ["query"]),
    spec("propose_action", "Propose an action on a player's account. It is NOT carried out: a person must "
         "approve it first. Actions: " + "; ".join(f"{k}: {v}" for k, v in ACTIONS.items()) + ".",
         PLAYER_ID | {"action": {"type": "string", "enum": list(ACTIONS)},
                      "reason": {"type": "string", "description": "Why, citing the data and the policy."},
                      "weekly_deposit_limit": {"type": "number", "description": "Only for apply_deposit_limit."}},
         ["player_id", "action", "reason"]),
]

SYSTEM = (f"You are an operations assistant for an online casino platform. Today is {TODAY}. "
          "Use the tools to look up facts and never guess numbers or reasons. "
          "Always check the relevant policy with search_policy before recommending an action. "
          "You cannot change any data. When the policy calls for an action, propose it with propose_action; "
          "a person approves it, so say it is proposed, not done. "
          "Answer briefly, and mention the data you based the answer on.")


def call_tool(name, args):
    """Run a tool the model asked for. Errors go back to the model as data, so it
    can correct itself (wrong name, bad argument) instead of crashing the agent."""
    if name not in TOOLS:
        return {"error": f"unknown tool {name!r}, available: {list(TOOLS)}"}
    try:
        return TOOLS[name](**args)
    except Exception as e:  # e.g. TypeError from a missing/extra argument
        return {"error": f"{type(e).__name__}: {e}"}


def chat(messages, **extra):
    payload = {"model": MODEL, "messages": messages, "stream": False,
               "think": False, "options": {"temperature": 0}} | extra  # think=False: skip qwen3's slow reasoning
    return post("/api/chat", payload)["message"]


def run_agent(question, max_steps=8, verbose=True):
    """Returns (final answer text, list of tool calls made)."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    trace = []
    for _ in range(max_steps):
        msg = chat(messages, tools=TOOL_SPECS)
        messages.append(msg)  # the model needs to see its own tool calls in the history
        if not msg.get("tool_calls"):
            # Guardrail: small models often stop at the decline code. Make them read
            # the policy once, so the answer says what to actually do about it.
            if not any(t["tool"] == "search_policy" for t in trace):
                trace.append({"tool": "(nudge)", "args": {}})
                messages.append({"role": "user", "content": "Before answering, look up the relevant "
                                 "policy with search_policy and use it in your answer."})
                continue
            return re.sub(r"<think>.*?</think>", "", msg["content"], flags=re.S).strip(), trace
        for call in msg["tool_calls"]:
            name, args = call["function"]["name"], call["function"]["arguments"]
            result = call_tool(name, args)
            trace.append({"tool": name, "args": args})
            if name == "propose_action" and "proposal" in result:
                trace[-1]["proposal"] = result["proposal"]
            if verbose:
                print(f"  -> {name}({args})", file=sys.stderr)
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result, default=str)})
    return "Stopped: too many steps without an answer.", trace


# --- Structured output: free text -> typed, validated Report -----------------
class Report(BaseModel):
    player_id: int | None
    category: Literal["kyc", "bonus_wagering", "responsible_gambling", "self_exclusion", "other"] = Field(
        description="kyc: identity not verified (kyc_not_verified). bonus_wagering: bonus not wagered "
                    "enough (wagering_requirement_not_met). responsible_gambling: risky play, escalating "
                    "deposits, deposit limits hit. self_exclusion: self-excluded player activity.")
    root_cause: str
    evidence: list[str]
    recommended_action: str
    escalate_to_responsible_gambling: bool


def to_report(question, answer) -> Report:
    """Ollama constrains generation to the JSON schema, so the output always parses;
    pydantic then validates it. Done as a separate call because small models
    handle tools and a forced JSON format badly at the same time."""
    msg = chat([{"role": "user", "content": f"Question: {question}\n\nInvestigation result:\n{answer}\n\n"
                 "Fill in the report from the investigation result. Do not add facts that are not in it."}],
               format=Report.model_json_schema())
    return Report.model_validate_json(msg["content"])


def investigate(question) -> dict:
    answer, trace = run_agent(question)
    proposals = {(p["player_id"], p["action"]): p for t in trace if (p := t.pop("proposal", None))}  # dedupe repeats
    return {"answer": answer, "report": to_report(question, answer).model_dump(), "tool_calls": trace,
            "proposals": list(proposals.values())}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Why was player 1042's withdrawal declined?"
    print(json.dumps(investigate(q), indent=2))
