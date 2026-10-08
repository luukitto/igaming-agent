"""Fast checks for the tools and the agent's error handling. No LLM needed.
Run:  python test_tools.py
"""
import shutil
import sqlite3
import tempfile
from pathlib import Path

import actions
from agent import (call_tool, get_betting_summary, get_player, get_rg_signals, get_transactions, propose_action,
                   top_risk_players)

assert get_player(1042)["kyc_status"] == "pending"
assert "error" in get_player(9999)

declined = get_transactions(1077, status="declined")
assert [t["decline_reason"] for t in declined] == ["wagering_requirement_not_met"]
dates = [t["created_at"] for t in get_transactions(1100)]
assert dates == sorted(dates, reverse=True)  # newest first

s = get_betting_summary(1077)
assert s["last_bonus_amount"] == 100 and s["wagered_since_last_bonus"] < 35 * 100  # requirement not met
assert get_betting_summary(1100, days=14)["share_of_bets_midnight_to_5am"] == 1.0
assert "last_bonus_amount" not in get_betting_summary(1042)

# injection attempt is just a string that matches nothing, not SQL
assert get_transactions(1042, status="declined' OR '1'='1") == []

# bad calls from the model come back as errors, not crashes
assert "unknown tool" in call_tool("drop_tables", {})["error"]
assert "TypeError" in call_tool("get_player", {"id": 1042})["error"]

# --- responsible gambling score: the two planted risky players top the ranking, for different reasons
assert [p["player_id"] for p in top_risk_players()] == [1100, 1180]
assert "raises stakes after losses (loss chasing)" in get_rg_signals(1180)["reasons"]
assert get_rg_signals(1180)["cancelled_withdrawals_gambled"] == 2
assert get_rg_signals(1042)["risk_score"] == 0 and "error" in get_rg_signals(9999)

# --- write actions: the agent only proposes; the tool itself writes nothing
assert "error" in propose_action(1100, "delete_player", "x")
assert "error" in propose_action(1100, "apply_deposit_limit", "x")  # limit missing
assert "error" in propose_action(9999, "block_account", "x")
# proposals that would do nothing when approved are refused, so "approved" always means a change
assert "error" in propose_action(1150, "block_account", "x")  # self-excluded
assert "error" in propose_action(1100, "apply_deposit_limit", "x", weekly_deposit_limit=5000)  # current is 3500
p = propose_action(1100, "apply_deposit_limit", "RG", weekly_deposit_limit=500)["proposal"]

# approvals run against a throwaway copy of the DB, so tests never add to the real audit log
actions.PATH = Path(tempfile.mkdtemp()) / "casino.db"
shutil.copy(Path(__file__).parent / "casino.db", actions.PATH)
a = actions.propose(**p, question="q", proposed_by="agent:test")
assert a["status"] == "pending" and a["params"] == {"limit": 500}
assert actions.propose(**p, question="asked again", proposed_by="agent:test")["id"] == a["id"]  # no duplicate
assert actions.decide(a["id"], True, "alice")["decided_by"] == "alice"
with actions.connect() as con:
    assert con.execute("SELECT weekly_deposit_limit FROM players WHERE id = 1100").fetchone()[0] == 500
higher = actions.propose(**p | {"params": {"limit": 900}}, question="q", proposed_by="agent:test")
actions.decide(higher["id"], True, "alice")  # a limit can only go down
blocked = actions.propose(1150, "block_account", {}, "x", "q", "agent:test")
actions.decide(blocked["id"], True, "alice")  # self-excluded stays self-excluded
rejected = actions.propose(1180, "block_account", {}, "x", "q", "agent:test")
actions.decide(rejected["id"], False, "bob")  # rejected: no effect
with actions.connect() as con:
    assert con.execute("SELECT weekly_deposit_limit FROM players WHERE id = 1100").fetchone()[0] == 500
    assert con.execute("SELECT account_status FROM players WHERE id = 1150").fetchone()[0] == "self_excluded"
    assert con.execute("SELECT account_status FROM players WHERE id = 1180").fetchone()[0] == "active"

# the audit log can't be rewritten: a decision is final, rows can't be edited or deleted
actions.propose(1180, "flag_for_rg_review", {}, "x", "q", "agent:test")  # one still pending
for bad in (lambda: actions.decide(a["id"], False, "mallory"), lambda: actions.decide(12345, True, "x")):
    try:
        bad(); raise AssertionError("decided twice")
    except LookupError:
        pass
for sql in ("DELETE FROM actions", "UPDATE actions SET decided_by = 'mallory'",
            "UPDATE actions SET reason = 'edited', decided_by = 'x' WHERE status = 'pending'"):
    try:
        with actions.connect() as con:
            con.execute(sql)
        raise AssertionError(f"audit log allowed: {sql}")
    except sqlite3.IntegrityError:
        pass
# approvers must log in: the audit log gets the login name, never a name from the request
import os
os.environ["APPROVERS"] = " alice:s3cret, broken, :nouser, nopw:"
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials
import api
assert api.APPROVERS == {"alice": "s3cret"}
for user, pw in (("alice", "wrong"), ("mallory", "s3cret"), ("nopw", ""), ("", "")):
    try:
        api.approver(HTTPBasicCredentials(username=user, password=pw)); raise AssertionError(f"logged in: {user}")
    except HTTPException as e:
        assert e.status_code == 401 and "Basic" in e.headers["WWW-Authenticate"]
by = api.approver(HTTPBasicCredentials(username="alice", password="s3cret"))
pending = actions.propose(1180, "flag_for_rg_review", {}, "x", "q", "agent:test")
assert api.post_decision(pending["id"], api.Decision(approve=True, by="mallory"), by)["decided_by"] == "alice"
print("ok")
