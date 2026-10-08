"""Approval and audit trail for the agent's write actions.

The agent can only *propose* actions (agent.propose_action changes nothing). The API
saves each proposal here as pending. A person approves or rejects it, and only then
does an approved action touch player data. The `actions` table is the audit log,
and the database enforces it (see seed.py): rows can't be deleted, a decision is
final, and the proposal can't be edited afterwards.

This is the only module that opens the database for writing; the agent never imports it.
"""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

PATH = Path(__file__).parent / "casino.db"

# What an approved action does to player data. request_kyc_documents and flag_for_rg_review
# change nothing here: the approved row is the work item for the KYC / RG team.
EFFECTS = {
    # self-excluded players stay self-excluded: blocking must not hide the exclusion
    "block_account": "UPDATE players SET account_status = 'blocked' WHERE id = :player_id AND account_status = 'active'",
    # MIN: an operator-set limit can only tighten an existing one (policy: increases need a cooling period)
    "apply_deposit_limit": "UPDATE players SET weekly_deposit_limit = MIN(COALESCE(weekly_deposit_limit, :limit), :limit) "
                           "WHERE id = :player_id",
}


@contextmanager
def connect():
    """One short-lived connection per call: FastAPI runs requests in threads, so nothing is shared.
    `with con` commits on success and rolls back on an exception."""
    con = sqlite3.connect(PATH, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        with con:
            yield con
    finally:
        con.close()


def propose(player_id, action, params, reason, question, proposed_by):
    with connect() as con:
        cur = con.execute("INSERT INTO actions (player_id, action, params, reason, question, proposed_by) "
                          "VALUES (?,?,?,?,?,?)", (player_id, action, json.dumps(params), reason, question, proposed_by))
    return get(cur.lastrowid)


def decide(action_id, approve: bool, by: str):
    """Approve or reject a pending action. On approve, the effect runs in the same transaction,
    so the audit row and the data change are saved together or not at all."""
    with connect() as con:
        cur = con.execute("UPDATE actions SET status = ?, decided_by = ?, decided_at = datetime('now') "
                          "WHERE id = ? AND status = 'pending'", ("approved" if approve else "rejected", by, action_id))
        if cur.rowcount == 0:  # missing, or already decided (also covers two people clicking at once)
            raise LookupError(f"no pending action with id {action_id}")
        row = con.execute("SELECT * FROM actions WHERE id = ?", (action_id,)).fetchone()
        if approve and row["action"] in EFFECTS:
            con.execute(EFFECTS[row["action"]], {"player_id": row["player_id"]} | json.loads(row["params"]))
    return get(action_id)


def get(action_id) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM actions WHERE id = ?", (action_id,)).fetchone()
    return dict(row) | {"params": json.loads(row["params"])} if row else None


def log(limit=100) -> list[dict]:
    """Pending first (they need a person), then newest decisions."""
    with connect() as con:
        rows = con.execute("SELECT * FROM actions ORDER BY status != 'pending', id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) | {"params": json.loads(r["params"])} for r in rows]
