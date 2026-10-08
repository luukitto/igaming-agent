"""Create a fake casino platform database (casino.db) for the agent to investigate.

Random players and activity, plus planted cases the agent must be able to explain:
  1042  withdrawal declined: KYC (identity check) not verified
  1077  withdrawal declined: bonus wagering requirement not met
  1100  risky gambling pattern: escalating deposits, late-night play, chasing losses
  1150  self-excluded player whose deposit attempt was declined
  1180  loss chasing: doubles the stake after every loss, 4-hour sessions, cancels withdrawals to keep playing

Run:  python seed.py   (re-running rebuilds the same data: the random seed is fixed)
"""
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB = Path(__file__).parent / "casino.db"
TODAY = datetime(2026, 10, 1, 23, 59)  # fixed "now", so questions like "last week" have stable answers

SCHEMA = """
CREATE TABLE players (
  id INTEGER PRIMARY KEY, username TEXT NOT NULL, country TEXT, registered_at TEXT,
  kyc_status TEXT CHECK (kyc_status IN ('verified', 'pending', 'rejected')),
  account_status TEXT CHECK (account_status IN ('active', 'self_excluded', 'blocked')),
  weekly_deposit_limit REAL);
CREATE TABLE games (
  id INTEGER PRIMARY KEY, name TEXT, provider TEXT, category TEXT, rtp REAL);
CREATE TABLE transactions (
  id INTEGER PRIMARY KEY, player_id INTEGER REFERENCES players(id),
  type TEXT CHECK (type IN ('deposit', 'withdrawal', 'bonus')), amount REAL,
  status TEXT CHECK (status IN ('completed', 'pending', 'declined', 'cancelled')),
  decline_reason TEXT, created_at TEXT);
CREATE TABLE bets (
  id INTEGER PRIMARY KEY, player_id INTEGER REFERENCES players(id),
  game_id INTEGER REFERENCES games(id), stake REAL, payout REAL, created_at TEXT);

-- Actions the agent proposed and a person approved or rejected. This is the audit
-- trail, so the database itself enforces it: no deletes, and a decision is final.
CREATE TABLE actions (
  id INTEGER PRIMARY KEY, player_id INTEGER NOT NULL REFERENCES players(id),
  action TEXT NOT NULL CHECK (action IN
    ('request_kyc_documents', 'flag_for_rg_review', 'apply_deposit_limit', 'block_account')),
  params TEXT NOT NULL DEFAULT '{}', reason TEXT NOT NULL, question TEXT,
  proposed_by TEXT NOT NULL, proposed_at TEXT NOT NULL DEFAULT (datetime('now')),
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
  decided_by TEXT, decided_at TEXT);
CREATE TRIGGER actions_no_delete BEFORE DELETE ON actions
  BEGIN SELECT RAISE(ABORT, 'the audit log is append-only'); END;
CREATE TRIGGER actions_decide_once BEFORE UPDATE ON actions
  WHEN OLD.status != 'pending' OR NEW.decided_by IS NULL
    OR (NEW.id, NEW.player_id, NEW.action, NEW.params, NEW.reason, NEW.question, NEW.proposed_by, NEW.proposed_at)
       IS NOT (OLD.id, OLD.player_id, OLD.action, OLD.params, OLD.reason, OLD.question, OLD.proposed_by, OLD.proposed_at)
  BEGIN SELECT RAISE(ABORT, 'only a pending action can be decided, once, and the proposal cannot change'); END;
"""

GAMES = [  # (name, provider, category, rtp)  all fictional
    ("Golden Pharaoh", "Nova Studios", "slots", 0.96), ("Fruit Frenzy", "Nova Studios", "slots", 0.95),
    ("Dragon's Vault", "Redline Games", "slots", 0.97), ("Neon Nights", "Redline Games", "slots", 0.94),
    ("Viking Fortune", "Northwind", "slots", 0.965), ("Lucky Clover", "Northwind", "slots", 0.955),
    ("Live Roulette", "LiveCore", "live", 0.973), ("Live Blackjack", "LiveCore", "live", 0.995),
    ("Speed Baccarat", "LiveCore", "live", 0.988), ("Crash Rocket", "Orbit Labs", "crash", 0.97),
    ("Plinko Drop", "Orbit Labs", "crash", 0.97), ("Mega Wheel", "LiveCore", "live", 0.96),
]
COUNTRIES = ["GE", "DE", "BR", "CA", "FI", "MT", "PE", "NZ"]
PLANTED = {1042, 1077, 1100, 1150, 1180}


def at(days_ago, hour=None, minute=None):
    hour = random.randint(9, 23) if hour is None else hour
    minute = random.randint(0, 59) if minute is None else minute
    t = (TODAY - timedelta(days=days_ago)).replace(hour=hour, minute=minute)
    return t.isoformat(sep=" ", timespec="minutes")


def bet(db, player, day, hour=None, stake=None, win_chance=0.45, payout=None, minute=None):
    game_id, (_, _, _, rtp) = random.choice(list(enumerate(GAMES, start=1)))
    stake = stake or round(random.uniform(0.5, 20), 2)
    # ponytail: crude win model (win ~2.1x stake, else 0), roughly matches rtp on average
    if payout is None:
        payout = round(stake * rtp / win_chance, 2) if random.random() < win_chance else 0
    db.execute("INSERT INTO bets (player_id, game_id, stake, payout, created_at) VALUES (?,?,?,?,?)",
               (player, game_id, stake, payout, at(day, hour, minute)))


def tx(db, player, type_, amount, day, status="completed", reason=None, hour=None):
    db.execute("INSERT INTO transactions (player_id, type, amount, status, decline_reason, created_at) "
               "VALUES (?,?,?,?,?,?)", (player, type_, amount, status, reason, at(day, hour)))


def add_player(db, pid, kyc="verified", status="active", registered_days_ago=None):
    db.execute("INSERT INTO players (id, username, country, registered_at, kyc_status, account_status) "
               "VALUES (?,?,?,?,?,?)",
               (pid, f"player{pid}", random.choice(COUNTRIES),
                at(registered_days_ago or random.randint(60, 700)), kyc, status))


def seed():
    random.seed(42)
    DB.unlink(missing_ok=True)
    db = sqlite3.connect(DB)
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO games (name, provider, category, rtp) VALUES (?,?,?,?)", GAMES)

    # --- ordinary players: a few deposits, bets, sometimes a withdrawal -----
    for pid in range(1000, 1200):
        if pid in PLANTED:
            continue
        add_player(db, pid, kyc=random.choices(["verified", "pending"], [9, 1])[0])
        for _ in range(random.randint(0, 6)):
            day = random.randint(1, 60)
            tx(db, pid, "deposit", random.choice([10, 20, 25, 50, 100]), day)
            for _ in range(random.randint(3, 15)):
                bet(db, pid, max(day - random.randint(0, 2), 0))
        if random.random() < 0.3:
            tx(db, pid, "withdrawal", random.choice([30, 50, 80, 150]), random.randint(1, 30))

    # --- 1042: big win, but identity not verified -> withdrawal declined -----
    add_player(db, 1042, kyc="pending", registered_days_ago=20)
    for day in (15, 9, 4):
        tx(db, 1042, "deposit", 100, day)
    bet(db, 1042, 3, stake=50, payout=650)  # big win
    tx(db, 1042, "withdrawal", 500, 2, status="declined", reason="kyc_not_verified")

    # --- 1077: took a bonus, didn't wager it enough -> withdrawal declined ---
    add_player(db, 1077)
    tx(db, 1077, "deposit", 50, 6)
    tx(db, 1077, "bonus", 100, 6)  # 100% welcome bonus
    for _ in range(45):  # ~900 wagered, far below 35 x 100 = 3500
        bet(db, 1077, random.randint(1, 5), stake=20, win_chance=0.5)
    tx(db, 1077, "withdrawal", 300, 1, status="declined", reason="wagering_requirement_not_met")

    # --- 1100: escalating deposits, mostly 1-4 am, chasing losses ------------
    add_player(db, 1100, registered_days_ago=200)
    for day in (60, 45, 30):  # used to be a calm player
        tx(db, 1100, "deposit", 20, day)
    for day, amount in [(13, 50), (11, 100), (9, 100), (8, 200), (6, 300), (5, 300), (3, 500), (2, 500), (1, 800), (1, 1000)]:
        hour = random.randint(1, 4)
        tx(db, 1100, "deposit", amount, day, hour=hour)
        for _ in range(8):
            stake = round(amount / 6, 2)
            bet(db, 1100, day, hour=hour, stake=stake, payout=stake * 2 if random.random() < 0.15 else 0)
    db.execute("UPDATE players SET weekly_deposit_limit = 3500 WHERE id = 1100")  # the 3400 + 1000 below breaks it
    tx(db, 1100, "deposit", 1000, 0, status="declined", reason="deposit_limit_reached", hour=3)

    # --- 1150: self-excluded, tried to deposit anyway -------------------------
    add_player(db, 1150, status="self_excluded", registered_days_ago=400)
    tx(db, 1150, "deposit", 50, 90)
    tx(db, 1150, "deposit", 100, 3, status="declined", reason="self_exclusion_active")

    # --- 1180: martingale loss chasing, long sessions, cancelled withdrawals --
    add_player(db, 1180, registered_days_ago=300)
    for day in range(4, 60, 7):  # steady 200 a week, so no deposit spike: a different risk profile
        tx(db, 1180, "deposit", 200, day)
    for day in (5, 3, 1):
        if day != 1:  # asks for a withdrawal, cancels it, then plays the money
            tx(db, 1180, "withdrawal", 200, day, status="cancelled", hour=12)
        stake = 5
        for i in range(48):  # a 4-hour session, one bet every 5 minutes
            won = random.random() < 0.45
            bet(db, 1180, day, hour=13 + i // 12, minute=i % 12 * 5, stake=stake, payout=stake * 2 if won else 0)
            stake = 5 if won else min(stake * 2, 160)  # double the stake after every loss

    db.commit()
    return db


if __name__ == "__main__":
    db = seed()
    for table in ("players", "games", "transactions", "bets", "actions"):
        print(f"{table:12} {db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]:>6} rows")
