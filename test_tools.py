"""Fast checks for the tools and the agent's error handling. No LLM needed.
Run:  python test_tools.py
"""
from agent import call_tool, get_betting_summary, get_player, get_transactions

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
print("ok")
