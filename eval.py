"""End-to-end eval: run the full agent on the planted cases from seed.py and
check the structured report (category, escalation), the answer text, and the proposed actions.
Slow on CPU (a few minutes per case), because every case runs the real LLM.

Run:  python eval.py            (MODEL=llama3.1:8b python eval.py to compare models)
"""
import functools
import json
import time
from pathlib import Path

from agent import MODEL, investigate

print = functools.partial(print, flush=True)  # show results live even when output goes to a file

cases = json.loads((Path(__file__).parent / "eval_cases.json").read_text())
rows, start = [], time.time()
for c in cases:
    try:
        r = investigate(c["question"])
    except RuntimeError as e:  # Ollama hiccup: count it as a failure and keep going, don't lose the whole run
        rows.append(dict.fromkeys(("category", "escalate", "answer", "actions"), False))
        print(f"FAIL  {c['question']}\n      error: {e}")
        continue
    proposed = {p["action"] for p in r["proposals"]}
    checks = {"category": r["report"]["category"] == c["category"],
              "escalate": r["report"]["escalate_to_responsible_gambling"] == c["escalate"],
              "answer": any(m in r["answer"].lower() for m in c["mention_any"]),
              # write actions: proposes one of the right ones (if the case expects any), never a wrong one
              "actions": (not c.get("propose_any") or bool(proposed & set(c["propose_any"])))
                         and not proposed & set(c.get("never_propose", []))}
    rows.append(checks)
    tools = ", ".join(t["tool"] for t in r["tool_calls"]) + (f"  proposed: {', '.join(sorted(proposed))}" if proposed else "")
    print(f"{'PASS' if all(checks.values()) else 'FAIL'}  {c['question']}\n      {checks}  tools: {tools}")
    if not all(checks.values()):
        print(f"      report: {r['report']}\n      answer: {r['answer'][:300]}")

print(f"\n{MODEL}: {sum(all(r.values()) for r in rows)}/{len(rows)} cases fully correct, "
      + ", ".join(f"{k} {sum(r[k] for r in rows)}/{len(rows)}" for k in rows[0])
      + f", {(time.time() - start) / len(rows):.0f}s per case")
