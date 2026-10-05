"""End-to-end eval: run the full agent on the planted cases from seed.py and
check the structured report (category, escalation) and the answer text.
Slow on CPU (a few minutes per case), because every case runs the real LLM.

Run:  python eval.py            (MODEL=llama3.1:8b python eval.py to compare models)
"""
import json
import time
from pathlib import Path

from agent import MODEL, investigate

cases = json.loads((Path(__file__).parent / "eval_cases.json").read_text())
rows, start = [], time.time()
for c in cases:
    r = investigate(c["question"])
    checks = {"category": r["report"]["category"] == c["category"],
              "escalate": r["report"]["escalate_to_responsible_gambling"] == c["escalate"],
              "answer": any(m in r["answer"].lower() for m in c["mention_any"])}
    rows.append(checks)
    tools = ", ".join(t["tool"] for t in r["tool_calls"])
    print(f"{'PASS' if all(checks.values()) else 'FAIL'}  {c['question']}\n      {checks}  tools: {tools}")
    if not all(checks.values()):
        print(f"      report: {r['report']}\n      answer: {r['answer'][:300]}")

print(f"\n{MODEL}: {sum(all(r.values()) for r in rows)}/{len(rows)} cases fully correct, "
      + ", ".join(f"{k} {sum(r[k] for r in rows)}/{len(rows)}" for k in rows[0])
      + f", {(time.time() - start) / len(rows):.0f}s per case")
