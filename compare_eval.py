"""Measure BankSaathi on the 20 curated questions in three setups.

  1. base       - the original Llama-3.2-3B-Instruct
  2. aligned    - your Nugen aligned model
  3. aligned+RAG - aligned model with the matching RBI document in the prompt

Each answer is checked for the key facts it must contain (e.g. "zero" and
"3 working days"). Results go to eval_comparison.csv and eval_summary.md, which
you can include in your submission.

Usage:
    python compare_eval.py
    python compare_eval.py --modes aligned "aligned+RAG"     # skip base
"""

import argparse
import csv
import json
from pathlib import Path

from nugen_client import chat
from rag import RAG_INSTRUCTIONS, Retriever, build_context

SYSTEM = ("You are BankSaathi, a helpful assistant that explains Reserve Bank of India rules "
          "on bank customers' rights in simple language. Give the exact numbers.")

# For each question: a list of fact groups. A group is matched if ANY of its
# phrases appears in the answer (case-insensitive). All groups = fully correct.
KEY_FACTS = {
    1: [["zero", "nothing", "no liability", "not have to pay", "don't have to pay", "₹0", "rs 0"]],
    2: [["10,000", "10000"]],
    3: [["5,000", "5000"]],
    4: [["entire", "full", "whole", "all of the loss", "you bear"], ["report"]],
    5: [["10 working days", "ten working days"]],
    6: [["bank"], ["burden", "prove", "proof"]],
    7: [["t + 5", "t+5", "5 days", "five days"], ["100"]],
    8: [["t + 1", "t+1", "1 day", "one day"]],
    9: [["no", "automatic"]],
    10: [["no", "first", "bank"], ["30 days"]],
    11: [["cms.rbi.org.in"]],
    12: [["one year and 30 days", "1 year and 30 days", "a year and 30 days"]],
    13: [["20 lakh"], ["1 lakh"]],
    14: [["appellate"], ["30 days"]],
    15: [["no", "not"], ["commercial"]],
    16: [["10 years", "ten years"]],
    17: [["no", "not"], ["self-declaration", "self declaration", "declaration"]],
    18: [["no minimum", "zero balance", "no min"], ["four", "4"]],
    19: [["no", "only one"]],
    20: [["fair treatment"], ["transparency"], ["suitability"], ["privacy"], ["grievance"]],
}


def score(answer, groups):
    a = answer.lower()
    hits = sum(any(p in a for p in g) for g in groups)
    return hits / len(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["base", "aligned", "aligned+RAG"])
    ap.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    args = ap.parse_args()

    state = json.loads(Path("state.json").read_text(encoding="utf-8"))
    bench = json.loads(Path("benchmark_curated.json").read_text(encoding="utf-8"))
    retriever = Retriever("docs")

    rows, totals = [], {m: [] for m in args.modes}
    for item in bench:
        n, q = item["sample_num"], item["instruction"]
        row = {"#": n, "question": q, "expected": item["response"]}
        for mode in args.modes:
            system = SYSTEM
            if mode.endswith("RAG"):
                system += "\n\n" + RAG_INSTRUCTIONS + build_context(retriever.search_docs(q))
            model = args.base_model if mode == "base" else state["model_id"]
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
            try:
                answer = "".join(chat(model, msgs, max_tokens=300, temperature=0.1))
            except Exception as e:
                answer = f"ERROR: {e}"
            s = score(answer, KEY_FACTS[n])
            totals[mode].append(s)
            row[f"{mode} answer"], row[f"{mode} score"] = answer.strip(), round(s, 2)
        print(f"Q{n:>2}  " + "  ".join(f"{m}={row[f'{m} score']:.2f}" for m in args.modes))
        rows.append(row)

    with open("eval_comparison.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = ["# BankSaathi evaluation (20 curated questions)", "",
             "| Setup | Key-fact score | Fully correct |", "|---|---|---|"]
    for m in args.modes:
        sc = totals[m]
        lines.append(f"| {m} | {100 * sum(sc) / len(sc):.0f}% | {sum(s == 1 for s in sc)}/{len(sc)} |")
    Path("eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print("\nSaved eval_comparison.csv and eval_summary.md")


if __name__ == "__main__":
    main()
