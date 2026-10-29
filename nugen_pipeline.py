"""End-to-end Nugen alignment pipeline.

Steps: upload documents -> generate benchmark -> (optionally edit) -> create
alignment -> wait for training -> deploy -> wait for deployment -> smoke-test
inference.

Every ID is saved to state.json after each step, so if anything fails you can
re-run the script and it resumes where it stopped.

Usage:
    export NUGEN_API_KEY="..."
    python nugen_pipeline.py
    python nugen_pipeline.py --step status     # just print current state
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://api.nugen.in"
STATE_FILE = Path("state.json")
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


# ---------- helpers ----------

def headers():
    key = os.environ.get("NUGEN_API_KEY")
    if not key:
        sys.exit("Set NUGEN_API_KEY first (Nugen platform -> API Keys -> Create API Key).")
    return {"Authorization": f"Bearer {key}"}


def load_state():
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def call(method, path, **kwargs):
    resp = requests.request(method, f"{BASE_URL}{path}", headers=headers(), timeout=120, **kwargs)
    if not resp.ok:
        sys.exit(f"{method} {path} failed [{resp.status_code}]: {resp.text}")
    return resp.json()


def wait_for(path, label, done=("READY", "COMPLETED", "SUCCESS", "DEPLOYED", "PROCESSED"),
             failed=("FAILED", "ERROR"), interval=15, timeout=4 * 3600):
    """Poll a /status endpoint until it reaches a terminal state."""
    start = time.time()
    while True:
        data = call("GET", path)
        status = str(data.get("status", "")).upper()
        print(f"  [{label}] status={status}  ({int(time.time() - start)}s)")
        if status in done:
            return data
        if status in failed:
            sys.exit(f"{label} failed: {json.dumps(data, indent=2)}")
        if time.time() - start > timeout:
            sys.exit(f"Timed out waiting for {label}. Re-run later; state is saved.")
        time.sleep(interval)


# ---------- steps ----------

def upload_documents(state, docs_dir):
    if state.get("document_ids"):
        print(f"Documents already uploaded: {state['document_ids']}")
        return
    files = sorted(p for p in Path(docs_dir).iterdir()
                   if p.suffix.lower() in {".txt", ".md", ".pdf", ".json", ".jsonl", ".docx"})
    if not files:
        sys.exit(f"No documents found in {docs_dir}")
    ids = []
    for p in files:
        print(f"Uploading {p.name} ...")
        with open(p, "rb") as f:
            data = call("POST", "/api/v3/documents/create",
                        data={"categories": "text/json", "names": p.stem},
                        files={"files": (p.name, f)})
        ids.extend(data["document_ids"])
    state["document_ids"] = ids
    save_state(state)
    for doc_id in ids:
        wait_for(f"/api/v3/documents/{doc_id}/status", f"document {doc_id}", interval=5)


def generate_benchmark(state, num_questions):
    if state.get("benchmark_id"):
        print(f"Benchmark already exists: {state['benchmark_id']}")
        return
    data = call("POST", "/api/v3/benchmarks/create",
                json={"document_ids": state["document_ids"], "num_questions": num_questions})
    state["benchmark_id"] = data["benchmark_id"]
    save_state(state)
    wait_for(f"/api/v3/benchmarks/{state['benchmark_id']}/status", "benchmark", interval=10)

    # Save the generated Q&A pairs so you can review / edit them.
    bench = call("GET", f"/api/v3/benchmarks/{state['benchmark_id']}/data")
    samples = [{k: v for k, v in s.items() if k != "image_url"} for s in bench["samples"]]
    Path("benchmark_generated.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    print(f"Saved {len(samples)} benchmark samples to benchmark_generated.json")


def upload_edited_benchmark(state, path):
    """Optional: upload a hand-edited benchmark and use it instead of the generated one."""
    with open(path, "rb") as f:
        data = call("POST", "/api/v3/benchmarks/upload",
                    data={"name": "edited benchmark",
                          "document_id": state["document_ids"],
                          "description": "Benchmark reviewed and edited by hand"},
                    files={"file": (Path(path).name, f, "application/json")})
    state["generated_benchmark_id"] = state.get("benchmark_id")
    state["benchmark_id"] = data["benchmark_id"]
    save_state(state)
    wait_for(f"/api/v3/benchmarks/{state['benchmark_id']}/status", "edited benchmark", interval=5)


def create_alignment(state, name, base_model, workflow_id):
    if state.get("alignment_id"):
        print(f"Alignment already created: {state['alignment_id']}")
    else:
        payload = {
            "alignment_name": name,
            "base_model_id": base_model,
            "document_ids": state["document_ids"],
            "benchmark_id": state["benchmark_id"],
            "description": f"Domain alignment for {name}",
        }
        if workflow_id:
            payload["workflow_id"] = workflow_id
        data = call("POST", "/api/v3/alignment-projects/create", json=payload)
        state["alignment_id"] = data["alignment_id"]
        save_state(state)
    wait_for(f"/api/v3/alignment-projects/{state['alignment_id']}/status",
             "alignment", done=("READY",), interval=60)


def find_model(state):
    if state.get("model_id"):
        return
    models = call("GET", "/api/v3/models/aligned")["domain_aligned_models"]
    # Prefer the model produced by our alignment; fall back to the newest one.
    match = [m for m in models if state["alignment_id"] in json.dumps(m)]
    state["model_id"] = (match or models)[0]["model_id"]
    save_state(state)
    print(f"Aligned model: {state['model_id']}")


def deploy(state):
    if not state.get("deployed"):
        call("POST", f"/api/v3/models/{state['model_id']}/deployment")
    wait_for(f"/api/v3/models/{state['model_id']}/deployment/status", "deployment", interval=30)
    state["deployed"] = True
    save_state(state)


def evaluation(state):
    """Fetch the automatic benchmark evaluation of the aligned model, if available."""
    model = call("GET", f"/api/v3/models/{state['model_id']}")
    eval_id = (model.get("evaluation_data") or {}).get("evaluation_id")
    if not eval_id:
        print("No evaluation attached to the model yet.")
        return
    wait_for(f"/api/v3/evaluations/{eval_id}/status", "evaluation", interval=20)
    results = call("GET", f"/api/v3/evaluations/{eval_id}/results")
    Path("evaluation_results.json").write_text(json.dumps(results, indent=2))
    print("Saved evaluation_results.json")


def smoke_test(state, question):
    from nugen_client import chat
    print(f"\nQ: {question}\nA: ", end="", flush=True)
    for token in chat(state["model_id"], [{"role": "user", "content": question}]):
        print(token, end="", flush=True)
    print()


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs", help="folder of domain documents")
    ap.add_argument("--name", default="BankSaathi RBI customer rights")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--workflow-id", default=None, help="only if your account requires one")
    ap.add_argument("--num-questions", type=int, default=25)
    ap.add_argument("--edited-benchmark", help="path to an edited benchmark JSON to upload")
    ap.add_argument("--question", default="I reported a card fraud to my bank 2 working days after the SMS. How much do I have to pay?")
    ap.add_argument("--step", default="all",
                    choices=["all", "upload", "benchmark", "align", "deploy", "eval", "test", "status"])
    args = ap.parse_args()

    state = load_state()
    if args.step == "status":
        print(json.dumps(state, indent=2))
        return

    steps = {
        "upload": lambda: upload_documents(state, args.docs),
        "benchmark": lambda: generate_benchmark(state, args.num_questions),
        "align": lambda: (upload_edited_benchmark(state, args.edited_benchmark)
                          if args.edited_benchmark and not state.get("generated_benchmark_id") else None,
                          create_alignment(state, args.name, args.base_model, args.workflow_id)),
        "deploy": lambda: (find_model(state), deploy(state)),
        "eval": lambda: evaluation(state),
        "test": lambda: smoke_test(state, args.question),
    }
    for name, fn in steps.items():
        if args.step in ("all", name):
            print(f"\n== {name} ==")
            fn()


if __name__ == "__main__":
    main()
