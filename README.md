# BankSaathi: an RBI customer-rights assistant on a Nugen domain-aligned model

## Problem
When a bank transaction goes wrong in India (card fraud, an ATM that debits but
gives no cash, a complaint the bank ignores), most customers don't know the
RBI rules that protect them: zero liability if fraud is reported within
3 working days, ₹100/day automatic compensation for late reversals, and the
right to escalate to the RBI Ombudsman after 30 days. BankSaathi answers these
questions in plain language with the exact numbers.

## What I built (all through the Nugen API)
1. **Domain documents**: 5 plain-language guides based on RBI rules, covering:
   - Limiting liability in unauthorised electronic transactions (July 2017)
   - The Integrated Ombudsman Scheme, 2021
   - TAT and compensation for failed transactions (Sept 2019)
   - KYC updation and BSBD accounts (including rules effective 1 April 2026)
   - The Charter of Customer Rights (2014)
2. **Pipeline** (`nugen_pipeline.py`): uploads the documents, generates a
   benchmark, replaces it with a 20-question benchmark I checked by hand, runs
   the alignment on `Llama-3.2-3B-Instruct`, deploys the model and fetches
   Nugen's evaluation. It can be resumed, because every ID is saved to `state.json`.
3. **App** (`app.py`): a Streamlit chat app on the aligned model via the
   Inference API, with streaming answers, BM25 retrieval over the RBI documents (RAG),
   a "Sources used" panel, and a side-by-side comparison with the base model.
4. **Evaluation** (`compare_eval.py`): scores base, aligned and aligned+RAG
   setups on the 20 questions by checking each answer for the required key facts.

Aligned model: `model_01m64mchcw0jz89d`

## Demo

![BankSaathi answering a card-fraud question](screenshot.png)

![BankSaathi answering an Ombudsman question](screenshot2.png)

## Results

**Nugen platform evaluation (aligned vs base):**

| Metric | Base | Aligned | Change |
|---|---|---|---|
| Response relevance | 0.46 | 0.505 | +9.8% |
| Binary correctness | 0.05 | 0.05 | 0% |

**My key-fact evaluation (20 curated questions):**

| Setup | Key-fact score | Fully correct |
|---|---|---|
| Base Llama-3.2-3B | 37% | 4/20 |
| Nugen aligned | 30% | 3/20 |
| Nugen aligned + RAG | **90%** | **17/20** |

## What I learned
- Alignment shifted the model towards the domain (relevance +9.8%), but a 3B
  model aligned on about 25 KB of text doesn't reliably memorise exact figures
  like "₹10,000" or "T + 5 days". On my strict key-fact check, the aligned model
  alone scored slightly below the base model. Twenty questions is a small
  sample, so I read that as "no reliable gain on exact facts" rather than a regression.
- Putting the right RBI document into the prompt raised correctness from 3/20
  to 17/20. Retrieval found the right document for all 20 questions.
- I would combine the two: the aligned model for domain tone and
  understanding, and retrieval for exact, citable facts.

## Known limitation: arithmetic
On "An ATM debited ₹5,000 but gave no cash, what if the bank is late?", both
models found the right rule (T + 5 days, ₹100/day), but each made a mistake.
The aligned model said a reversal on T + 10 means ₹100 compensation, when the
correct amount is ₹500 for 5 days late. The base model invented a ₹500 cap,
and the rule has none. Small models get the rule right more often than the
arithmetic. A fix is to compute the compensation in code and let the model
only explain it.

## Next steps
- Align with more documents and a larger benchmark, then measure the aligned
  model without RAG again.
- Look into the 3 remaining misses. For example, the Ombudsman appeal answer
  left out "Appellate Authority".
- Use an LLM judge instead of keyword matching, and add Hindi questions.

---

## What's in this repo

| File | What it is |
|---|---|
| `docs/` | The 5 documents to upload (RBI rules, written as plain-language guides) |
| `benchmark_curated.json` | 20 hand-checked questions and answers to evaluate the model |
| `nugen_pipeline.py` | Does the whole Nugen workflow through the API |
| `nugen_client.py` | Small helper that sends questions to your model |
| `app.py` | The BankSaathi chat app |
| `rag.py` | Finds the matching RBI document for each question (RAG) |
| `compare_eval.py` | Scores base vs aligned vs aligned+RAG on the 20 questions |

The documents cover:
1. Liability for unauthorised electronic transactions / fraud (RBI circular, 6 July 2017)
2. RBI Integrated Ombudsman Scheme, 2021 (how to complain)
3. Failed transactions: auto-reversal deadlines and ₹100/day compensation (RBI circular, 20 Sept 2019)
4. KYC updation rules and zero-balance BSBD accounts (incl. rules effective 1 April 2026)
5. RBI Charter of Customer Rights (2014)

## How to run it (about 10 minutes of your time, plus training wait)

**1. Get your API key.** Log in to platform.nugen.in, open **API Keys**, click
**Create API Key**, and copy it.

**2. Install Python packages** (in a terminal, inside this folder):
```bash
pip install requests streamlit
```

**3. Set your key:**
```bash
export NUGEN_API_KEY="paste-your-key"        # Mac / Linux
set NUGEN_API_KEY=paste-your-key             # Windows (cmd)
```

**4. Run the pipeline.** This uploads the docs, creates a benchmark, swaps in
the curated benchmark, trains (aligns) the model, deploys it and asks it a
test question:
```bash
python nugen_pipeline.py --edited-benchmark benchmark_curated.json
```
Training can take a while. If the script stops or you close the terminal, just
run the same command again; it saves progress in `state.json` and continues.

**5. Open the app:**
```bash
streamlit run app.py
```
A browser tab opens. Ask a question or click one of the examples. Turn on
**Compare with base model** to show the aligned model's answer next to the
original model's.

## Measure the improvement

```bash
python compare_eval.py
```
Asks all 20 curated questions to the base model, the aligned model, and the
aligned model with RAG, checks each answer for the key facts, and writes
`eval_summary.md` and `eval_comparison.csv`.

## Useful extras

- See where you are: `python nugen_pipeline.py --step status`
- Fetch the evaluation scores: `python nugen_pipeline.py --step eval`
  (saved to `evaluation_results.json`; good to mention in your submission)
- Base model defaults to `meta-llama/Llama-3.2-3B-Instruct` (from the Nugen
  cookbook). Change with `--base-model`.
- If the API complains about a missing `workflow_id`, pass `--workflow-id ...`.

## If something fails

The script prints the exact API error. The most common causes are a wrong or
missing API key, or a field name that differs from the cookbook. Paste the
error back to Claude and it can fix the script.

Endpoints follow the official cookbook:
https://github.com/nugen-in/nugen-cookbook/tree/main/guides/alignment_with_nugen_api

*BankSaathi gives general information only, not legal or financial advice.*
