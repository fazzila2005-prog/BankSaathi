# BankSaathi: a bank customer-rights assistant on a Nugen aligned model

**The idea:** Most people in India don't know their rights when a bank
transaction goes wrong. BankSaathi is a chat assistant, aligned with the Nugen
API on Reserve Bank of India rules, that answers questions like *"An ATM
debited me but gave no cash, what now?"* with the exact deadlines, rupee
limits and next steps.

## What's in this folder

| File | What it is |
|---|---|
| `docs/` | The 5 documents to upload (RBI rules, written as plain-language guides) |
| `benchmark_curated.json` | 20 hand-checked questions and answers to evaluate the model |
| `nugen_pipeline.py` | Does the whole Nugen workflow through the API |
| `nugen_client.py` | Small helper that sends questions to your model |
| `app.py` | The BankSaathi chat app |

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
