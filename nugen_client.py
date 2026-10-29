"""Tiny client for the Nugen Inference API (chat completions, streaming)."""

import json
import os

import requests

INFERENCE_URL = "https://api.nugen.in/api/v3/inference/chat/completions"


def chat(model_id, messages, max_tokens=512, temperature=0.2):
    """Yield response text chunks from the aligned model as they stream in."""
    resp = requests.post(
        INFERENCE_URL,
        headers={
            "accept": "application/json",
            "Authorization": f"Bearer {os.environ['NUGEN_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        },
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8").strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            break
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for choice in event.get("choices", []):
            text = (choice.get("delta") or {}).get("content") \
                or (choice.get("message") or {}).get("content") \
                or choice.get("text")
            if text:
                yield text
