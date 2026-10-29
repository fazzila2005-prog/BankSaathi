"""Lightweight retrieval over the docs/ folder (BM25, no extra dependencies).

The aligned model knows the domain's style and vocabulary, but small models
still forget exact numbers. Retrieving the matching RBI rule and putting it in
the prompt lets the aligned model answer with the exact deadlines and limits,
and lets the app show which document the answer came from.
"""

import math
import re
from collections import Counter
from pathlib import Path

STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it
its me my no not of on or so than that the their them then there these they this to
was what when where which who will with within you your""".split())


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9₹]+", text.lower()) if w not in STOPWORDS]


def load_chunks(docs_dir="docs"):
    """Split each document into sections at markdown headings."""
    chunks = []
    for path in sorted(Path(docs_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()
        for section in re.split(r"\n(?=#{2,3} )", text):
            heading = section.splitlines()[0].lstrip("# ").strip()
            chunks.append({"doc": title, "file": path.name, "heading": heading,
                           "text": section.strip(), "tokens": tokenize(title + " " + section)})
    return chunks


class Retriever:
    def __init__(self, docs_dir="docs", k1=1.5, b=0.75):
        self.docs_dir = docs_dir
        self.chunks = load_chunks(docs_dir)
        self.k1, self.b = k1, b
        self.avg_len = sum(len(c["tokens"]) for c in self.chunks) / max(len(self.chunks), 1)
        df = Counter(t for c in self.chunks for t in set(c["tokens"]))
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.tf = [Counter(c["tokens"]) for c in self.chunks]

    def search(self, query, top_k=3):
        q = tokenize(query)
        scored = []
        for chunk, tf in zip(self.chunks, self.tf):
            length = len(chunk["tokens"])
            score = sum(
                self.idf.get(t, 0) * tf[t] * (self.k1 + 1)
                / (tf[t] + self.k1 * (1 - self.b + self.b * length / self.avg_len))
                for t in q if t in tf
            )
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for s, c in scored[:top_k] if s > 0]


    def search_docs(self, query, top_docs=2):
        """Rank whole documents by their best-matching section.

        Each document is only ~1,300 tokens, so passing the full document keeps
        related rules together (e.g. zero liability AND the 4-7 day caps).
        """
        best = {}
        for chunk in self.search(query, top_k=len(self.chunks)):
            best.setdefault(chunk["file"], chunk)
        files = list(best)[:top_docs]
        return [{"doc": best[f]["doc"], "file": f, "heading": best[f]["heading"],
                 "text": (Path(self.docs_dir) / f).read_text(encoding="utf-8")} for f in files]


def build_context(docs):
    return "\n\n---\n\n".join(d["text"] for d in docs)


RAG_INSTRUCTIONS = (
    "Answer the user's question using the RBI rules below. Quote the exact numbers "
    "(working days, rupee limits, deadlines) from them, and say which rule you used. "
    "If the rules below don't cover the question, say so.\n\nRBI RULES:\n"
)
