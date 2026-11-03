import os
import re
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
PROPOSALS_DIR = ROOT / "data" / "proposals"
CHROMA_DIR = ROOT / "chroma_db"
EMBED_MODEL = "gemini-embedding-001"
COLLECTION = "past_proposals"

_genai = None


def _client():
    global _genai
    if _genai is None:
        _genai = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _genai


def embed(texts: list[str], task_type: str) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), 50):
        batch = texts[i : i + 50]
        res = _client().models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        out.extend(e.values for e in res.embeddings)
    return out


# ---------- Chunking ----------
HEADING_RE = re.compile(r"^[A-Z][A-Z &/,\-]{2,}$")
CLIENT_RE = re.compile(r"^(?:Prepared for|Client):\s*(.+)$", re.MULTILINE)


def is_heading(line: str) -> bool:
    s = line.strip()
    return bool(HEADING_RE.match(s)) and s != "END"


def normalize_section(title: str) -> str:
    t = title.lower()
    if t == "cover":
        return "cover"
    if "timeline" in t and ("cost" in t or "budget" in t):
        return "timeline_budget"
    if "executive" in t or "overview" in t:
        return "summary"
    if "solution" in t or "what we will build" in t or "technical" in t:
        return "solution"
    if "timeline" in t:
        return "timeline"
    if "budget" in t or "cost" in t:
        return "budget"
    if "next" in t:
        return "next_steps"
    if "integration" in t:
        return "integrations"
    if "deliverable" in t:
        return "deliverables"
    return "other"


def chunk_proposal(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    m = CLIENT_RE.search(raw)
    client = m.group(1).strip() if m else "unknown"
    industry = re.sub(r"\d+$", "", path.stem.replace("proposal_", ""))

    sections, title, buf = [], "COVER", []
    for line in raw.splitlines():
        if line.strip() == "---":
            continue
        if is_heading(line):
            sections.append((title, "\n".join(buf).strip()))
            title, buf = line.strip(), []
        else:
            buf.append(line)
    sections.append((title, "\n".join(buf).strip()))

    chunks = []
    for idx, (sec_title, body) in enumerate(sections):
        if not body:
            continue
        sec_type = normalize_section(sec_title)
        chunks.append(
            {
                "id": f"{path.stem}::{idx}",
                # context prefix helps the embedding capture industry + section
                "text": f"[{industry} | {client} | {sec_title}]\n{body}",
                "metadata": {
                    "industry": industry,
                    "client": client,
                    "section_title": sec_title,
                    "section_type": sec_type,
                    "source_file": path.name,
                },
            }
        )
    return chunks


# ---------- Chroma ----------
def get_collection():
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return db.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def ingest(reset: bool = False) -> int:
    col = get_collection()
    if reset:
        chromadb.PersistentClient(path=str(CHROMA_DIR)).delete_collection(COLLECTION)
        col = get_collection()
    chunks = []
    for p in sorted(PROPOSALS_DIR.glob("proposal_*.txt")):
        chunks.extend(chunk_proposal(p))
    vectors = embed([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    col.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=vectors,
        metadatas=[c["metadata"] for c in chunks],
    )
    return len(chunks)


def ingest_if_empty() -> None:
    if get_collection().count() == 0:
        ingest()


def search(query: str, k: int = 5, industry: str | None = None,
           section_type: str | list[str] | None = None) -> list[dict]:
    conds = []
    if industry:
        conds.append({"industry": industry})
    if section_type:
        if isinstance(section_type, str):
            section_type = [section_type]
        conds.append({"section_type": {"$in": section_type}})
    where = None if not conds else (conds[0] if len(conds) == 1 else {"$and": conds})

    qvec = embed([query], "RETRIEVAL_QUERY")[0]
    res = get_collection().query(query_embeddings=[qvec], n_results=k, where=where)
    return [
        {"text": d, "distance": dist, **meta}
        for d, dist, meta in zip(res["documents"][0], res["distances"][0], res["metadatas"][0])
    ]

def _show(title, hits):
    print(f"\n=== {title} ===")
    for h in hits:
        print(f"{h['distance']:.3f} | {h['industry']:<13} | {h['section_type']:<16} | {h['client']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("Chunks ingested:", ingest(reset=True))

    q = "multi-warehouse inventory visibility, replace manual weekly reporting, do not rip out existing systems"
    _show("1. No filter", search(q, k=6))
    _show("2. section_type=budget", search(q, k=4, section_type="budget"))
    _show("3. industry=retail", search(q, k=4, industry="retail"))
    _show("4. industry=healthcare (sanity check)", search(q, k=3, industry="healthcare"))