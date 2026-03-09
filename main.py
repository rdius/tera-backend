import os, io, uuid, hashlib
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import fitz  # PyMuPDF
from groq import Groq
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, SearchRequest
)
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY    = os.environ["GROQ_API_KEY"]
GROQ_MODEL      = "llama-3.1-70b-versatile"        # gratuit, <1s
QDRANT_URL      = os.environ["QDRANT_URL"]          # from qdrant.io free cluster
QDRANT_API_KEY  = os.environ["QDRANT_API_KEY"]
COLLECTION      = "tera_mining_docs"
EMBED_MODEL     = "paraphrase-multilingual-mpnet-base-v2"  # 768 dims, FR/EN/multi
CHUNK_SIZE      = 800    # tokens approx
CHUNK_OVERLAP   = 100
TOP_K           = 6

# ── Global singletons ──────────────────────────────────────────────────────────
embedder: Optional[SentenceTransformer] = None
qdrant:   Optional[QdrantClient]        = None
ai_client: Optional[Groq] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, qdrant, ai_client
    print("⏳ Loading embedding model…")
    embedder  = SentenceTransformer(EMBED_MODEL)
    qdrant    = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    ai_client = Groq(api_key=GROQ_API_KEY)

    # Create collection if needed
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"✅ Collection '{COLLECTION}' created")
    else:
        print(f"✅ Collection '{COLLECTION}' ready")
    print("🚀 TERA MINE-INSIGHTS backend ready")
    yield

app = FastAPI(title="TERA MINE-INSIGHTS API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
        i += size - overlap
    return chunks

def extract_pdf_chunks(file_bytes: bytes, filename: str) -> list[dict]:
    """Extract structured chunks from PDF preserving page/section metadata."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    chunks = []
    current_section = "Document principal"

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if not text:
            continue

        # Detect section headers (heuristic: short lines in caps or ending with \n)
        lines = text.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) < 80 and (
                stripped.isupper() or
                stripped.startswith(('1.', '2.', '3.', '4.', '5.',
                                     'CHAPITRE', 'SECTION', 'Chapter', 'RÉSULTATS',
                                     'CONCLUSIONS', 'INTRODUCTION', 'MÉTHODE'))
            ):
                current_section = stripped[:60]
                break

        text_chunks = chunk_text(text)
        for idx, chunk in enumerate(text_chunks):
            chunk_id = hashlib.md5(f"{filename}_{page_num}_{idx}".encode()).hexdigest()
            chunks.append({
                "id":       chunk_id,
                "text":     chunk,
                "filename": filename,
                "page":     page_num,
                "section":  current_section,
                "total_pages": len(doc),
            })

    doc.close()
    return chunks

def embed(texts: list[str]) -> list[list[float]]:
    return embedder.encode(texts, batch_size=32, show_progress_bar=False).tolist()

SYSTEM_PROMPT = """Tu es TERA MINE-INSIGHTS, expert en analyse de documents miniers (géologie, exploration, HSE, contrats, géochimie).

RÈGLES STRICTES :
1. Réponds UNIQUEMENT sur la base des extraits de documents fournis en contexte.
2. Cite TOUJOURS : *Source : [nom_fichier], Page [X], Section : [Y]*
3. Si l'info est absente : "Cette information n'est pas disponible dans les documents fournis."
4. Donne un score : *Fiabilité : [X]% — [Très Haute/Haute/Moyenne/Faible]*
   - Très Haute (>90%) : info explicite et précise dans le document
   - Haute (70-90%) : info présente mais à interpréter
   - Moyenne (50-70%) : info partielle ou indirecte
   - Faible (<50%) : extrapolation nécessaire
5. Précis sur les valeurs numériques (concentrations g/t, profondeurs m, coordonnées GPS).
6. Structure : **Réponse directe** → Détails techniques → Sources → Fiabilité
7. Réponds dans la langue de la question."""

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "TERA MINE-INSIGHTS"}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    project_id: str  = Form(default="default"),
):
    """Upload and index a PDF document into Qdrant."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Seuls les fichiers PDF sont acceptés.")

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    print(f"📄 Indexing: {file.filename} ({size_mb:.1f} Mo) — project: {project_id}")

    # Extract chunks
    chunks = extract_pdf_chunks(file_bytes, file.filename)
    if not chunks:
        raise HTTPException(422, "Impossible d'extraire du texte de ce PDF.")

    # Embed
    texts   = [c["text"] for c in chunks]
    vectors = embed(texts)

    # Upsert into Qdrant
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload={
                "text":        chunks[i]["text"],
                "filename":    chunks[i]["filename"],
                "page":        chunks[i]["page"],
                "section":     chunks[i]["section"],
                "total_pages": chunks[i]["total_pages"],
                "project_id":  project_id,
            }
        )
        for i in range(len(chunks))
    ]

    qdrant.upsert(collection_name=COLLECTION, points=points)

    return {
        "status":   "indexed",
        "filename": file.filename,
        "chunks":   len(chunks),
        "pages":    chunks[-1]["page"] if chunks else 0,
        "size_mb":  round(size_mb, 1),
    }


class ChatRequest(BaseModel):
    question:   str
    project_id: str = "default"
    history:    list[dict] = []

@app.post("/chat")
async def chat(req: ChatRequest):
    """RAG chat endpoint: retrieve relevant chunks, then generate answer."""
    if not req.question.strip():
        raise HTTPException(400, "Question vide.")

    # 1. Embed the question
    q_vector = embed([req.question])[0]

    # 2. Retrieve top-K chunks from Qdrant filtered by project
    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=q_vector,
        query_filter=Filter(
            must=[FieldCondition(key="project_id", match=MatchValue(value=req.project_id))]
        ),
        limit=TOP_K,
        with_payload=True,
    )

    if not results:
        return {
            "answer":  "Aucun document n'a été indexé pour ce projet. Veuillez d'abord uploader vos PDFs.",
            "sources": [],
        }

    # 3. Build context from chunks
    context_parts = []
    sources = []
    seen = set()

    for r in results:
        p = r.payload
        src_key = f"{p['filename']}_p{p['page']}"
        context_parts.append(
            f"[Fichier: {p['filename']} | Page {p['page']}/{p['total_pages']} | Section: {p['section']} | Score: {r.score:.2f}]\n{p['text']}"
        )
        if src_key not in seen:
            sources.append({
                "filename": p["filename"],
                "page":     p["page"],
                "section":  p["section"],
                "score":    round(r.score, 3),
            })
            seen.add(src_key)

    context = "\n\n---\n\n".join(context_parts)

    # 4. Build messages for Claude
    messages = []
    for h in req.history[-6:]:  # last 3 turns
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})

    messages.append({
        "role": "user",
        "content": f"CONTEXTE DOCUMENTAIRE (extraits les plus pertinents) :\n\n{context}\n\n---\n\nQUESTION : {req.question}"
    })

    # 5. Generate answer — Groq LLaMA 3.1 70B
    response = ai_client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=1500,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
    )

    answer = response.choices[0].message.content

    return {
        "answer":  answer,
        "sources": sources,
        "chunks_used": len(results),
    }


@app.delete("/project/{project_id}")
def delete_project(project_id: str):
    """Remove all vectors for a given project."""
    qdrant.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="project_id", match=MatchValue(value=req.project_id))]
        )
    )
    return {"status": "deleted", "project_id": project_id}


@app.get("/project/{project_id}/docs")
def list_docs(project_id: str):
    """List indexed documents for a project."""
    results = qdrant.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="project_id", match=MatchValue(value=project_id))]
        ),
        limit=500,
        with_payload=True,
    )
    seen, docs = set(), []
    for point in results[0]:
        fname = point.payload.get("filename", "")
        if fname not in seen:
            seen.add(fname)
            docs.append({
                "filename":    fname,
                "total_pages": point.payload.get("total_pages", 0),
            })
    return {"project_id": project_id, "documents": docs}
