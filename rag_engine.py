"""
rag_engine.py – Retrieval-Augmented Generation engine.

Handles PDF ingestion, embedding, Pinecone storage, and querying
with Ollama-backed answer generation.

Core functions accept ``embedder`` and ``index`` as explicit parameters
so that callers (FastAPI lifespan, CLI, tests) control resource lifetime.
Use :func:`init_resources` to create these objects.
"""

import hashlib
import logging
import os

import ollama
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatdatabase")
EMBED_MODEL = "thenlper/gte-large"
EMBED_DIM = 1024  # gte-large output dimension
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
UPSERT_BATCH_SIZE = 100
ENCODE_BATCH_SIZE = 32

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set. Please check your .env file.")


# ---------------------------------------------------------------------------
# Custom exceptions – fatal conditions that should end a chat session
# ---------------------------------------------------------------------------

class RAGEngineError(Exception):
    """Base for fatal RAG errors that terminate a chat session."""

    error_code: str = "UNKNOWN"

    def __init__(self, message: str, *, original: Exception | None = None):
        super().__init__(message)
        self.original = original


class BillingOrQuotaError(RAGEngineError):
    """HTTP 402 / 429, quota exhaustion, billing issues, rate limits."""

    error_code = "BILLING_OR_QUOTA"


class AuthFailedError(RAGEngineError):
    """HTTP 401 / 403 – invalid or expired API key."""

    error_code = "AUTH_FAILED"


class EngineOfflineError(RAGEngineError):
    """LLM or vector-DB engine is unreachable or returned a server error."""

    error_code = "ENGINE_OFFLINE"


# Keywords scanned in error messages (all compared lower-cased)
_BILLING_KEYWORDS = (
    "quota", "billing", "credit", "cost", "rate limit",
    "rate_limit", "insufficient_quota",
)


def _classify_error(exc: Exception) -> None:
    """Inspect *exc* and raise a typed :class:`RAGEngineError` if it
    represents a fatal session-ending condition.

    If the error is **not** recognised as fatal it is re-raised as-is so
    callers can still handle it.
    """
    msg = str(exc).lower()
    # Duck-type: Pinecone and httpx exceptions expose .status / .status_code
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)

    # --- billing / quota / rate-limit ------------------------------------- #
    if status in (402, 429):
        raise BillingOrQuotaError(
            f"Service quota or billing limit reached (HTTP {status}).",
            original=exc,
        ) from exc

    if any(kw in msg for kw in _BILLING_KEYWORDS):
        raise BillingOrQuotaError(
            f"Billing / quota issue detected: {exc}",
            original=exc,
        ) from exc

    # --- auth ------------------------------------------------------------- #
    if status in (401, 403):
        raise AuthFailedError(
            f"Authentication failed (HTTP {status}). Check your API key.",
            original=exc,
        ) from exc

    # --- engine offline --------------------------------------------------- #
    if isinstance(exc, (ConnectionError, OSError)):
        raise EngineOfflineError(
            f"Cannot reach service: {exc}",
            original=exc,
        ) from exc

    # ollama.ResponseError (model not found, server 500, etc.)
    if type(exc).__name__ == "ResponseError":
        raise EngineOfflineError(
            f"LLM engine error: {exc}",
            original=exc,
        ) from exc

    # Not a recognised fatal error – re-raise unchanged
    raise exc


# ---------------------------------------------------------------------------
# Resource initialisation
# ---------------------------------------------------------------------------

def init_resources():
    """Create and return the heavy shared objects.

    Returns
    -------
    tuple[SentenceTransformer, Pinecone, pinecone.Index]
        ``(embedder, pc, index)``
    """
    logger.info("Initialising resources (model=%s, index=%s) …", EMBED_MODEL, INDEX_NAME)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    idx = pc.Index(INDEX_NAME)
    emb = SentenceTransformer(EMBED_MODEL)
    logger.info("Resources ready.")
    return emb, pc, idx


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_pdf_hash(file_path: str) -> str:
    """Return a short MD5 hex digest (12 chars) identifying a PDF file."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def check_embeddings_exist(pdf_hash: str, *, index) -> bool:
    """Check whether Pinecone already contains vectors for *pdf_hash*.

    Raises :class:`RAGEngineError` sub-classes for fatal conditions
    (billing, auth, connectivity).  Non-fatal Pinecone errors are logged
    and the function returns ``False``.

    Parameters
    ----------
    pdf_hash : str
        Hash identifying the PDF.
    index : pinecone.Index
        Pinecone index to query.
    """
    try:
        dummy_vector = [0.0] * EMBED_DIM
        results = index.query(
            vector=dummy_vector,
            filter={"pdf_hash": pdf_hash},
            top_k=1,
        )
        return len(results.matches) > 0
    except Exception as exc:
        # Let fatal errors propagate as typed RAGEngineErrors
        try:
            _classify_error(exc)
        except RAGEngineError:
            raise  # fatal – bubble up
        except Exception:
            pass   # non-fatal – fall through to return False

        logger.exception(
            "Failed to check embeddings for pdf_hash=%s – "
            "possible transient Pinecone error",
            pdf_hash,
        )
        return False


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

def create_chunks(file_path: str):
    """Load a PDF with PyMuPDF and split it into text chunks."""
    logger.info("Loading PDF: %s", file_path)
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    logger.info("Created %d chunks from %d pages", len(chunks), len(docs))
    return chunks


def create_and_store_embeddings(chunks, pdf_hash: str, *, embedder, index) -> None:
    """Embed *chunks* in batches and upsert into Pinecone.

    Parameters
    ----------
    chunks : list[Document]
        LangChain document chunks.
    pdf_hash : str
        Hash identifying the source PDF.
    embedder : SentenceTransformer
        Embedding model instance.
    index : pinecone.Index
        Pinecone index to upsert into.
    """
    logger.info("Creating embeddings (batch_size=%d) …", ENCODE_BATCH_SIZE)

    texts = [chunk.page_content for chunk in chunks]

    # --- batch encode all texts at once ----------------------------------- #
    all_embeddings = embedder.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=True,
    )

    # --- build Pinecone vector dicts -------------------------------------- #
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
        vectors.append(
            {
                "id": f"{pdf_hash}_{i}",
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk.page_content,
                    "pdf_hash": pdf_hash,
                    "chunk_id": i,
                    "page": chunk.metadata.get("page", 0),
                },
            }
        )

    # --- upsert in batches ------------------------------------------------ #
    logger.info("Uploading %d vectors to Pinecone …", len(vectors))
    total_batches = (len(vectors) - 1) // UPSERT_BATCH_SIZE + 1
    for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[i : i + UPSERT_BATCH_SIZE]
        try:
            index.upsert(vectors=batch)
        except Exception as exc:
            _classify_error(exc)  # raises typed RAGEngineError or re-raises
        logger.info(
            "Uploaded batch %d/%d", i // UPSERT_BATCH_SIZE + 1, total_batches
        )

    logger.info("Stored %d embeddings", len(vectors))


# ---------------------------------------------------------------------------
# Query pipeline
# ---------------------------------------------------------------------------

def query_system(
    question: str,
    pdf_hash: str,
    *,
    embedder,
    index,
    top_k: int = 5,
    model: str | None = None,
) -> dict:
    """Query Pinecone for relevant chunks and generate an answer with Ollama.

    Parameters
    ----------
    question : str
        The user's question.
    pdf_hash : str
        Hash of the PDF to restrict the search to.
    embedder : SentenceTransformer
        Embedding model instance.
    index : pinecone.Index
        Pinecone index to query.
    top_k : int, optional
        Number of chunks to retrieve (default 5).
    model : str, optional
        LLM model to use (default: LLM_MODEL / os.getenv('LLM_MODEL', 'llama3.2')).

    Returns
    -------
    dict
        ``{"answer": str, "pages": list[int]}``
    """
    logger.info("Question: %s", question)

    # --- embed the question ----------------------------------------------- #
    question_embedding = embedder.encode(question)

    # --- search Pinecone (filtered by pdf_hash) --------------------------- #
    try:
        results = index.query(
            vector=question_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter={"pdf_hash": pdf_hash},
        )
    except Exception as exc:
        _classify_error(exc)  # raises typed RAGEngineError or re-raises

    if not results.matches:
        return {"answer": "No relevant information found.", "pages": []}

    # --- gather context & source pages ------------------------------------ #
    context_parts: list[str] = []
    pages_seen: set[int] = set()

    for match in results.matches:
        context_parts.append(match.metadata["text"])
        pages_seen.add(match.metadata.get("page", 0))

    context = "\n\n".join(context_parts)
    source_pages = sorted(pages_seen)

    # --- build LLM prompt ------------------------------------------------- #
    prompt = (
        "Based on the following context, answer the question. "
        "If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    # --- generate answer via Ollama --------------------------------------- #
    active_model = model or LLM_MODEL
    logger.info("Generating answer with model=%s …", active_model)
    try:
        response = ollama.generate(model=active_model, prompt=prompt)
        answer = response["response"]
    except Exception as exc:
        _classify_error(exc)  # raises typed RAGEngineError or re-raises
        # If _classify_error re-raised a non-fatal exception, surface it
        raise

    logger.info("Answer: %s", answer)
    return {"answer": answer, "pages": source_pages}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Ingest a PDF (if needed) and start an interactive Q&A loop."""
    embedder, _pc, index = init_resources()

    pdf_file = os.getenv("PDF_FILE", "os.pdf")
    pdf_hash = get_pdf_hash(pdf_file)
    logger.info("PDF Hash: %s", pdf_hash)

    if check_embeddings_exist(pdf_hash, index=index):
        logger.info("Embeddings already exist – skipping creation")
    else:
        logger.info("Creating new embeddings …")
        chunks = create_chunks(pdf_file)
        create_and_store_embeddings(chunks, pdf_hash, embedder=embedder, index=index)

    print("\nSystem ready! Ask your questions (type 'quit' to exit)")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if question:
            result = query_system(question, pdf_hash, embedder=embedder, index=index)
            print(f"\nAnswer: {result['answer']}")
            print(f"Source pages: {result['pages']}")


if __name__ == "__main__":
    main()
