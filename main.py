"""
main.py – FastAPI backend for the RAG system.

Endpoints
---------
POST /api/upload   – Upload a PDF, chunk & embed (or return cached).
POST /api/chat     – Ask a question against an uploaded PDF.
POST /api/reset    – Acknowledge a session reset.
"""

import logging
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import rag_engine
from rag_engine import RAGEngineError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan – singleton model & index loading (Requirement #1)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy resources exactly once at server startup."""
    logger.info("Starting up – loading embedding model & Pinecone index …")
    embedder, pc, index = rag_engine.init_resources()
    app.state.embedder = embedder
    app.state.pc = pc
    app.state.index = index
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG API",
    description="Upload a PDF, ask questions, get AI-powered answers.",
    version="0.1.0",
    lifespan=lifespan,
)

# Requirement #6: CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static files & Frontend UI
# ---------------------------------------------------------------------------

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend():
        return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    pdf_hash: str
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    sources: list[int]


class UploadResponse(BaseModel):
    status: str
    pdf_hash: str
    filename: str
    cached: bool


class ResetResponse(BaseModel):
    status: str
    message: str


class ErrorResponse(BaseModel):
    """Standardised error payload."""
    error: bool = True
    fatal_session_end: bool
    error_code: str
    message: str


# ---------------------------------------------------------------------------
# Global exception handler – fatal RAG errors
# ---------------------------------------------------------------------------

# Maps RAGEngineError.error_code → HTTP status returned to the client.
_ERROR_CODE_TO_HTTP = {
    "BILLING_OR_QUOTA": 402,
    "AUTH_FAILED": 403,
    "ENGINE_OFFLINE": 503,
}


@app.exception_handler(RAGEngineError)
def handle_rag_engine_error(_request: Request, exc: RAGEngineError):
    """Catch any :class:`RAGEngineError` raised in an endpoint and return
    the structured JSON the frontend expects to end the session."""
    http_status = _ERROR_CODE_TO_HTTP.get(exc.error_code, 500)
    logger.error(
        "Fatal RAG error [%s]: %s (original=%s)",
        exc.error_code, exc, exc.original,
    )
    return JSONResponse(
        status_code=http_status,
        content=ErrorResponse(
            fatal_session_end=True,
            error_code=exc.error_code,
            message=str(exc),
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Helper – non-fatal validation error response
# ---------------------------------------------------------------------------

def _validation_error(message: str) -> JSONResponse:
    """Return a 400 response that keeps the chat session alive."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            fatal_session_end=False,
            error_code="VALIDATION_ERROR",
            message=message,
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# Endpoints are plain `def` so FastAPI dispatches them to a threadpool,
# keeping the async event loop free while CPU-bound embedder.encode()
# and ollama.generate() run.


@app.post("/api/upload", response_model=UploadResponse)
def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, compute its hash, and ingest into Pinecone if new.

    * Validation failures return ``fatal_session_end: false``.
    * Billing / auth / engine errors propagate as :class:`RAGEngineError`
      and are caught by the global exception handler.
    * The temp file is ALWAYS cleaned up in a ``finally`` block.
    """
    # --- validate --------------------------------------------------------- #
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return _validation_error("Only .pdf files are accepted.")

    content = file.file.read()  # sync read (we're in a threadpool)
    if len(content) == 0:
        return _validation_error("Uploaded file is empty.")

    tmp_path: str | None = None
    try:
        # --- save to a temporary file ------------------------------------- #
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        pdf_hash = rag_engine.get_pdf_hash(tmp_path)
        index = app.state.index
        embedder = app.state.embedder

        # --- check cache -------------------------------------------------- #
        # RAGEngineError propagates to the global handler if fatal
        if rag_engine.check_embeddings_exist(pdf_hash, index=index):
            logger.info("PDF %s already embedded – returning cached.", pdf_hash)
            return UploadResponse(
                status="ready",
                pdf_hash=pdf_hash,
                filename=file.filename,
                cached=True,
            )

        # --- ingest ------------------------------------------------------- #
        logger.info("Ingesting new PDF %s (%s) …", file.filename, pdf_hash)
        chunks = rag_engine.create_chunks(tmp_path)
        rag_engine.create_and_store_embeddings(
            chunks,
            pdf_hash,
            embedder=embedder,
            index=index,
        )

        return UploadResponse(
            status="ready",
            pdf_hash=pdf_hash,
            filename=file.filename,
            cached=False,
        )

    finally:
        # Always clean up the temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.debug("Cleaned up temp file %s", tmp_path)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Ask a question against a previously uploaded PDF.

    * Empty-question validation returns ``fatal_session_end: false``.
    * Fatal engine errors propagate as :class:`RAGEngineError` and are
      caught by the global exception handler.
    """
    if not req.question.strip():
        return _validation_error("Question cannot be empty.")

    # RAGEngineError (billing, auth, offline) propagates to the global handler
    result = rag_engine.query_system(
        req.question,
        req.pdf_hash,
        embedder=app.state.embedder,
        index=app.state.index,
        top_k=req.top_k,
    )

    return ChatResponse(answer=result["answer"], sources=result["pages"])


@app.post("/api/reset", response_model=ResetResponse)
def reset():
    """Acknowledge a session/chat reset from the frontend."""
    return ResetResponse(status="ok", message="Session reset acknowledged.")
