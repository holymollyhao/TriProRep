"""FastAPI server for the DNAChunker chunking-visualization demo.

This server is intended for local dev. In production, the static frontend
lives on GitHub Pages and only the /api/* endpoints below need to be reachable.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from inference import MAX_SEQ_LEN, get_inference

HERE = Path(__file__).resolve().parent          # demo_release/server
SITE_ROOT = HERE.parent                          # demo_release (Pages root)

app = FastAPI(title="DNAChunker Demo", docs_url=None, redoc_url=None)

# CORS allowlist. Override with DNA_CHUNKER_CORS="https://your.origin,...".
_origins = os.environ.get(
    "DNA_CHUNKER_CORS",
    "https://holymollyhao.github.io,http://localhost:8000,http://127.0.0.1:8000",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=SITE_ROOT / "static"), name="static")

# Server-side input allowlist: ACGTN only, up to MAX_SEQ_LEN.
_SEQ_RE = re.compile(rf"^[ACGTNacgtn]{{1,{MAX_SEQ_LEN}}}$")


class ChunkRequest(BaseModel):
    sequence: str = Field(..., description="Raw DNA sequence (A/C/G/T/N).")


@app.on_event("startup")
def _warmup() -> None:
    # Loading the 2GB checkpoint at first request is too slow; do it on boot.
    get_inference()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(SITE_ROOT / "index.html")


@app.get("/api/health")
def health() -> JSONResponse:
    inf = get_inference()
    return JSONResponse(
        {
            "status": "ok",
            "device": str(inf.device),
            "dtype": str(inf.autocast_dtype).replace("torch.", ""),
            "max_seq_len": MAX_SEQ_LEN,
        }
    )


@app.post("/api/chunk")
def chunk(req: ChunkRequest) -> JSONResponse:
    if not _SEQ_RE.match(req.sequence):
        raise HTTPException(status_code=400, detail="invalid sequence")
    inf = get_inference()
    try:
        result = inf.run(req.sequence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(result)
