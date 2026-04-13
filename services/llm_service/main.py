"""
CineForge — LLM Service  (port 8001)
POST /decompose  →  StoryResponse (list of SceneObjects)
POST /add_character  →  stores a character description in RAG
GET  /health     →  service health check
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from config import cfg
from schemas import StoryRequest, StoryResponse
from rag_store import RAGStore
from scene_decomposer import SceneDecomposer

cfg.ensure_dirs()

app = FastAPI(title="CineForge LLM Service", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialise once on startup — ChromaDB and Mistral client are cheap to hold
_rag: RAGStore | None = None
_decomposer: SceneDecomposer | None = None


@app.on_event("startup")
async def startup() -> None:
    global _rag, _decomposer
    logger.info("LLM service starting up...")
    _rag = RAGStore()
    _decomposer = SceneDecomposer(_rag)
    logger.info("LLM service ready.")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "llm_service"}


@app.post("/decompose", response_model=StoryResponse)
async def decompose(request: StoryRequest) -> StoryResponse:
    """
    Break a story into structured cinematic scenes.
    RAG context is injected automatically from the knowledge store.
    """
    logger.info(f"Decomposing story ({len(request.story_text)} chars) | style={request.style}")
    try:
        result = _decomposer.decompose(request)
        logger.info(f"Produced {result.total_scenes} scenes, est. {result.estimated_duration:.1f}s")
        return result
    except Exception as e:
        logger.error(f"Decomposition failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class CharacterAddRequest(BaseModel):
    character_id: str
    description: str


@app.post("/add_character")
async def add_character(req: CharacterAddRequest) -> dict:
    """Store a character description in the RAG vector store."""
    _rag.add_character(req.character_id, req.description)
    return {"status": "stored", "character_id": req.character_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("LLM_SERVICE_PORT", 8001)), reload=False)