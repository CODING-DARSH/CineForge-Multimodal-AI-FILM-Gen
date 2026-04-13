"""
CineForge — RAG knowledge store.
Seeds ChromaDB with cinematic knowledge: shot types, emotion→visual mappings,
style rules. At query time, retrieves relevant context for the LLM scene planner.
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import chromadb
from chromadb.utils import embedding_functions
from loguru import logger
from config import cfg


CINEMATIC_KNOWLEDGE: list[dict] = [
    # ── Shot types ────────────────────────────────────────────────────────────
    {
        "id": "shot_wide",
        "text": "Wide shot: establishes the full environment and character's place within it. Use for scene openings, action sequences, and moments of isolation or grandeur.",
        "category": "shot_type",
    },
    {
        "id": "shot_medium",
        "text": "Medium shot: shows character from waist up. Standard for dialogue scenes and character-focused moments. Balances environment and emotion.",
        "category": "shot_type",
    },
    {
        "id": "shot_closeup",
        "text": "Close-up: fills the frame with a face or object. Use for intense emotional moments, revelations, or when a single detail carries the scene's meaning.",
        "category": "shot_type",
    },
    {
        "id": "shot_overshoulder",
        "text": "Over-shoulder: shows one character from behind, framing another. Classic for conversations. Creates intimacy and perspective.",
        "category": "shot_type",
    },
    {
        "id": "shot_pov",
        "text": "POV shot: camera becomes the character's eyes. Pulls viewer into subjective experience. Use for moments of discovery or fear.",
        "category": "shot_type",
    },
    # ── Emotion → visual mappings ─────────────────────────────────────────────
    {
        "id": "emotion_melancholic",
        "text": "Melancholic scenes: desaturated colour palette, muted blues and greys, soft diffuse light, slow motion feel, shallow depth of field with blurred background, golden hour or overcast sky.",
        "category": "emotion_visual",
    },
    {
        "id": "emotion_urgent",
        "text": "Urgent scenes: high contrast, sharp shadows, dynamic diagonal compositions, motion blur on moving subjects, cool harsh lighting, rain or wind effects for texture.",
        "category": "emotion_visual",
    },
    {
        "id": "emotion_joyful",
        "text": "Joyful scenes: warm golden tones, bright saturated colours, soft bokeh background, natural sunlight, open compositions with breathing room, characters in motion.",
        "category": "emotion_visual",
    },
    {
        "id": "emotion_tense",
        "text": "Tense scenes: low key lighting, strong shadows cutting across faces, tight framing, dark backgrounds, high contrast between light and dark areas, claustrophobic compositions.",
        "category": "emotion_visual",
    },
    {
        "id": "emotion_mysterious",
        "text": "Mysterious scenes: fog or haze, silhouette lighting, cold blue or green tints, partially obscured subjects, reflections, symmetrical compositions, negative space.",
        "category": "emotion_visual",
    },
    {
        "id": "emotion_peaceful",
        "text": "Peaceful scenes: soft diffuse natural light, warm earth tones, wide open spaces, gentle bokeh, still compositions, slow movement, nature elements like water or trees.",
        "category": "emotion_visual",
    },
    # ── Cinematic style rules ─────────────────────────────────────────────────
    {
        "id": "style_photorealistic",
        "text": "Photorealistic cinematic style: 35mm film grain, anamorphic lens flare, natural colour grading, realistic skin tones, shallow depth of field, motivated lighting that matches the scene's light sources.",
        "category": "style",
    },
    {
        "id": "style_noir",
        "text": "Noir style: black and white or heavily desaturated, hard chiaroscuro lighting, venetian blind shadow patterns, rain-wet streets, smoke, high contrast, moody atmosphere.",
        "category": "style",
    },
    {
        "id": "style_anime",
        "text": "Anime style: clean linework, cel-shaded colours, expressive eyes, speed lines for action, distinctive hair colours, Japanese architectural or landscape elements.",
        "category": "style",
    },
    {
        "id": "style_watercolour",
        "text": "Watercolour painting style: visible brushstrokes, soft edges, pigment blooms, white paper showing through, translucent layered washes, loose impressionistic detail.",
        "category": "style",
    },
    # ── Prompt quality boosters ───────────────────────────────────────────────
    {
        "id": "quality_image",
        "text": "For high quality image generation always append: masterpiece, best quality, highly detailed, sharp focus, professional photography, 8k uhd resolution, cinematic lighting.",
        "category": "quality",
    },
    {
        "id": "quality_audio",
        "text": "For high quality audio generation always append: high quality audio, clear recording, professional sound design, no background noise, crisp sound effects.",
        "category": "quality",
    },
]


class RAGStore:
    """ChromaDB-backed knowledge store for cinematic context retrieval."""

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=cfg.CHROMA_DB_PATH)
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._collection = self._client.get_or_create_collection(
            name=cfg.CHROMA_COLLECTION,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._seed_if_empty()

    def _seed_if_empty(self) -> None:
        if self._collection.count() == 0:
            logger.info("Seeding RAG store with cinematic knowledge...")
            self._collection.add(
                ids=[k["id"] for k in CINEMATIC_KNOWLEDGE],
                documents=[k["text"] for k in CINEMATIC_KNOWLEDGE],
                metadatas=[{"category": k["category"]} for k in CINEMATIC_KNOWLEDGE],
            )
            logger.info(f"Seeded {len(CINEMATIC_KNOWLEDGE)} knowledge entries.")

    def add_character(self, character_id: str, description: str) -> None:
        """Store a character description for cross-scene retrieval."""
        self._collection.upsert(
            ids=[f"char_{character_id}"],
            documents=[description],
            metadatas=[{"category": "character"}],
        )
        logger.debug(f"Stored character: {character_id}")

    def query(
        self,
        query_text: str,
        n_results: int = 4,
        category_filter: str | None = None,
    ) -> list[str]:
        """Return top-n relevant knowledge snippets for the given query."""
        where = {"category": category_filter} if category_filter else None
        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self._collection.count()),
            where=where,
        )
        return results["documents"][0] if results["documents"] else []

    def get_context_for_scene(self, emotion: str, characters: list[str], style: str) -> str:
        """
        Build a rich context block for the LLM by retrieving relevant
        emotion visuals, shot type suggestions, character descriptions,
        and style rules.
        """
        snippets: list[str] = []

        # Emotion-specific visual guidance
        emotion_docs = self.query(f"{emotion} scene visual style", n_results=2, category_filter="emotion_visual")
        snippets.extend(emotion_docs)

        # Shot type suggestion based on emotion
        shot_docs = self.query(f"best shot type for {emotion} scene", n_results=1, category_filter="shot_type")
        snippets.extend(shot_docs)

        # Style rules
        style_docs = self.query(f"{style} visual style rules", n_results=1, category_filter="style")
        snippets.extend(style_docs)

        # Character descriptions
        for char in characters[:2]:  # limit to 2 characters to avoid context bloat
            char_docs = self.query(char, n_results=1, category_filter="character")
            snippets.extend(char_docs)

        # Quality booster
        quality_docs = self.query("image quality prompt", n_results=1, category_filter="quality")
        snippets.extend(quality_docs)

        return "\n".join(f"- {s}" for s in snippets if s)