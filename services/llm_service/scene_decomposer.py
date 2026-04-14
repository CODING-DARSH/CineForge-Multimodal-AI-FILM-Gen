"""
CineForge — LLM Scene Decomposer
Uses Mistral-7B via Ollama to break a story into structured SceneObjects.
RAG context is injected per-scene to enforce character/style/emotion consistency.
Retries up to 3 times with a stricter prompt on malformed JSON.
"""
from __future__ import annotations
import json
import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import ollama
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from config import cfg
from schemas import SceneObject, StoryRequest, StoryResponse
from rag_store import RAGStore


SYSTEM_PROMPT = """You are a professional screenwriter and AI film director.
Your job is to analyze a story and produce a structured scene breakdown in JSON.

RULES:
1. Return ONLY a valid JSON object — no markdown, no explanation, no code blocks.
2. Each scene must have all required fields.
3. visual_prompt must be rich, descriptive, and mention: subject, action, setting, lighting, mood, camera angle, style.
4. audio_prompt must describe SOUNDS only: ambient noise, music texture, sound effects. NO visual descriptions.
5. narration must be 1-3 natural spoken sentences matching the scene's emotion and pacing.
6. emotion must be exactly one of: melancholic, urgent, joyful, tense, mysterious, peaceful
7. shot_type must be exactly one of: wide, medium, close-up, over-shoulder, pov
8. Minimum 3 scenes, maximum 8 scenes.
9. Characters list must be consistent — use the EXACT same name string across scenes.
10. Always append quality boosters to visual_prompt: "masterpiece, best quality, highly detailed, cinematic lighting"
"""

SCENE_SCHEMA_EXAMPLE = """
Return this exact JSON structure:
{
  "protagonist_description": "A detailed physical description of the main character for identity consistency",
  "scenes": [
    {
      "scene_index": 0,
      "description": "One-sentence narrative summary",
      "visual_prompt": "Detailed image generation prompt with style, lighting, camera, mood",
      "audio_prompt": "Ambient sounds and audio texture for this scene",
      "narration": "Spoken narration text for this scene",
      "emotion": "melancholic",
      "characters": ["protagonist_name"],
      "shot_type": "wide",
      "duration_seconds": 6.0,
      "setting": "Location description"
    }
  ]
}
"""


class SceneDecomposer:
    def __init__(self, rag: RAGStore) -> None:
        self._rag = rag
        self._client = ollama.Client(host=cfg.OLLAMA_HOST)

    def _build_user_prompt(
        self,
        request: StoryRequest,
        rag_context: str,
        attempt: int,
    ) -> str:
        strictness = "" if attempt == 0 else (
            "\n⚠ CRITICAL: Your previous response had invalid JSON. Return ONLY the JSON object, nothing else."
        )
        protagonist_hint = (
            f"\nProtagonist physical description: {request.protagonist_description}"
            if request.protagonist_description
            else ""
        )
        return f"""Analyze this story and produce a JSON scene breakdown.

STORY:
{request.story_text}

VISUAL STYLE: {request.style}
{protagonist_hint}

CINEMATIC CONTEXT (retrieved from knowledge base — use this to enhance scene quality):
{rag_context}

{SCENE_SCHEMA_EXAMPLE}
{strictness}"""

    def _parse_response(self, raw: str) -> dict:
        """Extract JSON from model response, stripping any surrounding text."""
        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Try extracting JSON block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"No valid JSON found in response: {raw[:200]}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _call_llm(self, user_prompt: str, attempt: int) -> dict:
        logger.info(f"LLM call attempt {attempt + 1}/3")
        response = self._client.chat(
            model=cfg.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": 0.7, "num_ctx": 4096},
        )
        raw = response["message"]["content"]
        logger.debug(f"Raw LLM response (first 300 chars): {raw[:300]}")
        return self._parse_response(raw)

    def decompose(self, request: StoryRequest) -> StoryResponse:
        """
        Main entry point. Returns a validated StoryResponse with all scenes.
        Injects RAG context for the overall story, then refines per-scene.
        """
        # Global RAG query for story-level context
        global_rag = self._rag.get_context_for_scene(
            emotion="cinematic",
            characters=[],
            style=request.style,
        )

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                user_prompt = self._build_user_prompt(request, global_rag, attempt)
                data = self._call_llm(user_prompt, attempt)

                protagonist_desc = data.get("protagonist_description", "")
                raw_scenes = data.get("scenes", [])

                if not raw_scenes:
                    raise ValueError("LLM returned empty scenes list")

                scenes: list[SceneObject] = []
                for i, s in enumerate(raw_scenes):
                    s["scene_index"] = i  # ensure correct indexing
                    scene = SceneObject(**s)

                    # Inject per-scene RAG context into visual_prompt
                    scene_rag = self._rag.get_context_for_scene(
                        emotion=scene.emotion,
                        characters=scene.characters,
                        style=request.style,
                    )
                    if scene_rag:
                        scene.visual_prompt = (
                            f"{scene.visual_prompt}, "
                            f"{cfg.STYLE_TRIGGER_WORD}, "
                            f"masterpiece, best quality, highly detailed"
                        )
                        logger.debug(f"Scene {i} RAG context injected.")

                    scenes.append(scene)

                # Store protagonist description in RAG for future retrieval
                if protagonist_desc:
                    char_name = (scenes[0].characters[0] if scenes[0].characters else "protagonist")
                    self._rag.add_character(char_name, protagonist_desc)

                total_duration = sum(s.duration_seconds for s in scenes)
                return StoryResponse(
                    scenes=scenes,
                    protagonist_description=protagonist_desc,
                    total_scenes=len(scenes),
                    estimated_duration=total_duration,
                )

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

        raise RuntimeError(f"Scene decomposition failed after 3 attempts. Last error: {last_error}")