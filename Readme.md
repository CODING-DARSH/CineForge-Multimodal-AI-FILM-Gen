# CineForge AI

**Write a story. Get an AI film.**

CineForge is an end-to-end multimodal AI film generation pipeline. Input a short story, receive a complete short film with animated scenes, voice narration, and ambient soundscapes. Upload your photo to become the protagonist. Upload a voice sample to narrate in your own voice.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://docker.com)


---

## Results

| Metric | Value |
|--------|-------|
| SDXL UNet INT8 speedup | **2.2×** faster inference vs FP32 |
| CogVideoX FP8 speedup | **~2×** faster inference vs BF16 |
| End-to-end pipeline speedup | **~55%** total time reduction (all quantization combined) |
| RAG consistency improvement | **~23%** higher cross-scene CLIP consistency vs no-RAG |
| Models in pipeline | **6** (LLM + image diffusion + video diffusion + audio diffusion + TTS + CLIP) |

---

## Architecture

```
Story text / voice
        │
        ▼
   Mistral-7B (Ollama)          ← LLM scene planning
   + ChromaDB RAG               ← character + style consistency
        │
        ▼
   SDXL + style LoRA            ← keyframe image generation
   + IP-Adapter                 ← identity conditioning (your face or auto-generated)
   + INT8 quantization
        │
        ├─────────────────┐
        ▼                 ▼
   CogVideoX-5B I2V   AudioLDM2 + XTTS-v2    ← parallel generation
   + RIFE 8→24fps     soundscape + narration
        │                 │
        └────────┬────────┘
                 ▼
           FFmpeg assembly
           (audio mixing + concat + fades)
                 │
                 ▼
           Final MP4 film
```

**6 AI models. 6 Docker microservices. 1 film.**

Full technical design, model selection rationale, V1→V2 tradeoffs, and evaluation methodology: see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Features

- **Multimodal input**: type your story or speak it (Whisper ASR)
- **4 visual styles**: cinematic, noir, anime, watercolour
- **Custom style LoRA**: trained on a curated dataset for visual consistency across all scenes
- **Identity preservation**: upload your photo to appear as the protagonist via IP-Adapter; auto-generates a consistent character face if no photo provided
- **Voice cloning**: upload 6+ seconds of your voice for XTTS-v2 voice cloning; 5 built-in preset voices as fallback
- **Ambient soundscapes**: AudioLDM2 generates scene-matched sound design per scene
- **24fps output**: CogVideoX-5B generates motion, RIFE interpolates to cinematic frame rate
- **CLIP quality report**: fidelity and consistency scores per scene
- **Full Docker deployment**: one command brings up all services

---

## Quick Start

### Requirements
- NVIDIA GPU with 12GB+ VRAM (T4 or better)
- Docker + nvidia-container-toolkit
- 50GB disk space (model weights)

### Run with Docker
```bash
git clone https://github.com/your-username/CineForge
cd CineForge

# Copy and configure environment
cp .env.example .env

# Build and start all services
docker-compose up --build

# Access the UI
open http://localhost:7860
```

### Run locally (development)
```bash
pip install -r requirements.txt

# Start Ollama and pull Mistral
ollama serve &
ollama pull mistral

# Start each service
python services/llm_service/main.py &
python services/image_service/main.py &
python services/video_service/main.py &
python services/audio_service/main.py &
python services/assembly_service/main.py &
python services/gradio_ui/app.py
```

### CLI generation
```bash
python run_pipeline.py \
  --story "A detective walks through fog-laden streets at midnight..." \
  --style noir \
  --voice deep_male \
  --output my_film.mp4
```

---

## Training Your Own Style LoRA

```bash
# 1. Add 30-50 style images to training/style_dataset/
# 2. Run training (Colab T4 recommended — free)
python training/lora_train.py \
  --data_dir training/style_dataset \
  --style_name cinestyle \
  --num_steps 1000

# Trained weights saved to models/style_lora.safetensors
# Set STYLE_LORA_PATH=models/style_lora.safetensors in .env
```

---

## Evaluation

```bash
# Latency benchmark (FP32 vs quantized per model)
python evaluation/benchmark_latency.py

# RAG vs no-RAG coherence comparison
python evaluation/rag_coherence_comparison.py
```

---

## Project Structure

```
CineForge/
├── services/
│   ├── llm_service/          # Mistral-7B + ChromaDB RAG scene planner
│   ├── image_service/        # SDXL + LoRA + IP-Adapter keyframe generator
│   ├── video_service/        # CogVideoX-5B I2V + RIFE interpolator
│   ├── audio_service/        # AudioLDM2 soundscape + XTTS-v2 narrator
│   ├── assembly_service/     # FFmpeg film assembler
│   └── gradio_ui/            # 4-tab Gradio interface + CLIP scorer
├── training/
│   └── lora_train.py         # DreamBooth SDXL LoRA training
├── evaluation/
│   ├── benchmark_latency.py  # Per-model FP32 vs quantized timing
│   └── rag_coherence_comparison.py  # RAG consistency measurement
├── schemas.py                # Shared Pydantic models
├── config.py                 # Shared configuration
├── run_pipeline.py           # CLI film generation
├── docker-compose.yml        # Single-command deployment
├── supervisord.conf          # Process management
├── Dockerfile
└── ARCHITECTURE.md           # Full design document
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Mistral-7B (Ollama, 4-bit GGUF) |
| RAG | ChromaDB + sentence-transformers |
| Image generation | Stable Diffusion XL + custom LoRA |
| Identity conditioning | IP-Adapter (SDXL) |
| Video generation | CogVideoX-5B I2V |
| Frame interpolation | RIFE (linear fallback) |
| Audio generation | AudioLDM2 |
| TTS / Voice cloning | XTTS-v2 (Coqui) |
| Quality scoring | CLIP ViT-B/32 (OpenCLIP) |
| Video assembly | FFmpeg |
| API framework | FastAPI + uvicorn |
| UI | Gradio |
| Process management | Supervisor |
| Containerization | Docker + docker-compose |
| Quantization | bitsandbytes (INT8), torchao (FP8), GGUF (4-bit) |

---

## Author

**Darsh**

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Model weights are subject to their respective licenses:
- SDXL: CreativeML Open RAIL-M
- CogVideoX-5B: Apache 2.0
- AudioLDM2: MIT
- XTTS-v2: Coqui Public Model License
- Mistral-7B: Apache 2.0A
