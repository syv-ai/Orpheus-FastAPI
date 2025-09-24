# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Starting the Server
```bash
# Native installation
python app.py

# Or with uvicorn (for development with reload)
uvicorn app:app --host 0.0.0.0 --port 5005 --reload
```

### Docker Compose Options
```bash
# GPU (CUDA) support
docker compose -f docker-compose-gpu.yml up

# ROCm GPU support  
docker compose -f docker-compose-gpu-rocm.yml up

# CPU support
docker compose -f docker-compose-cpu.yml up
```

### Dependencies
```bash
# Install PyTorch with CUDA support first
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Or for ROCm support
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4/

# Then install other dependencies
pip3 install -r requirements.txt
```

## Architecture Overview

### Core Components
- **app.py**: Main FastAPI server that handles HTTP requests, serves web UI, and provides OpenAI-compatible API endpoints
- **tts_engine/inference.py**: Manages LLM token generation, API communication with external inference servers, and hardware optimization
- **tts_engine/speechpipe.py**: Converts token sequences to audio using the SNAC model with CUDA/ROCm acceleration
- **tts_engine/__init__.py**: Package exports for the TTS engine

### Request Flow
1. FastAPI receives text input via `/v1/audio/speech` (OpenAI-compatible) or `/speak` endpoints  
2. `inference.py` sends text prompts to external LLM inference server to generate tokens
3. `speechpipe.py` converts tokens to audio using SNAC model with hardware-optimized processing
4. Audio is processed with crossfade stitching for long texts and returned as WAV format

### External Dependencies
- Requires separate LLM inference server running Orpheus model (llama.cpp, LM Studio, GPUStack, etc.)
- Connects via `ORPHEUS_API_URL` environment variable
- Supports quantized models: Q2_K (fastest), Q4_K_M (balanced), Q8_0 (highest quality)

### Hardware Optimization
- **High-End GPU Mode**: 16GB+ VRAM or compute capability 8.0+ triggers advanced parallel processing with 4 workers
- **Standard GPU Mode**: CUDA acceleration with GPU-optimized parameters  
- **CPU Mode**: Conservative processing with 2 workers and smaller batch sizes
- Automatic hardware detection configures optimal batch sizes and processing parameters

### Long Text Processing
- Automatically splits texts >1000 characters into manageable chunks
- Processes each chunk independently for reliability
- Combines audio segments with 50ms crossfades for seamless output
- Supports unlimited text length through intelligent batching

## Configuration

### Environment Variables (.env file)
Key variables for development and deployment:
- `ORPHEUS_API_URL`: External LLM inference server endpoint
- `ORPHEUS_MAX_TOKENS`: Maximum tokens to generate (default: 8192)  
- `ORPHEUS_API_TIMEOUT`: Request timeout in seconds (default: 120)
- `ORPHEUS_MODEL_NAME`: Model variant (Q2_K, Q4_K_M, or Q8_0)
- `ORPHEUS_PORT`: Web server port (default: 5005)
- `ORPHEUS_SAMPLE_RATE`: Audio sample rate (default: 24000)

### Voice Support
- 24 voices across 8 languages (English, French, German, Korean, Hindi, Mandarin, Spanish, Italian)
- Emotion tags supported: `<laugh>`, `<sigh>`, `<chuckle>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`
- Voice definitions in `AVAILABLE_VOICES` list in `inference.py`

## Token Processing Details
- Context window: 49 tokens (7²) for mathematical alignment
- Batch processing: 7 tokens (Orpheus model standard)
- Fixed repetition penalty: 1.1 (hardcoded for optimal quality)
- Parallel processing with CUDA streams for RTX GPUs

## API Compatibility
- OpenAI-compatible `/v1/audio/speech` endpoint
- Legacy `/speak` endpoint for simpler integration
- Supports OpenWebUI integration as TTS provider
- WAV format output only (MP3 conversion not implemented)