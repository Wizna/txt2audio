# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a text-to-audio book converter (txt2audio) that transforms Chinese novels into audiobooks with video covers. The application processes text files, generates audio using TTS models, and creates MP4 videos with generated cover images.

## Commands

### Running the Application
- Main entry point: `uv run python src/transform_to_audio.py ../demo/《英雄志》（校对第1-22卷）作者：孙晓.txt`
- Alternative entry point using utility module: `uv run python -c "from utility import cli_main_process; cli_main_process()"`

### Dependencies Management with uv
- Install core dependencies: `uv sync`
- Add new dependency: `uv add package_name`
- Enter virtual environment: `uv shell` or `source .venv/bin/activate`
- Legacy support: Install from requirements.txt: `pip3 install -r requirements.txt`

### TTS Dependencies Installation
Due to system dependency requirements (MeCab), TTS installation may require additional steps:
- macOS: `brew install mecab mecab-ipadic` (if using Homebrew)
- Alternative: Use legacy pip installation: `pip3 install TTS==0.13.0`
- For development without TTS: Remove TTS from pyproject.toml dependencies temporarily

### Testing
- No formal test framework is configured in this project
- Manual testing involves running the main script with sample text files

## Architecture

### Core Components

1. **Main Processing Pipeline** (`utility.py:246-275`)
   - Entry point: `cli_main_process()` 
   - Handles file parsing, chapter extraction, audio generation, and video creation
   - Processes books in configurable ranges (user can specify chapters to convert)

2. **Text Processing** (`utility.py:167-234`)
   - `construct_text_and_name()`: Parses book structure using delimiter patterns
   - Supports chapter/volume delimiters: 卷章 (volume/chapter)
   - Generates table of contents automatically
   - Handles special sections: 序, 序章, 序言, 前言, 楔子, 引言, 后记, 终章

3. **Audio Generation** (`utility.py:66-91`)
   - Uses XTTS v2 model: `tts_models/multilingual/multi-dataset/xtts_v2`
   - Splits long text into manageable chunks (30-character limit for model)
   - Applies audio time stretching (1.24x speed) for better listening experience
   - Generates ~30min audio clips (6300 Chinese characters per clip)

4. **Video Generation** (`video.py:70-78`)
   - Creates video covers with dynamic colors based on book title hash
   - Uses custom fonts: YunFengFeiYunTi, YangRenDongZhuShiTi, DTM-Mono
   - Combines audio with static image using ffmpeg

### TTS Model Options

The project supports multiple TTS approaches:
- **XTTS v2** (current): `xtts_v2.py` - multilingual model with voice cloning
- **Bark** (alternative): `bark_util.py` - generative audio model  
- **Tacotron2** (legacy): Chinese-specific model in commented code

### File Organization

- **Input**: Text files (typically Chinese novels)
- **Output**: `output/{book_name}/{chapter_structure}/` containing WAV and MP4 files
- **Resources**: Voice samples (`female.wav`, `male.wav`), fonts, and dictionary files
- **Demo**: Sample text file for testing

### Key Configuration

- Character limit per audio segment: 6300 characters (~30 minutes)
- Audio processing: 1.24x time stretch for better pacing
- TTS model: Uses female voice sample for voice cloning
- Video format: 720x1280 portrait with generated covers

### Known Issues

- **MeCab dependency error** when installing TTS:
  - Root cause: Missing system MeCab library
  - Solution: `brew install mecab mecab-ipadic` or use pip fallback
  - Alternative: Temporarily remove TTS from pyproject.toml for development

- **"Decoder stopped with max_decoder_steps 500" error**: 
  - Fix: Modify `~/Library/Application Support/tts/tts_models--zh-CN--baker--tacotron2-DDC-GST/config.json`
  - Change `max_decoder_steps` to 10000

- **Text ending with certain punctuation** may cause processing issues

- **Numpy version conflicts**: TTS requires numpy<1.24, which may conflict with other packages

### Output Management

- Automatically resumes from last processed chapter
- Skips existing files to avoid reprocessing
- User can specify ranges for partial conversion
- Generates table of contents file for navigation