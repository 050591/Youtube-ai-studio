"""
Multi-Modal YouTube Viral, Multi-Scene Shorts MP4, 2D/3D & Voice AI Agent
=========================================================================
Upgraded with:
1. Gemini 2.5 Flash Viral Storyboard (Supports both AIzaSy & AQ. Express Keys)
2. Multi-Scene 2D Images (Gemini Imagen 3 -> HF FLUX.1 -> Pollinations -> Pillow)
3. Natural Neural Voiceover (Edge-TTS Tamil/English -> gTTS fallback)
4. Audio-Synced Multi-Scene MP4 with Ken Burns Motion & Captions
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import quote

import edge_tts
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import requests
from dotenv import load_dotenv
from gtts import gTTS
from langgraph.graph import END, START, StateGraph
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

OUTPUT_DIR = Path("agent_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Agent State Definition
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    user_prompt: str
    content_format: str  # "shorts" | "long_form"
    dry_run: bool
    generate_video: bool
    trending_videos: list[dict[str, Any]]
    hook_package: dict[str, Any]
    scene_image_paths: list[str]
    scene_audio_paths: list[str]
    image_2d_path: str
    model_3d_task_id: str
    music_audio_path: str
    video_mp4_path: str
    warnings: list[str]


def _is_tamil(text: str) -> bool:
    return bool(re.search(r"[\u0B80-\u0BFF]", text))


def _call_with_retry(fn, retries: int = 3, delay: float = 3.0):
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_err = exc
            if attempt < retries - 1 and ("503" in str(exc) or "429" in str(exc) or "UNAVAILABLE" in str(exc)):
                time.sleep(delay * (attempt + 1))
                continue
            raise
    raise last_err


def _create_fallback_image(path: Path, width: int, height: int, idx: int = 0) -> None:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    shift = (idx * 25) % 80
    arr[:, :, 0] = np.linspace(25 + shift, 95, height, dtype=np.uint8)[:, None]
    arr[:, :, 1] = np.linspace(40, 130 + shift // 2, height, dtype=np.uint8)[:, None]
    arr[:, :, 2] = np.linspace(95, 210, height, dtype=np.uint8)[:, None]
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        [width // 4, height // 3, 3 * width // 4, height // 3 + width // 2],
        outline=(255, 220, 120),
        width=6
