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
        width=6,
    )
    img.save(path, format="PNG")


def _draw_top_caption(img: Image.Image, text: str) -> Image.Image:
    """Draws a crisp top-center caption pill on a frame."""
    if not text:
        return img
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    try:
        font = ImageFont.load_default(size=max(26, w // 20))
    except TypeError:
        font = ImageFont.load_default()

    display_text = text[:42] + ("..." if len(text) > 42 else "")
    bbox = draw.textbbox((0, 0), display_text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 26, 16
    box_x1 = max(20, (w - tw) // 2 - pad_x)
    box_y1 = int(h * 0.08)
    box_x2 = min(w - 20, (w + tw) // 2 + pad_x)
    box_y2 = box_y1 + th + pad_y * 2

    draw.rounded_rectangle([box_x1, box_y1, box_x2, box_y2], radius=20, fill=(15, 23, 42))
    draw.text(((w - tw) // 2, box_y1 + pad_y), display_text, fill=(255, 235, 59), font=font)
    return out


def _get_audio_duration(audio_path: Path) -> float:
    """Measures exact MP3 duration using FFmpeg."""
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        res = subprocess.run(
            [ffmpeg_exe, "-i", str(audio_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", res.stderr)
        if match:
            hrs, mins, secs = float(match.group(1)), float(match.group(2)), float(match.group(3))
            return max(2.0, hrs * 3600 + mins * 60 + secs)
    except Exception:
        pass
    return 3.5


async def _synthesize_edge_tts(text: str, out_path: Path, voice: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


# ---------------------------------------------------------------------------
# 2. Node 1: YouTube Algorithm & Trend Researcher
# ---------------------------------------------------------------------------
def research_youtube(state: AgentState) -> dict[str, Any]:
    topic = state["user_prompt"]
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()

    if state["dry_run"] or not api_key:
        return {
            "trending_videos": [
                {"title": f"{topic} — Wait Till The End!", "views": "1,450,000"},
                {"title": f"Why {topic} Went Viral", "views": "890,000"},
            ]
        }

    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": topic,
            "type": "video",
            "order": "viewCount",
            "maxResults": 5,
            "key": api_key,
        }
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        trends = [
            {
                "title": item["snippet"]["title"],
                "channel": item["snippet"]["channelTitle"],
                "video_id": item["id"]["videoId"],
            }
            for item in items
        ]
        return {"trending_videos": trends}
    except Exception:
        return {"trending_videos": []}


# ---------------------------------------------------------------------------
# 3. Node 2: 4-Scene Viral Storyboard & Hook Architect
# ---------------------------------------------------------------------------
def write_hooks_and_prompts(state: AgentState) -> dict[str, Any]:
    topic = state["user_prompt"]
    fmt = state["content_format"]
    trends = state.get("trending_videos", [])
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    warnings = list(state.get("warnings", []))
    tamil_mode = _is_tamil(topic) or "tamil" in topic.lower()

    fallback_pkg = {
        "viral_titles": [
            f"{topic} — முதல் 3 வினாடியில் ஆச்சரியம்!",
            f"Nobody Expected This {topic} Twist!",
        ],
        "first_3s_hook": "நில்லுங்க! இதை நீங்க இதற்கு முன் பார்த்திருக்கவே மாட்டீங்க...",
        "retention_loop": "End with an unresolved cliffhanger looping seamlessly to scene 1.",
        "prompt_2d": f"Cute 2D storybook illustration of {topic}, warm pastel background",
        "prompt_3d": f"Stylized 3D character asset for {topic}, clean topology, game-ready",
        "prompt_music": f"Upbeat playful pop beat for {topic}, 120 BPM, bright synths",
        "scenes": [
            {
                "caption": "WAIT FOR IT...",
                "voice_line": "நில்லுங்க! இன்னைக்கு ஒரு செம ஆச்சரியம் காத்திருக்கு!" if tamil_mode else f"Look at this amazing {topic}!",
                "image_prompt": f"Cute 2D vibrant cartoon illustration of {topic}, curious expression, cinematic lighting",
            },
            {
                "caption": "THE BIG TWIST!",
                "voice_line": "திடீர்னு அங்க ஒரு பெரிய மாற்றம் நடந்துச்சு!" if tamil_mode else "Suddenly, something magical happened!",
                "image_prompt": f"Same cute 2D character from {topic} looking amazed at glowing sparkles, vibrant cartoon style",
            },
            {
                "caption": "UNBELIEVABLE!",
                "voice_line": "யாரும் எதிர்பார்க்காத ஒரு திருப்பம்!" if tamil_mode else "Up and away into the sky!",
                "image_prompt": f"Same cute 2D character from {topic} in an exciting action pose, vibrant cartoon style",
            },
            {
                "caption": "WATCH AGAIN!",
                "voice_line": "உங்களுக்கு பிடிச்சிருந்தா லைக் பண்ணுங்க!" if tamil_mode else "And that is how the magic begins!",
                "image_prompt": f"Same cute 2D character from {topic} laughing and waving happily, vibrant cartoon style",
            },
        ],
    }

    if state["dry_run"] or not gemini_key:
        return {"hook_package": fallback_pkg, "warnings": warnings}

    try:
        from google import genai

        client = genai.Client(api_key=gemini_key, vertexai=gemini_key.startswith("AQ."))
        voice_lang_rule = (
            "Natural spoken Tamil in Tamil Unicode script (தமிழ்)"
            if tamil_mode
            else "Natural punchy spoken English"
        )
        system_prompt = (
            "You are a viral YouTube Shorts director. "
            f"Format: {fmt}. Trends: {json.dumps(trends, ensure_ascii=False)}. "
            "Return ONLY valid JSON with keys: "
            "viral_titles (list of 2 catchy Tamil/English titles), "
            "first_3s_hook (Tamil), retention_loop (string), "
            "prompt_2d (English), prompt_3d (English), prompt_music (English), "
            "and scenes (list of EXACTLY 4 objects, each with: "
            "'caption' (short 3-5 word punchy English overlay text), "
            f"'voice_line' (1 short spoken sentence in {voice_lang_rule}), "
            "'image_prompt' (detailed English prompt keeping the SAME character & art style across all 4 scenes))."
        )
        response = _call_with_retry(
            lambda: client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{system_prompt}\n\nUser Topic: {topic}",
                config={"response_mime_type": "application/json"},
            )
        )
        hook_pkg = json.loads(response.text)
        return {"hook_package": hook_pkg, "warnings": warnings}
    except Exception as exc:
        warnings.append(f"Gemini API குறிப்பு ({type(exc).__name__}): {str(exc)[:120]} — Smart Template பயன்படுத்தப்பட்டது.")
        return {"hook_package": fallback_pkg, "warnings": warnings}


# ---------------------------------------------------------------------------
# 4. Node 3: Multi-Scene 2D Image & 3D Model Generator
# ---------------------------------------------------------------------------
def _fetch_single_image(prompt: str, path: Path, width: int, height: int, dry_run: bool, idx: int) -> bool:
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()

    if not dry_run and gemini_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_key, vertexai=gemini_key.startswith("AQ."))
            aspect = "9:16" if height > width else "16:9"
            res = client.models.generate_images(
                model="imagen-3.0-generate-002",
                prompt=prompt,
                config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio=aspect),
            )
            if res.generated_images:
                path.write_bytes(res.generated_images[0].image.image_bytes)
                return True
        except Exception:
            pass

    if not dry_run and hf_token:
        try:
            hf_url = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
            r = requests.post(
                hf_url,
                headers={"Authorization": f"Bearer {hf_token}"},
                json={"inputs": prompt, "parameters": {"width": width, "height": height}},
                timeout=60,
            )
            if r.status_code == 200 and len(r.content) > 5000:
                path.write_bytes(r.content)
                return True
        except Exception:
            pass

    if not dry_run:
        try:
            encoded = quote(prompt[:350])
            poll_url = (
                f"https://image.pollinations.ai/prompt/{encoded}"
                f"?width={width}&height={height}&nologo=true&model=flux"
            )
            r = requests.get(poll_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            if r.status_code == 200 and len(r.content) > 5000:
                path.write_bytes(r.content)
                return True
        except Exception:
            pass

    _create_fallback_image(path, width, height, idx=idx)
    return False


def generate_2d_and_3d(state: AgentState) -> dict[str, Any]:
    pkg = state["hook_package"]
    tripo_key = os.getenv("TRIPO_API_KEY", "").strip()
    warnings = list(state.get("warnings", []))

    width, height = (720, 1280) if state["content_format"] == "shorts" else (1280, 720)
    scenes = pkg.get("scenes") or [{"image_prompt": pkg.get("prompt_2d", state["user_prompt"])}]
    if not state.get("generate_video"):
        scenes = scenes[:1]

    scene_paths: list[str] = []
    any_fallback = False
    for idx, sc in enumerate(scenes[:4]):
        sc_path = OUTPUT_DIR / f"scene_{idx + 1}.png"
        ok = _fetch_single_image(sc["image_prompt"], sc_path, width, height, state["dry_run"], idx)
        if not ok and not state["dry_run"]:
            any_fallback = True
        scene_paths.append(str(sc_path))

    if any_fallback:
        warnings.append("சில காட்சிகளுக்கு fallback graphics பயன்படுத்தப்பட்டது (HF_TOKEN / GEMINI_API_KEY சரிபார்க்கவும்).")

    tripo_task_id = "dry_run_tripo_task_001"
    if not state["dry_run"] and tripo_key:
        try:
            tripo_resp = requests.post(
                "https://api.tripo3d.ai/v2/openapi/task",
                headers={"Authorization": f"Bearer {tripo_key}", "Content-Type": "application/json"},
                json={"type": "text_to_model", "prompt": pkg["prompt_3d"]},
                timeout=30,
            )
            tripo_resp.raise_for_status()
            tripo_task_id = tripo_resp.json().get("data", {}).get("task_id", "")
        except Exception as exc:
            warnings.append(f"3D generation skipped: {exc}")

    return {
        "scene_image_paths": scene_paths,
        "image_2d_path": scene_paths[0] if scene_paths else "",
        "model_3d_task_id": tripo_task_id,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 5. Node 4: Per-Scene Natural Neural Voiceover (Edge-TTS Tamil/English)
# ---------------------------------------------------------------------------
def compose_music(state: AgentState) -> dict[str, Any]:
    """Generates scene-by-scene Edge-TTS neural voiceover and merges full audio."""
    pkg = state["hook_package"]
    warnings = list(state.get("warnings", []))
    scenes = pkg.get("scenes", [])[:4]
    full_audio_path = OUTPUT_DIR / "generated_audio.mp3"
    scene_audio_paths: list[str] = []

    for idx, sc in enumerate(scenes):
        line = sc.get("voice_line", "").strip() or state["user_prompt"]
        sc_audio = OUTPUT_DIR / f"scene_audio_{idx + 1}.mp3"
        voice = "ta-IN-ValluvarNeural" if _is_tamil(line) else "en-US-ChristopherNeural"
        try:
            asyncio.run(_synthesize_edge_tts(line, sc_audio, voice))
            scene_audio_paths.append(str(sc_audio))
        except Exception:
            try:
                lang = "ta" if _is_tamil(line) else "en"
                gTTS(text=line, lang=lang, slow=False).save(str(sc_audio))
                scene_audio_paths.append(str(sc_audio))
            except Exception as exc:
                warnings.append(f"Scene {idx + 1} voice failed: {exc}")

    if scene_audio_paths:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        concat_txt = OUTPUT_DIR / "audio_list.txt"
        concat_txt.write_text(
            "\n".join(f"file '{Path(p).resolve()}'" for p in scene_audio_paths),
            encoding="utf-8",
        )
        try:
            subprocess.run(
                [ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(full_audio_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            warnings.append(f"Audio merge warning: {exc}")

    return {
        "scene_audio_paths": scene_audio_paths,
        "music_audio_path": str(full_audio_path) if full_audio_path.exists() else "",
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 6. Node 5: Audio-Synced Multi-Scene MP4 with Motion & Captions
# ---------------------------------------------------------------------------
def generate_cinematic_video(state: AgentState) -> dict[str, Any]:
    if not state.get("generate_video"):
        return {"video_mp4_path": ""}

    scene_paths = state.get("scene_image_paths") or []
    scene_audio_paths = state.get("scene_audio_paths") or []
    scenes = state.get("hook_package", {}).get("scenes", [])
    if not scene_paths:
        return {"video_mp4_path": ""}

    width, height = (720, 1280) if state["content_format"] == "shorts" else (1280, 720)
    raw_mp4 = OUTPUT_DIR / "raw_motion.mp4"
    final_mp4 = OUTPUT_DIR / "cinematic_clip.mp4"
    fps = 24

    writer = imageio.get_writer(str(raw_mp4), fps=fps, codec="libx264", pixelformat="yuv420p")
    for idx, img_file in enumerate(scene_paths):
        p = Path(img_file)
        if not p.exists():
            continue

        sec_per_scene = 3.2
        if idx < len(scene_audio_paths) and Path(scene_audio_paths[idx]).exists():
            sec_per_scene = _get_audio_duration(Path(scene_audio_paths[idx]))
        frames_per_scene = max(fps * 2, int(fps * sec_per_scene))

        caption = scenes[idx].get("caption", "") if idx < len(scenes) else ""
        base_img = Image.open(p).convert("RGB").resize((int(width * 1.16), int(height * 1.16)), Image.LANCZOS)
        bw, bh = base_img.size

        for f_idx in range(frames_per_scene):
            progress = f_idx / max(frames_per_scene - 1, 1)
            scale = 1.0 + 0.12 * progress
            crop_w = int(bw / scale)
            crop_h = int(bh / scale)
            left = (bw - crop_w) // 2
            top = (bh - crop_h) // 2
            frame = base_img.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.BILINEAR)
            frame = _draw_top_caption(frame, caption)
            writer.append_data(np.array(frame))
    writer.close()

    audio_path = Path(state.get("music_audio_path", ""))
    if audio_path.exists() and audio_path.stat().st_size > 1000:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i",
                str(raw_mp4),
                "-i",
                str(audio_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(final_mp4),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        raw_mp4.replace(final_mp4)

    return {"video_mp4_path": str(final_mp4)}


# ---------------------------------------------------------------------------
# 7. Build LangGraph Workflow
# ---------------------------------------------------------------------------
def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("research_youtube", research_youtube)
    graph.add_node("write_hooks_and_prompts", write_hooks_and_prompts)
    graph.add_node("generate_2d_and_3d", generate_2d_and_3d)
    graph.add_node("compose_music", compose_music)
    graph.add_node("generate_cinematic_video", generate_cinematic_video)

    graph.add_edge(START, "research_youtube")
    graph.add_edge("research_youtube", "write_hooks_and_prompts")
    graph.add_edge("write_hooks_and_prompts", "generate_2d_and_3d")
    graph.add_edge("generate_2d_and_3d", "compose_music")
    graph.add_edge("compose_music", "generate_cinematic_video")
    graph.add_edge("generate_cinematic_video", END)

    return graph.compile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multi-Modal YouTube AI Agent")
    parser.add_argument("--prompt", type=str, default="Cute 3D Squirrel Tamil Kids Song")
    parser.add_argument("--format", choices=["shorts", "long_form"], default="shorts")
    parser.add_argument("--live", action="store_true", help="Call live APIs using .env keys")
    parser.add_argument("--video", action="store_true", help="Generate Multi-Scene MP4")
    args = parser.parse_args()

    app = build_agent()
    result = app.invoke(
        {
            "user_prompt": args.prompt,
            "content_format": args.format,
            "dry_run": not args.live,
            "generate_video": args.video,
            "trending_videos": [],
            "hook_package": {},
            "scene_image_paths": [],
            "scene_audio_paths": [],
            "image_2d_path": "",
            "model_3d_task_id": "",
            "music_audio_path": "",
            "video_mp4_path": "",
            "warnings": [],
        }
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
