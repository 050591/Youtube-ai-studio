
"""
Multi-Modal YouTube Viral, 2D/3D Image, Music & Cinematic Video AI Agent
========================================================================
Orchestrates:
1. YouTube Trend & Hook Research (YouTube Data API v3 + Gemini)
2. 2D Image & Thumbnail Generation (Gemini Nano Banana / gemini-3.1-flash-image)
3. 3D Model Generation (Tripo AI API)
4. Suno-style Song & Music Generation (ElevenLabs Eleven Music API)
5. Optional Cinematic MP4 Video Generation (Gemini Video / Veo API)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, TypedDict

import requests
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

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
    image_2d_path: str
    model_3d_task_id: str
    music_audio_path: str
    video_mp4_path: str
    warnings: list[str]


def _call_with_retry(fn, retries: int = 3, delay: float = 3.0):
    """Retries transient 503/429 API spikes automatically."""
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_err = exc
            if attempt < retries - 1 and ("503" in str(exc) or "UNAVAILABLE" in str(exc)):
                time.sleep(delay * (attempt + 1))
                continue
            raise
    raise last_err


# ---------------------------------------------------------------------------
# 2. Node 1: YouTube Algorithm & Trend Researcher
# ---------------------------------------------------------------------------
def research_youtube(state: AgentState) -> dict[str, Any]:
    """Fetches top-performing YouTube videos for the topic to analyze hooks."""
    topic = state["user_prompt"]
    api_key = os.getenv("YOUTUBE_API_KEY")

    if state["dry_run"] or not api_key:
        sample_trends = [
            {
                "title": f"{topic} — Wait Till The End! (99% Miss This)",
                "views": "1,450,000",
                "hook_pattern": "Curiosity gap + fast visual reveal in first 2s",
            },
            {
                "title": f"Why {topic} Went Viral Overnight",
                "views": "890,000",
                "hook_pattern": "Bold contrarian claim + rhythmic beat drop",
            },
        ]
        return {"trending_videos": sample_trends}

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


# ---------------------------------------------------------------------------
# 3. Node 2: Viral Hook, Script & Cross-Modal Prompt Architect
# ---------------------------------------------------------------------------
def write_hooks_and_prompts(state: AgentState) -> dict[str, Any]:
    """Generates viral Tamil/English hooks and specialized prompts for 2D, 3D, Music & Video."""
    topic = state["user_prompt"]
    fmt = state["content_format"]
    trends = state.get("trending_videos", [])
    gemini_key = os.getenv("GEMINI_API_KEY")

    if state["dry_run"] or not gemini_key:
        hook_pkg = {
            "viral_titles": [
                f"{topic} — முதல் 3 வினாடியில் ஆச்சரியம்!",
                f"Nobody Expected This {topic} Twist!",
            ],
            "first_3s_hook": "நில்லுங்க! இதை நீங்க இதற்கு முன் பார்த்திருக்கவே மாட்டீங்க...",
            "retention_loop": "End with an unresolved visual cliffhanger looping back to frame 1.",
            "prompt_2d": (
                f"Vibrant 3D Pixar-style illustration for '{topic}', expressive hero character "
                "centered in 9:16 vertical frame, dramatic studio rim lighting, clean composition"
            ),
            "prompt_3d": f"Stylized cute 3D character asset for {topic}, clean topology, game-ready",
            "prompt_music": (
                f"Upbeat energetic Tamil-fusion pop beat for {topic}, catchy hook melody, "
                "120 BPM, punchy bass, bright synths"
            ),
            "prompt_video": (
                f"Cinematic 3D animated scene of {topic}, smooth camera push-in, dramatic golden hour "
                "lighting, expressive character motion, shallow depth of field"
            ),
        }
        return {"hook_package": hook_pkg}

    from google import genai

    client = genai.Client(api_key=gemini_key)
    system_prompt = (
        "You are a YouTube algorithm strategist & creative director. "
        f"Format: {fmt}. Trends: {json.dumps(trends, ensure_ascii=False)}. "
        "Return ONLY valid JSON with keys: viral_titles (list of 2 Tamil/English titles), "
        "first_3s_hook (Tamil), retention_loop (string), prompt_2d (English), "
        "prompt_3d (English), prompt_music (English), prompt_video (cinematic English video prompt)."
    )
    response = _call_with_retry(
        lambda: client.models.generate_content(
            model="gemini-3.6-flash",
            contents=f"{system_prompt}\n\nUser Topic: {topic}",
            config={"response_mime_type": "application/json"},
        )
    )
    hook_pkg = json.loads(response.text)
    return {"hook_package": hook_pkg}


# ---------------------------------------------------------------------------
# 4. Node 3: 2D Image & 3D Model Generator
# ---------------------------------------------------------------------------
def generate_2d_and_3d(state: AgentState) -> dict[str, Any]:
    """Generates a 2D image via Gemini Nano Banana and submits a 3D task to Tripo AI."""
    pkg = state["hook_package"]
    gemini_key = os.getenv("GEMINI_API_KEY")
    tripo_key = os.getenv("TRIPO_API_KEY")
    warnings = list(state.get("warnings", []))

    image_path = str(OUTPUT_DIR / "generated_2d_scene.png")
    tripo_task_id = "dry_run_tripo_task_001"

    # --- 2D Generation (Gemini Nano Banana) ---
    if not state["dry_run"] and gemini_key:
        try:
            from google import genai

            client = genai.Client(api_key=gemini_key)
            res = _call_with_retry(
                lambda: client.models.generate_content(
                    model="gemini-3.1-flash-image",
                    contents=pkg["prompt_2d"],
                )
            )
            for part in res.candidates[0].content.parts:
                if getattr(part, "inline_data", None):
                    Path(image_path).write_bytes(part.inline_data.data)
                    break
        except Exception as exc:
            warnings.append(f"2D Image generation skipped: {exc}")

    # --- 3D Generation (Tripo AI API) ---
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

    return {"image_2d_path": image_path, "model_3d_task_id": tripo_task_id, "warnings": warnings}


# ---------------------------------------------------------------------------
# 5. Node 4: Suno-Style Music & Song Composer (ElevenLabs Music API)
# ---------------------------------------------------------------------------
def compose_music(state: AgentState) -> dict[str, Any]:
    """Generates custom music/song from prompt via Eleven Music API."""
    pkg = state["hook_package"]
    eleven_key = os.getenv("ELEVENLABS_API_KEY")
    warnings = list(state.get("warnings", []))
    audio_path = str(OUTPUT_DIR / "generated_song.mp3")

    if not state["dry_run"] and eleven_key:
        try:
            resp = requests.post(
                "https://api.elevenlabs.io/v1/music",
                headers={"xi-api-key": eleven_key, "Content-Type": "application/json"},
                json={
                    "prompt": pkg["prompt_music"],
                    "music_length_ms": 30000,
                },
                timeout=120,
            )
            resp.raise_for_status()
            Path(audio_path).write_bytes(resp.content)
        except Exception as exc:
            warnings.append(f"Music generation skipped: {exc}")

    return {"music_audio_path": audio_path, "warnings": warnings}


# ---------------------------------------------------------------------------
# 6. Node 5: Cinematic MP4 Video Generator (Gemini Video / Veo API)
# ---------------------------------------------------------------------------
def generate_cinematic_video(state: AgentState) -> dict[str, Any]:
    """Generates a cinematic MP4 clip using Gemini Video API when enabled."""
    if not state.get("generate_video") or state["dry_run"]:
        return {"video_mp4_path": ""}

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return {"video_mp4_path": ""}

    pkg = state["hook_package"]
    warnings = list(state.get("warnings", []))
    video_path = str(OUTPUT_DIR / "cinematic_clip.mp4")
    aspect_ratio = "9:16" if state["content_format"] == "shorts" else "16:9"

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=gemini_key)
        operation = client.models.generate_videos(
            model="veo-3.1-generate-001",
            prompt=pkg.get("prompt_video", pkg["prompt_2d"]),
            config=types.GenerateVideosConfig(aspect_ratio=aspect_ratio),
        )
        while not operation.done:
            time.sleep(10)
            operation = client.operations.get(operation)

        generated = operation.response.generated_videos[0]
        client.files.download(file=generated.video)
        generated.video.save(video_path)
        return {"video_mp4_path": video_path, "warnings": warnings}
    except Exception as exc:
        warnings.append(f"Cinematic Video (Veo) skipped: {exc}")
        return {"video_mp4_path": "", "warnings": warnings}


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
    parser.add_argument("--video", action="store_true", help="Generate cinematic MP4 via Veo")
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
            "image_2d_path": "",
            "model_3d_task_id": "",
            "music_audio_path": "",
            "video_mp4_path": "",
            "warnings": [],
        }
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

