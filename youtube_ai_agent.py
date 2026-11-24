"""
YouTube AI Studio — Automated Prompt-to-Video Agent (LangGraph)
================================================================
Generates YouTube Shorts (9:16 MP4) from a single prompt:
1. YouTube Trend & Hook Research (YouTube Data API v3 + Gemini 3.6 Flash)
2. Multi-Character Storyboard & Spoken Dialogue (Gemini 3.6 Flash)
3. Distinct Character Voiceovers (Edge-TTS Neural Voices)
4. Real AI Video Clips per Scene (HF InferenceClient text_to_video -> hf-inference FLUX -> Pollinations -> Pillow fallback)
5. AI Background Music (ElevenLabs Music API) + FFmpeg Audio/Video Mixing
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, TypedDict

import edge_tts
import requests
from google import genai
from huggingface_hub import InferenceClient
from langgraph.graph import END, START, StateGraph
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path("agent_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Agent State
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    user_prompt: str
    language: str  # "Tamil" | "English"
    content_format: str  # "shorts" | "long_form"
    dry_run: bool
    trending_videos: list[dict[str, Any]]
    storyboard: dict[str, Any]
    scene_assets: list[dict[str, Any]]
    music_path: str
    final_video_path: str
    logs: list[str]


# ---------------------------------------------------------------------------
# Voice Mapping for Distinct Animated Characters
# ---------------------------------------------------------------------------
EDGE_VOICE_MAP = {
    ("Tamil", "female"): ("ta-IN-PallaviNeural", "+0Hz", "+0%"),
    ("Tamil", "male"): ("ta-IN-ValluvarNeural", "+0Hz", "+0%"),
    ("Tamil", "child"): ("ta-IN-PallaviNeural", "+35Hz", "+8%"),
    ("English", "female"): ("en-IN-NeerjaNeural", "+0Hz", "+0%"),
    ("English", "male"): ("en-IN-PrabhatNeural", "+0Hz", "+0%"),
    ("English", "child"): ("en-US-AnaNeural", "+15Hz", "+5%"),
}


def _get_secret(name: str) -> str:
    return os.getenv(name, "").strip()


def _run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _get_media_duration(file_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return max(1.5, float(res.stdout.strip()))


def _make_fallback_card(character_name: str, scene_num: int, out_path: str) -> None:
    """Creates a clean 720x1280 stylized card if all external image/video APIs fail."""
    img = Image.new("RGB", (720, 1280), color=(24, 24, 56))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([(48, 120), (672, 1160)], radius=40, fill=(38, 42, 92), outline=(129, 140, 248), width=4)
    font = ImageFont.load_default()
    draw.text((120, 560), f"Scene {scene_num}: {character_name}", fill=(255, 255, 255), font=font)
    img.save(out_path)


# ---------------------------------------------------------------------------
# 2. Node 1: YouTube Trend & Hook Researcher
# ---------------------------------------------------------------------------
def research_youtube(state: AgentState) -> dict[str, Any]:
    topic = state["user_prompt"]
    api_key = _get_secret("YOUTUBE_API_KEY")
    logs = list(state.get("logs", []))

    if state.get("dry_run") or not api_key:
        logs.append("Trend Research: Using sample viral hook patterns (Dry Run / No YOUTUBE_API_KEY).")
        return {
            "trending_videos": [
                {"title": f"{topic} — Wait Till The End!", "channel": "Sample Creator"},
                {"title": f"3 Things About {topic}", "channel": "Shorts Lab"},
            ],
            "logs": logs,
        }

    try:
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": topic,
                "type": "video",
                "order": "viewCount",
                "maxResults": 5,
                "key": api_key,
            },
            timeout=20,
        )
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
        logs.append(f"Trend Research: Fetched {len(trends)} top YouTube references.")
        return {"trending_videos": trends, "logs": logs}
    except Exception as exc:
        logs.append(f"Trend Research warning: {exc}")
        return {"trending_videos": [], "logs": logs}


# ---------------------------------------------------------------------------
# 3. Node 2: Multi-Character Storyboard & Spoken Script Director
# ---------------------------------------------------------------------------
def plan_character_storyboard(state: AgentState) -> dict[str, Any]:
    topic = state["user_prompt"]
    language = state.get("language", "Tamil")
    trends = state.get("trending_videos", [])
    gemini_key = _get_secret("GEMINI_API_KEY")
    logs = list(state.get("logs", []))

    if state.get("dry_run") or not gemini_key:
        storyboard = {
            "title": f"{topic} — Animated Short",
            "description": f"AI-generated animated story about {topic}. #Shorts #AI",
            "music_prompt": "Upbeat playful cinematic background music, bright marimba and warm synth bass, 115 BPM, no vocals",
            "scenes": [
                {
                    "scene_number": 1,
                    "character_name": "Kavi",
                    "voice_role": "female",
                    "dialogue": "நில்லுங்க! இன்னைக்கு ஒரு சூப்பரான விஷயம் பார்க்கப் போறோம்!",
                    "video_prompt": f"3D Pixar-style animated young girl character speaking excitedly to camera, expressive face and hand gestures, colorful background about {topic}, vertical 9:16, smooth motion",
                },
                {
                    "scene_number": 2,
                    "character_name": "Arjun",
                    "voice_role": "male",
                    "dialogue": "நிஜமாவா? அது எப்படி வேலை செய்யுதுன்னு கொஞ்சம் சொல்லேன்!",
                    "video_prompt": f"3D Pixar-style animated friendly boy character looking curious and talking, dynamic lighting, vertical 9:16, smooth animation about {topic}",
                },
                {
                    "scene_number": 3,
                    "character_name": "Kavi",
                    "voice_role": "female",
                    "dialogue": "ரொம்ப சிம்பிள்! இதை நீங்களும் ட்ரை பண்ணிப் பாருங்க, வீடியோ பிடிச்சா சப்ஸ்கிரைப் பண்ணுங்க!",
                    "video_prompt": f"3D Pixar-style animated girl smiling and waving goodbye, vibrant studio lighting, vertical 9:16, smooth cinematic motion about {topic}",
                },
            ],
        }
        logs.append("Storyboard: Built sample 3-scene multi-character script (Dry Run).")
        return {"storyboard": storyboard, "logs": logs}

    client = genai.Client(api_key=gemini_key)
    lang_rule = (
        "Write all 'dialogue' lines in natural spoken Tamil script (conversational, NOT slide bullet points)."
        if language == "Tamil"
        else "Write all 'dialogue' lines in natural spoken English."
    )

    system_prompt = f"""
You are an animation director for viral YouTube Shorts.
Create a 3-scene animated short where distinct characters speak naturally to each other or the viewer.
{lang_rule}
Trending context: {json.dumps(trends, ensure_ascii=False)}

Return ONLY valid JSON (no markdown fences) with this exact structure:
{{
  "title": "Catchy YouTube Shorts title",
  "description": "Short YouTube description with 3 hashtags",
  "music_prompt": "English prompt for instrumental background music matching the mood",
  "scenes": [
    {{
      "scene_number": 1,
      "character_name": "Character name",
      "voice_role": "female" | "male" | "child",
      "dialogue": "Natural spoken line (1-2 short sentences, ~4 seconds when spoken)",
      "video_prompt": "Detailed English 3D/2D animation text-to-video prompt showing the character speaking and acting, vertical 9:16, vibrant lighting"
    }}
  ]
}}
"""
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=f"{system_prompt}\n\nUser Prompt: {topic}",
    )
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    storyboard = json.loads(raw)
    logs.append(f"Storyboard: Planned {len(storyboard.get('scenes', []))} animated character scenes.")
    return {"storyboard": storyboard, "logs": logs}


# ---------------------------------------------------------------------------
# 4. Node 3: Distinct Character Voiceovers (Edge-TTS Neural Voices)
# ---------------------------------------------------------------------------
async def _synthesize_edge_tts(text: str, voice: str, pitch: str, rate: str, out_path: str) -> None:
    communicate = edge_tts.Communicate(text=text, voice=voice, pitch=pitch, rate=rate)
    await communicate.save(out_path)


def generate_character_voices(state: AgentState) -> dict[str, Any]:
    storyboard = state["storyboard"]
    language = state.get("language", "Tamil")
    lang_key = "Tamil" if language == "Tamil" else "English"
    logs = list(state.get("logs", []))
    run_id = uuid.uuid4().hex[:8]

    scene_assets: list[dict[str, Any]] = []
    for scene in storyboard.get("scenes", []):
        idx = scene["scene_number"]
        role = scene.get("voice_role", "female").lower()
        if role not in ("female", "male", "child"):
            role = "female"

        voice_name, pitch, rate = EDGE_VOICE_MAP.get(
            (lang_key, role), EDGE_VOICE_MAP[(lang_key, "female")]
        )
        audio_path = str(OUTPUT_DIR / f"{run_id}_scene_{idx}_voice.mp3")
        asyncio.run(_synthesize_edge_tts(scene["dialogue"], voice_name, pitch, rate, audio_path))
        duration = _get_media_duration(audio_path)

        scene_assets.append(
            {
                **scene,
                "voice_id_used": voice_name,
                "audio_path": audio_path,
                "audio_duration": duration,
            }
        )
        logs.append(
            f"Voiceover Scene {idx} ({scene.get('character_name', role)}): {voice_name} ({duration:.1f}s)"
        )

    return {"scene_assets": scene_assets, "logs": logs}


# ---------------------------------------------------------------------------
# 5. Node 4: Real AI Video Clips per Scene (Multi-Tier Fallback)
# ---------------------------------------------------------------------------
def generate_video_clips(state: AgentState) -> dict[str, Any]:
    hf_token = _get_secret("HF_TOKEN")
    scene_assets = list(state.get("scene_assets", []))
    logs = list(state.get("logs", []))
    run_id = uuid.uuid4().hex[:8]

    video_client = InferenceClient(provider="auto", api_key=hf_token, timeout=180) if hf_token else None
    image_client = InferenceClient(provider="hf-inference", api_key=hf_token, timeout=60) if hf_token else None

    updated_assets: list[dict[str, Any]] = []
    for asset in scene_assets:
        idx = asset["scene_number"]
        target_dur = asset["audio_duration"]
        raw_clip_path = str(OUTPUT_DIR / f"{run_id}_scene_{idx}_raw.mp4")
        fitted_clip_path = str(OUTPUT_DIR / f"{run_id}_scene_{idx}_fitted.mp4")

        if state.get("dry_run"):
            _run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=c=#1e1b4b:s=720x1280:d={target_dur:.2f}",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    fitted_clip_path,
                ]
            )
            logs.append(f"Video Scene {idx}: Created Dry Run clip ({target_dur:.1f}s).")
            updated_assets.append({**asset, "video_path": fitted_clip_path})
            continue

        # Tier 1: Hugging Face Text-to-Video (Wan2.2)
        video_ok = False
        if video_client:
            try:
                video_bytes = video_client.text_to_video(
                    asset["video_prompt"],
                    model="Wan-AI/Wan2.2-TI2V-5B",
                )
                with open(raw_clip_path, "wb") as f:
                    f.write(video_bytes)
                _run_ffmpeg(
                    [
                        "ffmpeg",
                        "-y",
                        "-stream_loop",
                        "-1",
                        "-i",
                        raw_clip_path,
                        "-t",
                        f"{target_dur:.2f}",
                        "-vf",
                        "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,fps=24",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        fitted_clip_path,
                    ]
                )
                logs.append(f"Video Scene {idx}: Generated AI video via Wan2.2 ({target_dur:.1f}s).")
                video_ok = True
            except Exception as exc:
                logs.append(f"Video Scene {idx} Wan2.2 skipped ({type(exc).__name__}): {exc}")

        if not video_ok:
            img_path = str(OUTPUT_DIR / f"{run_id}_scene_{idx}.png")
            img_ok = False

            # Tier 2: Hugging Face Free Serverless FLUX.1-schnell (hf-inference)
            if image_client:
                try:
                    img = image_client.text_to_image(
                        asset["video_prompt"],
                        model="black-forest-labs/FLUX.1-schnell",
                    )
                    img.save(img_path)
                    logs.append(f"Video Scene {idx}: Generated FLUX.1-schnell visual (hf-inference).")
                    img_ok = True
                except Exception as exc:
                    logs.append(f"Video Scene {idx} HF FLUX skipped ({type(exc).__name__}): {exc}")

            # Tier 3: Free Pollinations.ai fallback (no token required)
            if not img_ok:
                try:
                    encoded_prompt = urllib.parse.quote(asset["video_prompt"][:350])
                    pol_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=720&height=1280&nologo=true"
                    r = requests.get(pol_url, timeout=45)
                    r.raise_for_status()
                    with open(img_path, "wb") as f:
                        f.write(r.content)
                    logs.append(f"Video Scene {idx}: Generated visual via Pollinations fallback.")
                    img_ok = True
                except Exception as exc:
                    logs.append(f"Video Scene {idx} Pollinations skipped ({type(exc).__name__}): {exc}")

            # Tier 4: Guaranteed local Pillow card fallback so pipeline never crashes
            if not img_ok:
                _make_fallback_card(asset.get("character_name", "Character"), idx, img_path)
                logs.append(f"Video Scene {idx}: Used local fallback visual card.")

            frames = max(36, int(target_dur * 24))
            _run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-loop",
                    "1",
                    "-i",
                    img_path,
                    "-vf",
                    f"scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,zoompan=z='min(zoom+0.0015,1.12)':d={frames}:s=720x1280:fps=24",
                    "-t",
                    f"{target_dur:.2f}",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    fitted_clip_path,
                ]
            )

        updated_assets.append({**asset, "video_path": fitted_clip_path})

    return {"scene_assets": updated_assets, "logs": logs}


# ---------------------------------------------------------------------------
# 6. Node 5: AI Background Music (ElevenLabs Music API)
# ---------------------------------------------------------------------------
def generate_background_music(state: AgentState) -> dict[str, Any]:
    eleven_key = _get_secret("ELEVENLABS_API_KEY")
    storyboard = state["storyboard"]
    scene_assets = state.get("scene_assets", [])
    logs = list(state.get("logs", []))

    total_ms = int(sum(a["audio_duration"] for a in scene_assets) * 1000) + 1000
    total_ms = max(5000, min(total_ms, 60000))

    if state.get("dry_run") or not eleven_key:
        logs.append("Music: Skipped (Dry Run or ELEVENLABS_API_KEY not set in Secrets).")
        return {"music_path": "", "logs": logs}

    try:
        music_path = str(OUTPUT_DIR / f"{uuid.uuid4().hex[:8]}_bgm.mp3")
        resp = requests.post(
            "https://api.elevenlabs.io/v1/music",
            headers={
                "xi-api-key": eleven_key,
                "Content-Type": "application/json",
            },
            json={
                "prompt": storyboard.get("music_prompt", "Upbeat background music, no vocals"),
                "music_length_ms": total_ms,
            },
            timeout=120,
        )
        resp.raise_for_status()
        with open(music_path, "wb") as f:
            f.write(resp.content)
        logs.append("Music: Generated custom AI background track via ElevenLabs Music API.")
        return {"music_path": music_path, "logs": logs}
    except Exception as exc:
        logs.append(f"Music warning: {exc}")
        return {"music_path": "", "logs": logs}


# ---------------------------------------------------------------------------
# 7. Node 6: Final Video + Voiceover + Music Assembly
# ---------------------------------------------------------------------------
def assemble_final_video(state: AgentState) -> dict[str, Any]:
    scene_assets = state.get("scene_assets", [])
    music_path = state.get("music_path", "")
    logs = list(state.get("logs", []))
    run_id = uuid.uuid4().hex[:8]

    muxed_clips: list[str] = []
    for asset in scene_assets:
        idx = asset["scene_number"]
        muxed_path = str(OUTPUT_DIR / f"{run_id}_scene_{idx}_muxed.mp4")
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                asset["video_path"],
                "-i",
                asset["audio_path"],
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                muxed_path,
            ]
        )
        muxed_clips.append(muxed_path)

    concat_txt = OUTPUT_DIR / f"{run_id}_concat.txt"
    with open(concat_txt, "w", encoding="utf-8") as f:
        for clip in muxed_clips:
            f.write(f"file '{os.path.abspath(clip)}'\n")

    merged_voice_video = str(OUTPUT_DIR / f"{run_id}_merged_voice.mp4")
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_txt),
            "-c",
            "copy",
            merged_voice_video,
        ]
    )

    final_path = str(OUTPUT_DIR / f"youtube_short_{run_id}.mp4")
    if music_path and os.path.exists(music_path):
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                merged_voice_video,
                "-stream_loop",
                "-1",
                "-i",
                music_path,
                "-filter_complex",
                "[0:a]volume=1.0[vo];[1:a]volume=0.22[bgm];[vo][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                "-map",
                "0:v",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                final_path,
            ]
        )
        logs.append("Assembly: Final 9:16 MP4 rendered with character voices + background music.")
    else:
        os.replace(merged_voice_video, final_path)
        logs.append("Assembly: Final 9:16 MP4 rendered with character voices.")

    return {"final_video_path": final_path, "logs": logs}


# ---------------------------------------------------------------------------
# 8. Build & Run LangGraph Workflow
# ---------------------------------------------------------------------------
def build_youtube_agent():
    graph = StateGraph(AgentState)
    graph.add_node("research_youtube", research_youtube)
    graph.add_node("plan_character_storyboard", plan_character_storyboard)
    graph.add_node("generate_character_voices", generate_character_voices)
    graph.add_node("generate_video_clips", generate_video_clips)
    graph.add_node("generate_background_music", generate_background_music)
    graph.add_node("assemble_final_video", assemble_final_video)

    graph.add_edge(START, "research_youtube")
    graph.add_edge("research_youtube", "plan_character_storyboard")
    graph.add_edge("plan_character_storyboard", "generate_character_voices")
    graph.add_edge("generate_character_voices", "generate_video_clips")
    graph.add_edge("generate_video_clips", "generate_background_music")
    graph.add_edge("generate_background_music", "assemble_final_video")
    graph.add_edge("assemble_final_video", END)

    return graph.compile()


def run_youtube_agent(
    user_prompt: str,
    language: str = "Tamil",
    content_format: str = "shorts",
    dry_run: bool = False,
) -> AgentState:
    app = build_youtube_agent()
    initial_state: AgentState = {
        "user_prompt": user_prompt,
        "language": language,
        "content_format": content_format,
        "dry_run": dry_run,
        "logs": [],
    }
    return app.invoke(initial_state)
