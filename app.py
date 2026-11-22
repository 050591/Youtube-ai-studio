import os
import json
import asyncio
import uuid
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from google import genai
from google.genai import types
import edge_tts
from moviepy.editor import (
    ColorClip, TextClip, CompositeVideoClip, AudioFileClip, concatenate_videoclips
)

app = FastAPI()

os.makedirs("static/outputs", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY"))

jobs = {}

class VideoRequest(BaseModel):
    prompt: str
    style: str = "2D Stickman Comedy"
    language: str = "Tamil"

async def generate_tts(text: str, output_path: str, voice: str = "ta-IN-ValluvarNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def generate_script_from_prompt(user_prompt: str, style: str, language: str):
    system_instruction = f"""
    You are an expert YouTube Shorts/Video creator. Convert the user's idea into a 4-scene viral video script.
    Style: {style}
    Voiceover Language: {language} (If Tamil, write natural spoken Tamil script in Tamil Unicode).
    
    Return strictly a JSON array of 4 objects with keys:
    - "scene_number": int
    - "visual_prompt": detailed English prompt for AI video/image generator
    - "voiceover": spoken dialogue for this scene (short, 1-2 sentences)
    - "caption": short punchy subtitle text
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Idea: {user_prompt}",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json"
        )
    )
    return json.loads(response.text)

async def process_video_pipeline(job_id: str, prompt: str, style: str, language: str):
    try:
        jobs[job_id]["status"] = "Generating Script & Scenes..."
        jobs[job_id]["progress"] = 20
        
        scenes = generate_script_from_prompt(prompt, style, language)
        jobs[job_id]["scenes"] = scenes
        
        voice_name = "ta-IN-ValluvarNeural" if language == "Tamil" else "en-US-ChristopherNeural"
        clips = []
        bg_colors = [(25, 35, 60), (45, 25, 65), (20, 55, 50), (65, 30, 35)]
        
        for idx, scene in enumerate(scenes):
            jobs[job_id]["status"] = f"Creating Scene {idx+1}/4 (Voice & Visuals)..."
            jobs[job_id]["progress"] = 30 + (idx * 15)
            
            audio_path = f"static/outputs/{job_id}_scene_{idx}.mp3"
            await generate_tts(scene["voiceover"], audio_path, voice=voice_name)
            
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration + 0.4
            
            bg_clip = ColorClip(size=(720, 1280), color=bg_colors[idx % len(bg_colors)], duration=duration)
            
            prompt_txt = TextClip(
                f"[Scene {idx+1}: {style}]\n{scene['visual_prompt'][:90]}...",
                fontsize=28, color="lightgray", size=(640, None), method="caption"
            ).set_position(("center", 350)).set_duration(duration)
            
            subtitle_txt = TextClip(
                scene["caption"],
                fontsize=44, color="yellow", bg_color="black", size=(650, None), method="caption"
            ).set_position(("center", 950)).set_duration(duration)
            
            scene_clip = CompositeVideoClip([bg_clip, prompt_txt, subtitle_txt]).set_audio(audio_clip)
            clips.append(scene_clip)
            
        jobs[job_id]["status"] = "Merging Video, Audio & Subtitles..."
        jobs[job_id]["progress"] = 90
        
        final_video = concatenate_videoclips(clips, method="compose")
        output_video_path = f"static/outputs/{job_id}_final.mp4"
        final_video.write_videofile(output_video_path, fps=24, codec="libx264", audio_codec="aac")
        
        jobs[job_id]["status"] = "Completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["video_url"] = f"/{output_video_path}"
        
    except Exception as e:
        jobs[job_id]["status"] = f"Error: {str(e)}"
        jobs[job_id]["progress"] = 0

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
async def start_generation(req: VideoRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "Starting...", "progress": 5, "video_url": None, "scenes": []}
    background_tasks.add_task(process_video_pipeline, job_id, req.prompt, req.style, req.language)
    return JSONResponse({"job_id": job_id})

@app.get("/status/{job_id}")
async def check_status(job_id: str):
    return JSONResponse(jobs.get(job_id, {"status": "Not Found"}))

