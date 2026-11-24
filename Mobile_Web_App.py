"""
YouTube AI Studio — Mobile Web App (Streamlit)
==============================================
Password-protected mobile UI for generating prompt-to-video YouTube Shorts
with autonomous character voiceovers and background music.
"""

import os
import streamlit as st
from youtube_ai_agent import run_youtube_agent

st.set_page_config(page_title="YouTube AI Studio", page_icon="🎬", layout="centered")

# Load Streamlit Cloud Secrets into environment variables
for key in ["APP_PASSWORD", "GEMINI_API_KEY", "YOUTUBE_API_KEY", "HF_TOKEN", "ELEVENLABS_API_KEY"]:
    if key in st.secrets and st.secrets[key]:
        os.environ[key] = str(st.secrets[key])

# Simple Password Gate
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

app_pw = os.getenv("APP_PASSWORD", "")
if app_pw and not st.session_state.authenticated:
    st.title("🎬 YouTube AI Studio")
    entered = st.text_input("Password", type="password")
    if st.button("Unlock", use_container_width=True):
        if entered == app_pw:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

st.title("🎬 YouTube AI Studio")
st.caption("Prompt → AI Video + Character Voices + Background Music")

prompt = st.text_area(
    "உங்கள் வீடியோ ப்ராம்ட் (Video Prompt)",
    placeholder="உதா: இரண்டு நண்பர்கள் AI பற்றி பேசும் ஒரு நகைச்சுவை Tamil Short...",
    height=110,
)

col1, col2 = st.columns(2)
with col1:
    language = st.selectbox("குரல் மொழி (Voice Language)", ["Tamil", "English"])
with col2:
    dry_run = st.checkbox("Dry Run (Test Mode)", value=False)

if st.button("🚀 Generate Full Video", type="primary", use_container_width=True):
    if not prompt.strip():
        st.warning("தயவுசெய்து ஒரு ப்ராம்ட் உள்ளிடவும்.")
        st.stop()

    with st.spinner("AI வீடியோ, கேரக்டர் குரல்கள் மற்றும் இசை உருவாகிறது..."):
        result = run_youtube_agent(
            user_prompt=prompt.strip(),
            language=language,
            content_format="shorts",
            dry_run=dry_run,
        )

    storyboard = result.get("storyboard", {})
    st.subheader(storyboard.get("title", "Generated Short"))
    st.write(storyboard.get("description", ""))

    video_path = result.get("final_video_path", "")
    if video_path and os.path.exists(video_path):
        st.video(video_path)
        with open(video_path, "rb") as f:
            st.download_button(
                "⬇️ Download MP4",
                data=f,
                file_name="youtube_ai_short.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

    with st.expander("🎭 Character Script & Logs"):
        for scene in result.get("scene_assets", []):
            st.markdown(
                f"**Scene {scene['scene_number']} — {scene.get('character_name', 'Character')} "
                f"({scene.get('voice_role', '')})**: {scene['dialogue']}"
            )
        for line in result.get("logs", []):
            st.text(line)
