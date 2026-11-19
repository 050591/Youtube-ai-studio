"""
Password-Protected Mobile Web App for the Multi-Modal YouTube AI Agent
======================================================================
Run locally or deploy to Streamlit Community Cloud / Render / Railway:
  pip install -r requirements.txt
  streamlit run Mobile_Web_App.py
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from youtube_ai_agent import build_agent

load_dotenv()

st.set_page_config(
    page_title="YouTube AI Studio",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def check_password() -> bool:
    """Simple timing-safe password gate for mobile browser access."""
    expected_pw = os.getenv("APP_PASSWORD", "tamilai2026")

    if st.session_state.get("authenticated"):
        return True

    st.title("🎬 YouTube AI ஏஜென்ட்")
    st.caption("பாதுகாப்பான மொபைல் உள்நுழைவு (Mobile Login)")

    with st.form("login_form"):
        entered_pw = st.text_input("கடவுச்சொல் (Password)", type="password")
        submitted = st.form_submit_button("உள்நுழைக (Login)", use_container_width=True)

    if submitted:
        if hmac.compare_digest(entered_pw, expected_pw):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("தவறான கடவுச்சொல். மீண்டும் முயற்சிக்கவும்.")

    return False


if not check_password():
    st.stop()


# ---------------------------------------------------------------------------
# Authenticated Mobile Dashboard
# ---------------------------------------------------------------------------
col_head, col_out = st.columns([4, 1])
with col_head:
    st.title("🎬 YouTube AI Studio")
with col_out:
    if st.button("Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

st.write("ஒரே பிராம்ட் மூலம் வைரல் ஹூக், 2D/3D காட்சி, பாடல் மற்றும் இலவச MP4 வீடியோவை உருவாக்குங்கள்.")

with st.form("agent_form"):
    user_prompt = st.text_area(
        "உங்கள் வீடியோ தலைப்பு / பிராம்ட் (Tamil or English)",
        placeholder="எ.கா. Cute 3D Squirrel Tamil Kids Balloon Song",
        height=90,
    )
    content_format = st.radio(
        "வீடியோ வகை",
        options=["shorts", "long_form"],
        format_func=lambda x: "YouTube Shorts (9:16)" if x == "shorts" else "Long-Form (16:9)",
        horizontal=True,
    )
    use_live = st.toggle("Live APIs பயன்படுத்துக (.env விசைகள் தேவை)", value=False)
    make_video = st.toggle("🎬 இலவச MP4 வீடியோ உருவாக்குக (Auto-Motion)", value=True)
    run_btn = st.form_submit_button("ஏஜென்ட்டை இயக்கு (Generate)", use_container_width=True)

if run_btn:
    if not user_prompt.strip():
        st.warning("தயவுசெய்து ஒரு தலைப்பு அல்லது பிராம்ட்டை உள்ளிடவும்.")
        st.stop()

    with st.spinner("AI ஏஜென்ட் ஆய்வு, 2D/3D, இசை மற்றும் MP4 வீடியோவை உருவாக்குகிறது..."):
        agent = build_agent()
        result = agent.invoke(
            {
                "user_prompt": user_prompt.strip(),
                "content_format": content_format,
                "dry_run": not use_live,
                "generate_video": make_video,
                "trending_videos": [],
                "hook_package": {},
                "image_2d_path": "",
                "model_3d_task_id": "",
                "music_audio_path": "",
                "video_mp4_path": "",
                "warnings": [],
            }
        )

    for warn in result.get("warnings", []):
        st.warning(warn)

    pkg = result.get("hook_package", {})
    st.success("உருவாக்கம் முடிந்தது!")

    st.subheader("🔥 வைரல் தலைப்புகள் & 3-வினாடி ஹூக்")
    for idx, title in enumerate(pkg.get("viral_titles", []), start=1):
        st.markdown(f"**{idx}.** {title}")

    st.info(f"**முதல் 3 வினாடி ஹூக்:** {pkg.get('first_3s_hook', '')}")
    st.caption(f"**தக்கவைப்பு உத்தி (Loop):** {pkg.get('retention_loop', '')}")

    st.subheader("🎬 இலவச MP4 வீடியோ (Auto-Motion)")
    vid_path = Path(result.get("video_mp4_path", ""))
    if vid_path.exists() and vid_path.stat().st_size > 0:
        vid_bytes = vid_path.read_bytes()
        st.video(vid_bytes, format="video/mp4")
        st.download_button(
            "MP4 வீடியோவைப் பதிவிறக்குக",
            data=vid_bytes,
            file_name=vid_path.name,
            mime="video/mp4",
            use_container_width=True,
        )
    else:
        st.info("MP4 வீடியோ உருவாக்கப்படவில்லை.")

    st.subheader("🎨 2D & 3D காட்சித் தகவல்கள்")
    st.text_area("2D Image Prompt", value=pkg.get("prompt_2d", ""), height=80)
    img_path = Path(result.get("image_2d_path", ""))
    if img_path.exists():
        img_bytes = img_path.read_bytes()
        st.image(img_bytes, caption="உருவாக்கப்பட்ட 2D காட்சி", use_container_width=True)
        st.download_button(
            "2D படத்தைப் பதிவிறக்குக",
            data=img_bytes,
            file_name=img_path.name,
            mime="image/png",
            use_container_width=True,
        )

    st.code(f"3D Prompt: {pkg.get('prompt_3d', '')}\nTripo 3D Task ID: {result.get('model_3d_task_id', '')}")

    st.subheader("🎵 பாடல் & இசை (Suno-Style)")
    st.text_area("Music Prompt", value=pkg.get("prompt_music", ""), height=70)
    audio_path = Path(result.get("music_audio_path", ""))
    if audio_path.exists():
        audio_bytes = audio_path.read_bytes()
        st.audio(audio_bytes, format="audio/mp3")
        st.download_button(
            "பாடலைப் பதிவிறக்குக (MP3)",
            data=audio_bytes,
            file_name=audio_path.name,
            mime="audio/mpeg",
            use_container_width=True,
        )
