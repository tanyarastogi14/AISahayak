# app.py
# Streamlit frontend for AI Sahayak (AssemblyAI STT + uploader + manual transcript)
# Flow:
# 1) Upload audio OR paste manual transcript
# 2) Transcribe audio via AssemblyAI
# 3) POST transcript -> /nlu/fill-form
# 4) Receive slots -> prefill editable form
# 5) POST final JSON -> /forms/submit
# 6) GET /forms for history

import streamlit as st
import requests
import json
import io
import os
import time
import re
from pydub import AudioSegment
from datetime import datetime
from datetime import timedelta

# ==============================
# Backend Endpoints
# ==============================
BASE_URL = "http://127.0.0.1:8002"
NLU_ENDPOINT = f"{BASE_URL}/nlu/fill-form"
SUBMIT_ENDPOINT = f"{BASE_URL}/forms/submit"
FORMS_ENDPOINT = f"{BASE_URL}/forms"


# ==============================
# AUDIO CONVERSION (WAV Linear16)
# ==============================
def convert_to_linear16(audio_bytes: bytes, target_sample_rate=16000):
    """Convert any audio format into WAV PCM S16LE (required for stable STT)."""
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_frame_rate(target_sample_rate).set_channels(1).set_sample_width(2)
    out_io = io.BytesIO()
    audio.export(out_io, format="wav")
    return out_io.getvalue()


# ==============================
# ASSEMBLYAI HELPERS
# ==============================
def upload_to_assemblyai(audio_bytes: bytes, api_key: str) -> str:
    """Uploads audio to AssemblyAI & returns upload URL. Longer timeout for network."""
    headers = {"authorization": api_key}
    upload_endpoint = "https://api.assemblyai.com/v2/upload"
    try:
        resp = requests.post(
            upload_endpoint, headers=headers, data=audio_bytes, timeout=120
        )
        resp.raise_for_status()
        return resp.json()["upload_url"]
    except Exception as e:
        # raise readable error for Streamlit to show
        raise RuntimeError(f"AssemblyAI upload failed: {e}")


def transcribe_with_assemblyai(
    upload_url: str, api_key: str, language_code="hi", translate_to: str = None
):
    """
    Start AssemblyAI transcription, poll until completed.
    If translate_to is provided (e.g. "en"), request translation during transcription
    and return the translated text when available. Otherwise return the original transcript.
    """
    headers = {"authorization": api_key, "content-type": "application/json"}
    transcript_endpoint = "https://api.assemblyai.com/v2/transcript"

    # base payload for transcription
    payload = {
        "audio_url": upload_url,
        "punctuate": True,
        "format_text": True,
        "language_code": language_code,
    }

    # If translation requested, add speech_understanding translation block
    if translate_to:
        payload["speech_understanding"] = {
            "request": {
                "translation": {
                    "target_languages": [translate_to],
                    # "formal": True  # optional: choose formal/informal tone
                }
            }
        }

    # create transcription job with retries
    create_attempts = 0
    max_create_attempts = 3
    job_id = None
    while create_attempts < max_create_attempts:
        try:
            create_resp = requests.post(
                transcript_endpoint, json=payload, headers=headers, timeout=30
            )
            create_resp.raise_for_status()
            job_id = create_resp.json().get("id")
            break
        except requests.exceptions.RequestException as e:
            create_attempts += 1
            if create_attempts >= max_create_attempts:
                raise RuntimeError(
                    f"Could not create AssemblyAI transcription job after {create_attempts} attempts: {e}"
                )
            time.sleep(1.5 * create_attempts)  # backoff

    if not job_id:
        raise RuntimeError("Failed to obtain job id from AssemblyAI.")

    poll_url = f"{transcript_endpoint}/{job_id}"
    total_poll_timeout = 240  # seconds (increase if you have long files)
    poll_interval = 3
    elapsed = 0

    while elapsed < total_poll_timeout:
        r = requests.get(poll_url, headers=headers, timeout=15)
        r.raise_for_status()
        job = r.json()
        status = job.get("status")
        if status == "completed":
            # If translation requested and AssemblyAI returned translations, pick it
            if translate_to:
                translated_texts = (
                    job.get("translated_texts", {})
                    or job.get("speech_understanding", {})
                    .get("response", {})
                    .get("translation", {})
                    .get("translated_texts")
                    or {}
                )
                # job may include top-level 'translated_texts' per docs
                en_text = translated_texts.get(translate_to)
                if en_text:
                    return en_text
                # fallback: sometimes `speech_understanding.response` contains translation; check there
                su = job.get("speech_understanding", {}).get("response", {})
                if isinstance(su, dict):
                    # defensively search for translated_texts anywhere
                    t = job.get("translated_texts") or su.get("translated_texts")
                    if t and isinstance(t, dict):
                        maybe = t.get(translate_to)
                        if maybe:
                            return maybe
            # fallback to original transcript
            return job.get("text", "")
        if status == "failed":
            raise RuntimeError(f"AssemblyAI transcription failed: {job}")
        time.sleep(poll_interval)
        elapsed += poll_interval

    raise RuntimeError("AssemblyAI transcription timed out while polling.")


# ==============================
# BACKEND HELPERS
# ==============================
def post_nlu(transcript_text, mock_mode=False):
    if mock_mode:
        return {
            "status": "ok",
            "intent": "fill_form",
            "slots": {
                "form_type": {"value": "Ayushman", "confidence": 0.9},
                "fields": {
                    "name": {"value": "Ramesh"},
                    "age": {"value": 60},
                    "relation": {"value": "father"},
                },
                "original_text": {"value": transcript_text},
            },
        }

    payload = {"text": transcript_text}
    try:
        resp = requests.post(NLU_ENDPOINT, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def submit_form_payload(form_type, fields):
    payload = {"form_type": form_type, "fields": fields}
    try:
        resp = requests.post(SUBMIT_ENDPOINT, json=payload, timeout=10)
        resp.raise_for_status()
        return {"ok": True, "response": resp.json()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_forms_history(mock_mode=False):
    if mock_mode:
        return [
            {
                "submission_id": "1",
                "form_type": "Ayushman",
                "fields": {"name": "Ramesh"},
            },
            {"submission_id": "2", "form_type": "PAN", "fields": {"name": "Sita"}},
        ]

    try:
        resp = requests.get(FORMS_ENDPOINT, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def iso8601_to_seconds(duration_iso: str) -> int:
    if not duration_iso:
        return 0
    pattern = re.compile(r"PT(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?")
    m = pattern.match(duration_iso)
    if not m:
        return 0
    h = int(m.group("h") or 0)
    m_ = int(m.group("m") or 0)
    s = int(m.group("s") or 0)
    return h * 3600 + m_ * 60 + s


# YouTube API search
def search_youtube_api(query: str, api_key: str, max_results=25):
    search_url = "https://www.googleapis.com/youtube/v3/search"
    videos_url = "https://www.googleapis.com/youtube/v3/videos"

    # Step 1 — search for video IDs
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "order": "relevance",
        "relevanceLanguage": "hi",
        "key": api_key,
    }
    r = requests.get(search_url, params=params, timeout=10)
    r.raise_for_status()
    items = r.json().get("items", [])
    video_ids = [i["id"]["videoId"] for i in items if "videoId" in i["id"]]

    if not video_ids:
        return []

    # Step 2 — fetch details (duration + views)
    params2 = {
        "part": "contentDetails,statistics,snippet",
        "id": ",".join(video_ids),
        "key": api_key,
    }
    r2 = requests.get(videos_url, params=params2, timeout=10)
    r2.raise_for_status()

    results = []
    for v in r2.json().get("items", []):
        dur = v["contentDetails"].get("duration")
        views = int(v["statistics"].get("viewCount", 0))
        results.append(
            {
                "id": v["id"],
                "title": v["snippet"]["title"],
                "duration_seconds": iso8601_to_seconds(dur),
                "views": views,
                "url": f"https://www.youtube.com/watch?v={v['id']}",
            }
        )
    return results


# Selection logic
def choose_shortest_popular(videos, max_short_seconds=300):
    if not videos:
        return None
    short = [v for v in videos if v["duration_seconds"] <= max_short_seconds]
    if short:
        return max(short, key=lambda v: (v["views"], -v["duration_seconds"]))
    return sorted(videos, key=lambda v: (v["duration_seconds"], -v["views"]))[0]


# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="AI Sahayak", layout="centered")
st.title("AI Sahayak by Guardians of the Git", text_alignment="center")


# Sidebar
st.sidebar.header("Developer Settings")
mock_mode = st.sidebar.checkbox("Mock mode (no backend needed)", value=False)
show_raw = st.sidebar.checkbox("Show raw JSON", value=False)
auto_send_nlu = st.sidebar.checkbox("Auto-send to NLU", value=True)

# -------------------------------
# 1) INPUT CAPTURE
# -------------------------------
st.header("1) Input Speech (Upload Audio)")

uploaded_file = st.file_uploader(
    "Upload audio file (wav/mp3/m4a/ogg)",
    type=["wav", "mp3", "m4a", "ogg"],
)

# manual_transcript = st.text_area("Or paste text:", height=140)

transcript_text = ""

# if manual_transcript.strip():
#    transcript_text = manual_transcript.strip()

if uploaded_file is not None:
    st.session_state["uploaded_audio"] = uploaded_file.read()
    st.success("Audio uploaded successfully. Click 'Transcribe' to process it.")

# Transcribe button
if st.button("Transcribe"):
    audio_bytes_raw = st.session_state.get("uploaded_audio")

    if not audio_bytes_raw:
        st.error("Please upload an audio file first.")
    else:
        # Convert to WAV Linear16 for AssemblyAI
        try:
            wav_bytes = convert_to_linear16(audio_bytes_raw)
        except Exception as e:
            st.error(f"Audio conversion failed: {e}")
            wav_bytes = None

        if wav_bytes:
            assembly_key = os.environ.get("ASSEMBLYAI_API_KEY")
            if not assembly_key:
                st.error(
                    "AssemblyAI API key not set in ASSEMBLYAI_API_KEY env variable."
                )
            else:
                try:
                    with st.spinner("Uploading to AssemblyAI..."):
                        upload_url = upload_to_assemblyai(wav_bytes, assembly_key)

                    with st.spinner("Transcribing (AssemblyAI)..."):
                        transcript_text = transcribe_with_assemblyai(
                            upload_url,
                            assembly_key,
                            language_code="hi",
                            translate_to="en",
                        )

                    # Save transcript to session state so UI persists and appears later
                    st.session_state["transcript_box"] = transcript_text
                    st.success("Transcription completed!")

                except Exception as e:
                    st.error(f"AssemblyAI Error: {e}")

# Transcript displayed AFTER transcription
transcript_text = st.session_state.get("transcript_box", "")
st.subheader("Transcript (translated to English)")
st.text_area(
    "Transcript:",
    value=transcript_text,
    height=140,
    key="transcript_box",
)


# -------------------------------
# SEND TO NLU
# -------------------------------
if st.button("Send to /nlu/fill-form"):
    if not transcript_text.strip():
        st.error("Transcript is empty!")
    else:
        st.info("Sending to backend NLU...")
        resp = post_nlu(transcript_text, mock_mode)
        st.session_state["nlu"] = resp
        st.rerun()

if auto_send_nlu and transcript_text and "nlu" not in st.session_state:
    st.session_state["nlu"] = post_nlu(transcript_text, mock_mode)


# -------------------------------
# 2) EDIT & CONFIRM FORM FIELDS
# -------------------------------
st.header("2) NLU Output → Form Fill")

nlu_resp = st.session_state.get("nlu", None)

if nlu_resp:
    if show_raw:
        st.json(nlu_resp)

    slots = nlu_resp.get("slots", {})
    form_type_val = slots.get("form_type", {}).get("value", "Ayushman")

    st.subheader("Form Type")
    form_type_val = st.text_input("Form Type", value=form_type_val)

    st.subheader("Fields")
    fields_out = {}

    slot_fields = slots.get("fields", {})
    if isinstance(slot_fields, dict):
        for key, val in slot_fields.items():
            initial = val.get("value") if isinstance(val, dict) else val
            fields_out[key] = st.text_input(key, value=str(initial))
    else:
        st.write("No fields returned — enter manually:")
        fields_out["name"] = st.text_input("name")
        fields_out["age"] = st.text_input("age")
        fields_out["relation"] = st.text_input("relation")

    # SUBMIT FINAL FORM
    if st.button("Submit to /forms/submit"):
        clean_fields = {}
        for k, v in fields_out.items():
            if v.isdigit():
                clean_fields[k] = int(v)
            else:
                clean_fields[k] = v

        result = submit_form_payload(form_type_val, clean_fields)
        if result["ok"]:
            st.success("Form submitted successfully!")
            st.json(result["response"])
            st.session_state.pop("nlu", None)
        else:
            st.error(result["error"])

else:
    st.info("No NLU output yet.")


# -------------------------------
# 3) HISTORY
# -------------------------------
st.header("3) Submission History")

with st.expander("View History"):
    if st.button("Load history"):
        hist = get_forms_history(mock_mode)
        if isinstance(hist, list):
            st.table(hist)
        else:
            st.json(hist)

# -------------------------------
# 4) YouTube Integration
# -------------------------------

st.markdown("---")
st.header("FinEd — Learn Finance with Short Hindi Videos")

# Input
fin_audio = st.file_uploader(
    "Upload Hindi audio", type=["wav", "mp3", "m4a", "ogg"], key="fin_audio"
)
fin_manual = st.text_input("Or type your question in Hindi:", key="fin_manual")

max_results = st.slider("YouTube results to scan", 10, 50, 25)
max_short_seconds = st.number_input("Max short video length (sec)", 30, 600, 300)

if st.button("Find Video"):
    query = fin_manual.strip()

    # If no manual text, use audio transcription
    if not query and fin_audio:
        st.info("Transcribing audio via AssemblyAI...")
        wav_bytes = convert_to_linear16(fin_audio.read())
        assembly_key = os.environ.get("ASSEMBLYAI_API_KEY")
        upload_url = upload_to_assemblyai(wav_bytes, assembly_key)
        query = transcribe_with_assemblyai(
            upload_url, assembly_key, language_code="hi", translate_to="en"
        )
        st.success(f"Detected query: {query}")

    if not query:
        st.error("Provide text or audio.")
        st.stop()

    # YouTube search
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        st.error("Missing YOUTUBE_API_KEY environment variable.")
        st.stop()

    st.info("Searching YouTube...")
    videos = search_youtube_api(
        query + "banking india", api_key, max_results=max_results
    )

    if not videos:
        st.error("No relevant videos found.")
        st.stop()

    chosen = choose_shortest_popular(videos, max_short_seconds=max_short_seconds)

    st.subheader("Recommended Video")
    st.write("**Title:**", chosen["title"])
    st.write("**Duration:**", str(timedelta(seconds=chosen["duration_seconds"])))
    st.write("**Views:**", chosen["views"])
    st.video(chosen["url"])

# Footer
st.markdown("---")
st.caption("AI Sahayak")
