# app_all_in_one.py
"""
Single-file Streamlit app that acts as frontend + local backend (SQLite).
Drop into your project root and run with:

    # Windows
    venv\Scripts\activate

    # macOS / Linux
    source venv/bin/activate

    streamlit run app_all_in_one.py

Notes:
 - Uses SQLite file aisahayak_local.db in the same folder.
 - NLU is mocked by default; you can paste your own NLU JSON in the UI.
 - AssemblyAI/Youtube features are optional and require keys set in env or cmd.
"""

import streamlit as st
import requests
import json
import io
import os
import time
import re
from form_filler import generate_pension_form_pdf   # <-- CHANGED
from form_config import get_form_definition
from datetime import timedelta
from pydub import AudioSegment
from gtts import gTTS


# SQLAlchemy for local storage
from sqlalchemy import (
    Column,
    Integer,
    String,
    TIMESTAMP,
    JSON as SQLJSON,
    create_engine,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# dotenv to read .env (optional)
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Local DB (SQLite) setup
# ---------------------------
BASE = declarative_base()
DB_FILE = "aisahayak_local.db"
DATABASE_URL = os.getenv("DATABASE_URL_LOCAL", f"sqlite:///{DB_FILE}")

# Create engine with sqlite-specific args if needed
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class FormSubmission(BASE):
    __tablename__ = "form_submissions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    form_type = Column(String(200), nullable=True)
    fields = Column(SQLJSON, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())


# create tables on startup
BASE.metadata.create_all(bind=engine)


def save_submission_local(form_type: str, fields: dict) -> int:
    db = SessionLocal()
    try:
        obj = FormSubmission(form_type=form_type, fields=fields)
        db.add(obj)
        db.commit()
        db.refresh(obj)
        return obj.id
    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()


def get_submissions_local(limit: int = 50):
    db = SessionLocal()
    try:
        rows = (
            db.query(FormSubmission)
            .order_by(FormSubmission.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "form_type": r.form_type,
                "fields": r.fields,
                "created_at": str(r.created_at),
            }
            for r in rows
        ]
    finally:
        db.close()


def get_latest_submission():
    db = SessionLocal()
    try:
        row = (
            db.query(FormSubmission).order_by(FormSubmission.created_at.desc()).first()
        )
        if not row:
            return None
        return {
            "id": row.id,
            "form_type": row.form_type,
            "fields": row.fields or {},
            "created_at": str(row.created_at),
        }
    finally:
        db.close()


def build_confirmation_text(submission: dict, language: str = "hi") -> str:
    """
    Build a Hindi confirmation sentence from a DB submission row.
    It reads out *all* key–value pairs stored in fields.
    """
    form_type = submission.get("form_type") or ""
    fields = submission.get("fields") or {}

    parts = []

    # Use form_type as scheme name if present
    if form_type:
        parts.append(f"आप {form_type} के लिए आवेदन कर रहे हैं.")

    if fields:
        parts.append("आपने जो जानकारी दी है, वह इस प्रकार है:")

        # Optional: nicer Hindi labels for common fields
        label_map = {
            "name": "नाम",
            "full_name": "पूरा नाम",
            "applicant_name": "आवेदक का नाम",
            "father_name": "पिता का नाम",
            "mother_name": "माता का नाम",
            "age": "उम्र",
            "address": "पता",
        }

        for key, value in fields.items():
            # Pick Hindi label if we know it, else humanize the key
            label_hi = label_map.get(key)
            if not label_hi:
                # replace underscores with spaces for readability
                label_hi = key.replace("_", " ")
            parts.append(f"{label_hi} {value} है.")
    else:
        parts.append("अभी तक कोई भी जानकारी नहीं मिली है.")

    parts.append("क्या यह जानकारी सही है?")

    return " ".join(parts)


def tts_to_bytes(text: str, language: str = "hi") -> bytes:
    """
    Generate MP3 audio for given text and return it as bytes.
    """
    buf = io.BytesIO()
    tts = gTTS(text=text, lang=language or "hi")
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ---------------------------
# AssemblyAI helpers (optional)
# ---------------------------
def convert_to_linear16(audio_bytes: bytes, target_sample_rate=16000):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_frame_rate(target_sample_rate).set_channels(1).set_sample_width(2)
    out_io = io.BytesIO()
    audio.export(out_io, format="wav")
    return out_io.getvalue()


def upload_to_assemblyai(audio_bytes: bytes, api_key: str) -> str:
    headers = {"authorization": api_key}
    upload_endpoint = "https://api.assemblyai.com/v2/upload"
    resp = requests.post(
        upload_endpoint, headers=headers, data=audio_bytes, timeout=120
    )
    resp.raise_for_status()
    return resp.json()["upload_url"]


def transcribe_with_assemblyai(
    upload_url: str, api_key: str, language_code="hi", translate_to: str = None
):
    headers = {"authorization": api_key, "content-type": "application/json"}
    transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
    payload = {
        "audio_url": upload_url,
        "punctuate": True,
        "format_text": True,
        "language_code": language_code,
    }
    if translate_to:
        payload["speech_understanding"] = {
            "request": {"translation": {"target_languages": [translate_to]}}
        }

    create_resp = requests.post(
        transcript_endpoint, json=payload, headers=headers, timeout=30
    )
    create_resp.raise_for_status()
    job_id = create_resp.json().get("id")
    poll_url = f"{transcript_endpoint}/{job_id}"
    poll_interval, elapsed, total_timeout = 3, 0, 240
    while elapsed < total_timeout:
        r = requests.get(poll_url, headers=headers, timeout=15)
        r.raise_for_status()
        job = r.json()
        status = job.get("status")
        if status == "completed":
            if translate_to:
                # Try to find translated text defensively
                translated = job.get("translated_texts") or job.get(
                    "speech_understanding", {}
                ).get("response", {}).get("translation", {}).get("translated_texts")
                if isinstance(translated, dict) and translate_to in translated:
                    return translated[translate_to]
            return job.get("text", "")
        if status == "failed":
            raise RuntimeError(f"AssemblyAI transcription failed: {job}")
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise RuntimeError("Transcription timed out.")


# ---------------------------
# NLU (mock + optional call)
# ---------------------------
# REAL GEMINI NLU + FALLBACK
# ---------------------------

import google.generativeai as genai

GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)


def gemini_nlu(transcript_text: str) -> dict:
    """
    Call Gemini to parse NLU slots from the transcript text.
    Returns a Python dict parsed from the JSON response.
    """
    model = genai.GenerativeModel("gemini-flash-latest")

    prompt = f"""
You are an NLU engine for an Indian government-form assistant.

Return JSON in EXACTLY this format:

{{
  "intent": "<string>",
  "slots": {{
    "form_type": {{ "value": "<form name>" }},
    "fields": {{
      "<field_name>": {{ "value": "<extracted value>" }}
    }}
  }}
}}

Rules:
- Every field must be inside "fields"
- Every value must be inside {{ "value": ... }}
- Return ONLY valid JSON
- No explanation text

Now extract intent and slots for this message:

"{transcript_text}"
"""

    response = model.generate_content(prompt)

    # Gemini sometimes wraps JSON in backticks or text, so strip and parse
    raw = response.text.strip()

    # Try to find JSON inside, if needed you can make this more robust
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # very naive fallback: try to extract the first {...} block
        import re

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"Gemini response not valid JSON:\n{raw}")
        data = json.loads(match.group(0))

    return data


# ✅ SAFE MOCK FALLBACK
def mock_nlu(transcript_text: str):
    return {
        "status": "ok",
        "intent": "fill_form",
        "slots": {
            "form_type": {"value": "Ayushman"},
            "fields": {"name": "Ramesh", "age": 60, "father name": "Rajesh"},
            "original_text": {"value": transcript_text},
        },
    }


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Sahayak by Guardians of the Git", layout="centered")
st.title("AI Sahayak — Local All-in-One (SQLite)")

# Sidebar
st.sidebar.header("Settings")
use_local_backend = st.sidebar.checkbox("Use local backend (SQLite)", value=True)
use_mock_nlu = st.sidebar.checkbox("Use mock NLU (no external calls)", value=True)
show_raw = st.sidebar.checkbox("Show raw JSON", value=False)
auto_send_nlu = st.sidebar.checkbox("Auto-send transcript to NLU", value=True)

# AssemblyAI / YouTube API keys (optional, set in env or set here)
assembly_key = os.environ.get("ASSEMBLYAI_API_KEY") or st.sidebar.text_input(
    "ASSEMBLYAI API KEY (optional)", value="", type="password"
)
youtube_key = os.environ.get("YOUTUBE_API_KEY") or st.sidebar.text_input(
    "YOUTUBE API KEY (optional)", value=""
)

# 1) Input capture
st.header("1) Input Speech (Upload Audio or paste transcript)")
uploaded_file = st.file_uploader(
    "Upload audio file (wav/mp3/m4a/ogg)", type=["wav", "mp3", "m4a", "ogg"]
)
transcript_box = st.text_area("Or paste transcript / manual input here:", height=140)

transcript_text = transcript_box.strip()

if uploaded_file is not None:
    st.session_state["uploaded_audio"] = uploaded_file.read()
    st.success("Audio uploaded. Click Transcribe to process.")

if st.button("Transcribe (AssemblyAI)"):
    audio_bytes_raw = st.session_state.get("uploaded_audio")
    if not audio_bytes_raw:
        st.error("Upload audio first.")
    else:
        if not assembly_key:
            st.error("AssemblyAI key not set. Either set it in sidebar or use mock.")
        else:
            try:
                wav_bytes = convert_to_linear16(audio_bytes_raw)
                upload_url = upload_to_assemblyai(wav_bytes, assembly_key)
                trans = transcribe_with_assemblyai(
                    upload_url, assembly_key, language_code="hi", translate_to="en"
                )
                st.session_state["transcript_box"] = trans
                transcript_text = trans
                st.success("Transcription complete.")
            except Exception as e:
                st.error(f"Transcription error: {e}")

# prefer session state transcript if present
transcript_text = st.session_state.get("transcript_box", transcript_text)

st.subheader("Transcript (editable)")
transcript_text = st.text_area(
    "Transcript", value=transcript_text, height=140, key="transcript_box"
)

# 2) Send to NLU (mock or optional external)
st.header("2) NLU → Prefill form")
if st.button("Send to NLU"):
    if not transcript_text.strip():
        st.error("Transcript is empty.")
    else:
        if use_mock_nlu:
            nlu_resp = mock_nlu(transcript_text)
        else:
            with st.spinner("Extracting fields using Gemini..."):
                nlu_resp = gemini_nlu(transcript_text)
        st.session_state["nlu"] = nlu_resp
        st.rerun()

if auto_send_nlu and transcript_text and "nlu" not in st.session_state:
    if use_mock_nlu:
        st.session_state["nlu"] = mock_nlu(transcript_text)
    else:
        st.session_state["nlu"] = gemini_nlu(transcript_text)

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
    if slot_fields == {} and isinstance(slots, dict):
        slot_fields = {
            k: {"value": v} for k, v in slots.items() if k not in ["form_type"]
        }
    if isinstance(slot_fields, dict):
        for key, val in slot_fields.items():
            initial = val.get("value") if isinstance(val, dict) else val
            fields_out[key] = st.text_input(key, value=str(initial))
    else:
        st.write("No fields returned — enter manually:")
        fields_out["name"] = st.text_input("name")
        fields_out["age"] = st.text_input("age")
        fields_out["father name"] = st.text_input("father name")

    if st.button("Submit to local DB"):
        # sanitize fields (convert digits)
        clean_fields = {}
        for k, v in fields_out.items():
            if isinstance(v, str) and v.isdigit():
                clean_fields[k] = int(v)
            else:
                clean_fields[k] = v
        try:
            new_id = save_submission_local(form_type_val, clean_fields)
            st.success(f"Saved submission id: {new_id}")
            st.session_state["last_submission_id"] = new_id  # 👈 remember this
            st.session_state.pop("nlu", None)
        except Exception as e:
            st.error(f"Save failed: {e}")

else:
    st.info("No NLU output yet.")

# 3) History
st.header("3) Submission History")
with st.expander("View history"):
    if st.button("Load history"):
        try:
            hist = get_submissions_local()
            st.table(hist)
        except Exception as e:
            st.error(f"Could not load history: {e}")

# 4) Confirmation (TTS from latest submission)
st.header("4) Confirmation")

if st.button("Generate confirmation audio from latest submission"):
    latest = get_latest_submission()
    if not latest:
        st.error("No submissions found in the database yet.")
    else:
        try:
            confirm_text = build_confirmation_text(latest, language="hi")
            audio_bytes = tts_to_bytes(confirm_text, language="hi")

            # store in session so it persists across reruns
            st.session_state["confirm_text"] = confirm_text
            st.session_state["confirm_audio"] = audio_bytes
            st.session_state["confirmed"] = False
        except Exception as e:
            st.error(f"TTS error: {e}")

# If we already generated confirmation, show it
confirm_text = st.session_state.get("confirm_text")
confirm_audio = st.session_state.get("confirm_audio")

if confirm_text and confirm_audio:
    st.write("**Confirmation text:**")
    st.write(confirm_text)
    st.audio(confirm_audio, format="audio/mp3")

    if st.button("Yes, move to form filling"):
        st.session_state["confirmed"] = True
        st.success("Confirmation accepted. Proceed to Step 5 — Form Filling below.")

# 5) Form Filling (New Pension Scheme – Annexure-I)
st.header("5) Form Filling (New Pension Scheme – Annexure-I)")

if st.session_state.get("confirmed"):
    # Use the most recent submission (same as confirmation step)
    latest = get_latest_submission()
    if not latest:
        st.error("No submission found to fill the form.")
    else:
        form_type = (latest["form_type"] or "").lower()

        # Require the correct form_type for NPS Annexure-I
        if form_type not in [
            "nps_annexure1",
            "new pension scheme",
            "new pension scheme annexure-i",
            "nps form",
        ]:
            st.warning(
                "Latest submission is not for the New Pension Scheme (Annexure-I). "
                "Please set 'Form Type' to 'nps_annexure1' in Step 2 before saving."
            )
        else:
            # Normalize aliases to the canonical key used in form_config
            form_key = form_type
            if form_key in [
                "new pension scheme",
                "new pension scheme annexure-i",
                "nps form",
            ]:
                form_key = "nps_annexure1"

            # Load form definition (field names + labels)
            form_def = get_form_definition(form_key)
            if not form_def:
                st.error("Form definition for 'nps_annexure1' not found.")
            else:
                st.subheader(form_def["human_name"])

                original_fields = latest["fields"] or {}
                filled_fields = {}

                st.markdown(
                    "### Confirm / edit details for New Pension Scheme (Annexure-I)"
                )

                for field in form_def["required_fields"]:
                    key = field["key"]
                    label = field["label"]
                    initial = str(original_fields.get(key, ""))
                    filled_fields[key] = st.text_input(label, value=initial)

                if st.button("Generate filled NPS Annexure-I Form (PDF)"):
                    try:
                        pdf_path = generate_pension_form_pdf(filled_fields)
                        # Read PDF bytes so Streamlit can offer a download
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()

                        st.success(
                            "Filled New Pension Scheme Annexure-I form generated successfully."
                        )
                        st.download_button(
                            label="Download filled Annexure-I",
                            data=pdf_bytes,
                            file_name="nps_annexure1_filled.pdf",
                            mime="application/pdf",
                        )
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
else:
    st.warning(
        "Please generate and confirm the audio in Step 4 before filling the form."
    )

# 6) Optional: YouTube quick search (FinEd)
st.markdown("---")
st.header("FinEd — Find short Hindi videos (optional)")

fin_audio = st.file_uploader(
    "Upload Hindi audio (for query)", type=["wav", "mp3", "m4a", "ogg"], key="fin_audio"
)
fin_manual = st.text_input("Or type your question in Hindi:", key="fin_manual")
max_results = st.slider("YouTube results to scan", 10, 50, 25)
max_short_seconds = st.number_input("Max short video length (sec)", 30, 600, 300)


def iso8601_to_seconds(duration_iso: str) -> int:
    if not duration_iso:
        return 0
    pattern = re.compile(r"PT(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?")
    m = pattern.match(duration_iso)
    if not m:
        return 0
    h = int(m.group("h") or 0)
    mm = int(m.group("m") or 0)
    s = int(m.group("s") or 0)
    return h * 3600 + mm * 60 + s


def search_youtube_api(query: str, api_key: str, max_results=25):
    search_url = "https://www.googleapis.com/youtube/v3/search"
    videos_url = "https://www.googleapis.com/youtube/v3/videos"
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


def choose_shortest_popular(videos, max_short_seconds=300):
    if not videos:
        return None
    short = [v for v in videos if v["duration_seconds"] <= max_short_seconds]
    if short:
        return max(short, key=lambda v: (v["views"], -v["duration_seconds"]))
    return sorted(videos, key=lambda v: (v["duration_seconds"], -v["views"]))[0]


if st.button("Find Video"):
    query = fin_manual.strip()
    if not query and fin_audio:
        st.info("Transcribing audio via AssemblyAI...")
        try:
            wav_bytes = convert_to_linear16(fin_audio.read())
            if not assembly_key:
                st.error("Set ASSEMBLYAI_API_KEY in sidebar or env to transcribe.")
            else:
                upload_url = upload_to_assemblyai(wav_bytes, assembly_key)
                query = transcribe_with_assemblyai(
                    upload_url, assembly_key, language_code="hi", translate_to="en"
                )
                st.success(f"Detected query: {query}")
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            st.stop()
    if not query:
        st.error("Provide text or audio.")
        st.stop()
    if not youtube_key:
        st.error("Set YOUTUBE_API_KEY in sidebar to search YouTube.")
        st.stop()
    st.info("Searching YouTube...")
    try:
        videos = search_youtube_api(
            query + " banking india", youtube_key, max_results=max_results
        )
    except Exception as e:
        st.error(f"YouTube search failed: {e}")
        st.stop()
    if not videos:
        st.error("No relevant videos found.")
        st.stop()
    chosen = choose_shortest_popular(videos, max_short_seconds=max_short_seconds)
    st.subheader("Recommended Video")
    st.write("**Title:**", chosen["title"])
    st.write("**Duration:**", str(timedelta(seconds=chosen["duration_seconds"])))
    st.write("**Views:**", chosen["views"])
    st.video(chosen["url"])

st.markdown("---")
st.caption("AI Sahayak — Local all-in-one (SQLite)")
