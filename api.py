# api.py (overwrite your existing file with this)

import io
import base64
import uuid
import os
from gtts import gTTS
from fastapi import Query

import json

from form_config import get_form_definition
from form_filler import generate_form49a_pdf


from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import Dict, Any, List
from datetime import datetime

from schemas import (
    NLURequest,
    NLUResponse,
    SubmitFormRequest,
    Submission,
    SubmissionsList,
)

# repository functions (DB-backed)
from repository import save_submission, get_submissions

# ---------- Gemini + env imports ----------
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env")

genai.configure(api_key=GEMINI_API_KEY)
print("USING GEMINI KEY PREFIX:", GEMINI_API_KEY[:8] + "********")


# ---------- FastAPI router ----------
router = APIRouter()


from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print("Loaded GEMINI_API_KEY:", api_key[:8] + "********")  # show only first 8 chars

genai.configure(api_key=api_key)


# ===================== 1) NLU / fill-form endpoint (Gemini) =====================
@router.post("/nlu/fill-form", response_model=NLUResponse)
async def fill_form(body: NLURequest) -> NLUResponse:
    """
    Uses Gemini to:
    - Understand the user's text (from STT or manual input)
    - Return:
        intent: short string label
        slots: dict with extracted fields
    Responds in the same JSON shape the frontend already expects.
    """
    text = body.text

    system_prompt = """
    You are an NLU engine for Indian government / ID / welfare forms.
    Your job is to:
      1. Understand what the user wants.
      2. Extract:
         - intent: short label like "fill_ayushman_form", "fill_pan_form",
                   "fill_ration_card_form", "general_query"
         - slots: JSON object with key-value pairs you can infer, such as:
             name, age, relation, scheme, location, etc.

    You MUST respond with ONLY valid JSON in this exact format:
    {
      "intent": "<string>",
      "slots": { ... }
    }

    Do NOT wrap it in markdown, code fences, or add explanations.
    Only JSON.
    """

    full_prompt = system_prompt + f'\n\nUser text: "{text}"'

    try:
        #model = genai.GenerativeModel("gemini-2.5-pro")  # chosen from available models
        model = genai.GenerativeModel("gemini-flash-latest")
       
        response = model.generate_content(full_prompt)

        # Gemini's text output
        raw_text = (response.text or "").strip()
        print("GEMINI RAW:", raw_text)

        # Sometimes models return JSON inside ``` fences – strip them if present
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            # If there's a language tag like "json\n", drop the first line
            if "\n" in raw_text:
                first_line, rest = raw_text.split("\n", 1)
                raw_text = rest.strip()

        # Try direct JSON parse
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            # Fallback: try to grab first {...} block
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1:
                json_str = raw_text[start:end + 1]
                data = json.loads(json_str)
            else:
                raise

        intent = data.get("intent", "unknown")
        slots = data.get("slots", {})

        # Ensure slots is a dict
        if not isinstance(slots, dict):
            slots = {"value": slots}

        return NLUResponse(intent=intent, slots=slots)

    except Exception as e:
        # Surface Gemini error nicely to frontend/docs
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")

#new endpoint to download/show filled PDF


@router.get("/forms/{submission_id}/pdf")
async def get_filled_form(submission_id: int):
    """
    Returns the filled PDF for a given submission.
    For hackathon demo: regenerate PDF from stored fields.
    """
    rows = get_submissions(limit=1000)  # simple; or write get_submission_by_id()
    row = next((r for r in rows if r["id"] == submission_id), None)
    if not row:
        raise HTTPException(status_code=404, detail="Submission not found")

    form_type = row.get("form_type")
    fields = row.get("fields", {})

    if form_type != "form49a":
        raise HTTPException(status_code=400, detail="PDF generation not implemented for this form type")

    pdf_path = generate_form49a_pdf(fields)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"submission_{submission_id}.pdf",
    )


@router.get("/forms/definition")
async def get_form_definition_endpoint(form_type: str = Query(...)):
    """
    Returns definition of a form type, e.g. which fields Form49A needs.
    """
    form_def = get_form_definition(form_type)
    if not form_def:
        raise HTTPException(status_code=404, detail="Unknown form_type")
    return form_def



# ===================== 2) Submit form endpoint (DB-backed) =====================
@router.post("/forms/submit")
async def submit_form(body: SubmitFormRequest):
    """
    1. Save the submission to DB.
    2. If it's form49a, generate filled PDF.
    """
    try:
        new_id = save_submission(body.form_type, body.fields)

        # Only Form49A supported for now
        if body.form_type == "form49a":
            pdf_path = generate_form49a_pdf(body.fields)
            # OPTIONAL: store pdf_path in DB if you add a column for it

        return {"id": new_id}
    except Exception as e:
        print("DB save error:", e)
        raise HTTPException(status_code=500, detail=f"DB error: {e}")




# ===================== 3) List forms endpoint (DB-backed) =====================
@router.get("/forms", response_model=SubmissionsList)
async def list_forms(limit: int = 50) -> SubmissionsList:
    """
    Return recent submissions using repository.get_submissions(limit).
    Converts DB dict rows into Pydantic Submission objects.
    """
    try:
        rows = get_submissions(limit=limit)  # returns list[dict] with id, form_type, fields, optional created_at
    except Exception as e:
        print("DB list error:", e)
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    items: List[Submission] = []
    for r in rows:
        created_at = r.get("created_at")
        if created_at:
            # if it's a string, try to parse ISO format
            if isinstance(created_at, str):
                try:
                    created_dt = datetime.fromisoformat(created_at)
                except Exception:
                    created_dt = datetime.now()
            elif isinstance(created_at, datetime):
                created_dt = created_at
            else:
                created_dt = datetime.now()
        else:
            created_dt = datetime.now()

        items.append(
            Submission(
                id=r["id"],
                form_type=r.get("form_type", ""),
                fields=r.get("fields", {}),
                created_at=created_dt,
            )
        )

    return SubmissionsList(items=items)


# ===================== 4) NLU + TTS endpoint =====================
@router.post("/nlu/fill-form-audio")
async def fill_form_audio(body: NLURequest, language: str = "hi"):
    """
    1. Uses the existing NLU to extract intent + slots.
    2. Builds a confirmation sentence in the chosen language.
    3. Converts that sentence to audio (MP3) using gTTS.
    4. Returns the audio file as FileResponse.
    """
    # 1) Reuse your existing NLU endpoint logic
    nlu_result = await fill_form(body)  # calls the Gemini-based fill_form

    # nlu_result is a Pydantic NLUResponse
    intent = nlu_result.intent
    slots = nlu_result.slots

    # 2) Build a confirmation sentence (example in Hindi)
    name = slots.get("name")
    age = slots.get("age")
    relation = slots.get("relation")
    scheme = slots.get("scheme") or slots.get("yojana") or slots.get("scheme_name")

    parts = []

    if scheme:
        parts.append(f"आप {scheme} के लिए आवेदन कर रहे हैं.")
    if name:
        parts.append(f"आवेदक का नाम {name} है.")
    if relation:
        parts.append(f"यह आपके {relation} के लिए है.")
    if age:
        parts.append(f"उम्र {age} वर्ष है.")

    # Fallback if we didn't extract much
    if not parts:
        original_text = slots.get("original_text", body.text)
        parts.append(f"आपने कहा था: {original_text}")

    parts.append("क्या यह जानकारी सही है?")

    confirmation_text = " ".join(parts)
    print("CONFIRMATION TEXT:", confirmation_text)

    # 3) Generate speech audio using gTTS
    try:
        os.makedirs("audio_cache", exist_ok=True)
        filename = f"confirm_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join("audio_cache", filename)

        tts = gTTS(text=confirmation_text, lang=language or "hi")
        tts.save(filepath)

        # 4) Return the audio file to the frontend
        return FileResponse(
            filepath,
            media_type="audio/mpeg",
            filename=filename,
        )
    except Exception as e:
        print("TTS error:", e)
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")