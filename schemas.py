from pydantic import BaseModel , ConfigDict
from typing import Dict, Any, List


# ---------- 1) NLU / GPT endpoint ----------

class NLURequest(BaseModel):
    text: str  # text from STT or user


class NLUResponse(BaseModel):
    intent: str              # e.g., "fill_pan_form"
    slots: Dict[str, Any]    # extracted fields (name, age, etc.)


# ---------- 2) Form submission endpoints ----------

class SubmitFormRequest(BaseModel):
    form_type: str           # e.g., "Ayushman", "RationCard"
    fields: Dict[str, Any]   # the filled form fields as JSON


class Submission(BaseModel):
    id: int
    form_type: str
    fields: Dict[str, Any]

    #pydantic v2 style 
    model_config = ConfigDict(from_attributes=True)


class SubmissionsList(BaseModel):
    items: List[Submission]