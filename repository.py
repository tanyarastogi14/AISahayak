# repository.py
from typing import Dict, Any, List, TypedDict
from db import SessionLocal
from models import FormSubmission

class Submission(TypedDict):
    id: int
    form_type: str
    fields: Dict[str, Any]

def save_submission(form_type: str, fields: Dict[str, Any]) -> int:
    db = SessionLocal()
    try:
        obj = FormSubmission(form_type=form_type, fields=fields)
        db.add(obj)
        db.commit()
        db.refresh(obj)
        return obj.id
    finally:
        db.close()

def get_submissions(limit: int = 50) -> List[Submission]:
    db = SessionLocal()
    try:
        rows = db.query(FormSubmission).order_by(FormSubmission.created_at.desc()).limit(limit).all()
        return [{"id": r.id, "form_type": r.form_type, "fields": r.fields} for r in rows]
    finally:
        db.close()

                 