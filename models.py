from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, TIMESTAMP, func
from sqlalchemy.dialects.mysql import JSON

Base = declarative_base()

class FormSubmission(Base):
    __tablename__ = "form_submissions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    form_type = Column(String(200))
    fields = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.now())

