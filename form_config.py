# form_config.py

FORM_DEFINITIONS = {
    "form49a": {
        "human_name": "PAN Form 49A",
        "required_fields": [
            {"key": "full_name", "label": "Full name (as per Aadhaar)"},
            {"key": "father_name", "label": "Father's name"},
            {"key": "dob", "label": "Date of birth (DD-MM-YYYY)"},
            {"key": "address", "label": "Residential address"},
            {"key": "phone", "label": "Mobile number"},
        ]
    }
    # later you can add more forms here
}


def get_form_definition(form_type: str):
    return FORM_DEFINITIONS.get(form_type)
