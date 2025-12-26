import os
import io
from typing import Dict, Any

from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter

# Base directory of this file (AISahayak folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _make_overlay(page_width, page_height, placements):
    """
    placements: list of dicts: { "x": float, "y": float, "text": str, "size": int }
    Coordinates: (0,0) is bottom-left of the page.
    """
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=(page_width, page_height))

    for p in placements:
        x = p["x"]
        y = p["y"]
        text = p.get("text", "")
        size = p.get("size", 10)

        c.setFont("Helvetica", size)
        c.drawString(x, y, text)

    c.save()
    packet.seek(0)
    return packet.read()


def _merge_overlay(template_pdf_path, overlays, output_path):
    """
    overlays: { page_index: overlay_bytes }
    """
    reader = PdfReader(template_pdf_path)
    writer = PdfWriter()

    for i, page in enumerate(reader.pages):
        base = page
        if i in overlays:
            overlay_reader = PdfReader(io.BytesIO(overlays[i]))
            base.merge_page(overlay_reader.pages[0])
        writer.add_page(base)

    with open(output_path, "wb") as f:
        writer.write(f)


def generate_pension_form_pdf(
    fields: Dict[str, Any],
    # Put the Annexure-I PDF in the same folder as this file,
    # or change this path to wherever you store templates.
    template_path: str = os.path.join(
        BASE_DIR, "F_List of joining form635216025148553564-5.pdf"
    ),
    output_dir: str = os.path.join(BASE_DIR, "generated_pdfs"),
) -> str:
    """
    Generate filled NEW PENSION SCHEME Annexure-I PDF.

    Expected fields dict (you can adjust keys as you like, but keep in sync
    with the Streamlit/app/frontend):

        {
          "govt_servant_name": "...",
          "designation": "...",
          "ministry_name": "...",
          "scale_of_pay": "...",
          "dob": "...",
          "date_of_joining": "...",
          "basic_pay": "...",

          "nominee1_name": "...",
          "nominee1_age": "...",
          "nominee1_share": "...",
          "nominee1_relation": "...",

          "nominee2_name": "...",
          "nominee2_age": "...",
          "nominee2_share": "...",
          "nominee2_relation": "...",

          "nominee3_name": "...",
          "nominee3_age": "...",
          "nominee3_share": "...",
          "nominee3_relation": "...",

          "nominee4_name": "...",
          "nominee4_age": "...",
          "nominee4_share": "...",
          "nominee4_relation": "...",

          "declarant_name": "...",
          "declarant_signature": "..."   # optional text near signature line
        }

    Returns: path to generated PDF.
    """

    print("TEMPLATE PATH:", template_path)
    print("TEMPLATE EXISTS:", os.path.exists(template_path))

    os.makedirs(output_dir, exist_ok=True)

    reader = PdfReader(template_path)
    page = reader.pages[0]
    w = float(page.mediabox.width)
    h = float(page.mediabox.height)

    # ---------- Top section fields ----------
    govt_servant_name = fields.get("govt_servant_name", "")
    designation = fields.get("designation", "")
    ministry_name = fields.get("ministry_name", "")
    scale_of_pay = fields.get("scale_of_pay", "")
    dob = fields.get("dob", "")
    date_of_joining = fields.get("date_of_joining", "")
    basic_pay = fields.get("basic_pay", "")

    # Starting X / Y for the right-hand blanks of items 1–7.
    # These are relative to page width/height so it works even if page size changes a bit.
    top_x = w * 0.35
    first_line_y = h - 200  # roughly aligned with item 1's blank
    line_gap = 30           # vertical distance between each line

    placements_page0 = [
        # 1. Name of Govt. Servant
        {"x": top_x, "y": first_line_y, "text": govt_servant_name, "size": 11},
        # 2. Designation
        {"x": top_x, "y": first_line_y - 1 * line_gap, "text": designation, "size": 11},
        # 3. Name of Ministry / Dept. / Organisation
        {"x": top_x, "y": first_line_y - 2 * line_gap, "text": ministry_name, "size": 11},
        # 4. Scale of Pay
        {"x": top_x, "y": first_line_y - 3 * line_gap, "text": scale_of_pay, "size": 11},
        # 5. Date of Birth
        {"x": top_x, "y": first_line_y - 4 * line_gap, "text": dob, "size": 11},
        # 6. Date of joining Govt. Service
        {"x": top_x, "y": first_line_y - 5 * line_gap, "text": date_of_joining, "size": 11},
        # 7. Basic Pay
        {"x": top_x, "y": first_line_y - 6 * line_gap, "text": basic_pay, "size": 11},
    ]

    # ---------- Nominee table (item 8) ----------
    # The table is around the middle/bottom of the page (see uploaded PDF).
    # Adjust these positions slightly if text doesn't sit perfectly in the cells.
    nominee_table_top_y = h - 360
    row_height = 35

    # X positions of each column (excluding Sr. No., which is already printed)
    name_col_x = w * 0.22
    age_col_x = w * 0.57
    share_col_x = w * 0.68
    relation_col_x = w * 0.80

    for i in range(1, 5):
        row_y = nominee_table_top_y - (i - 1) * row_height

        name_key = f"nominee{i}_name"
        age_key = f"nominee{i}_age"
        share_key = f"nominee{i}_share"
        relation_key = f"nominee{i}_relation"

        placements_page0.append(
            {"x": name_col_x, "y": row_y, "text": str(fields.get(name_key, "")), "size": 10}
        )
        placements_page0.append(
            {"x": age_col_x, "y": row_y, "text": str(fields.get(age_key, "")), "size": 10}
        )
        placements_page0.append(
            {"x": share_col_x, "y": row_y, "text": str(fields.get(share_key, "")), "size": 10}
        )
        placements_page0.append(
            {"x": relation_col_x, "y": row_y, "text": str(fields.get(relation_key, "")), "size": 10}
        )

    # ---------- Bottom Name & Signature ----------
    declarant_name = fields.get("declarant_name", govt_servant_name)
    signature_text = fields.get("declarant_signature", "")

    name_x = w * 0.60
    name_y = h * 0.14     # "Name ____________"
    sign_x = w * 0.60
    sign_y = h * 0.11     # "Signature _______"

    placements_page0.extend(
        [
            {"x": name_x, "y": name_y, "text": declarant_name, "size": 11},
            {"x": sign_x, "y": sign_y, "text": signature_text, "size": 11},
        ]
    )

    # ---------- Create overlay & merge ----------
    overlay0 = _make_overlay(w, h, placements_page0)
    overlays = {0: overlay0}

    out_path = os.path.join(output_dir, "new_pension_scheme_annexure1_filled.pdf")
    _merge_overlay(template_path, overlays, out_path)

    print("GENERATED PDF:", out_path)
    return out_path
