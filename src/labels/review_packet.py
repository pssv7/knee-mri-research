"""Render self-contained report packets without any model annotations."""
import html
import json

from src.labels.openai_extract import FIELDS
from src.validation.proxy import LABELS


def blank_annotations(packet):
    return {"packet_id": packet["packet_id"], "packet_sha256": packet["packet_sha256"],
            "review_mode": "AI-assisted only", "reviewer": "", "status": "draft",
            "rows": [{"case_id": case["case_id"], "finding": finding,
                      "report_sha256": case["report_sha256"],
                      **{field: "" for field in FIELDS}, "note": ""}
                     for case in packet["cases"] for finding in LABELS]}


def render(packet, template):
    payload = json.dumps({"packet": packet, "draft": blank_annotations(packet)}, ensure_ascii=False)
    # Report text cannot terminate the embedded script or create HTML elements.
    payload = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return template.replace("__TITLE__", html.escape(packet["title"])).replace("__PAYLOAD__", payload)
