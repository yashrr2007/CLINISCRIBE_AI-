"""
App-level privacy test: uploads an INVENTED note (with a fake Google Vision client, so no key and no
internet are needed) and checks that no identifier reaches the browser, the report or the file name.

Run with:  python -m pytest -q
"""
import io
import json
import random
from types import SimpleNamespace as NS

import app as A

LINES = [
    "Apollo Heritage Hospital, Chennai",
    "UHID: AB123456 Bill No 7788",
    "Name: Ravi Shankar Iyer Age: 45 M",
    "Address: 12, Gandhi Road, Kolathur, Chennai - 600099",
    "Ph: +91 98765 43210 Email: ravi.iyer@example.com",
    "DOB: 04/05/1979 Aadhaar: 1234 5678 9012 PAN ABCDE1234F",
    "Date: 08/07/2024",
    "Referring Mr Ravi Iyer aged 45 years for consultation.",
    "Dear Dr Meena Krishnan Sir,",
    "Indicate CKD, raised s. creatinine 2.4 mg/dL, Blood urea 80",
    "Metformin 500 mg twice daily. No fever.",
    "AR NO: 5089836",
    "Regards",
    "DR. K. ANIL ROY",
    "REG. No. 58368",
]
SECRETS = ["Ravi", "Shankar", "Iyer", "Meena", "Krishnan", "Anil", "Roy", "Apollo", "Heritage", "Gandhi",
           "Kolathur", "600099", "98765", "43210", "ravi.iyer", "example.com", "5678", "9012", "ABCDE1234F",
           "AB123456", "7788", "5089836", "58368", "08/07/2024", "04/05/1979", "ravi_iyer"]
MEDICAL = ["Metformin", "500 mg", "creatinine 2.4 mg/dL", "Blood urea 80", "CKD"]


class FakeVision:
    """Returns the note's words in scrambled order, like the real service does."""

    def document_text_detection(self, image=None, image_context=None):
        words = []
        for row, line in enumerate(LINES):
            x, y = 30, 40 + row * 55
            for tok in line.split():
                w = 12 * len(tok)
                verts = [NS(x=x, y=y), NS(x=x + w, y=y), NS(x=x + w, y=y + 28), NS(x=x, y=y + 28)]
                words.append(NS(confidence=0.9, symbols=[NS(text=c) for c in tok],
                                bounding_box=NS(vertices=verts), t=tok))
                x += w + 14
        random.Random(7).shuffle(words)
        ann = NS(text="\n".join(w.t for w in words), pages=[NS(blocks=[NS(paragraphs=[NS(words=words)])])])
        return NS(error=NS(message=""), full_text_annotation=ann)


def _client(monkeypatch):
    monkeypatch.setattr(A, "get_client", lambda: FakeVision())
    return A.app.test_client()


def _leaks(blob):
    return [s for s in SECRETS if s.lower() in blob.lower()]


def _upload(client, name="ravi_iyer_note.jpg"):
    return client.post("/extract-text", data={"image": (io.BytesIO(b"x"), name)},
                       content_type="multipart/form-data")


def test_upload_reaches_the_browser_fully_masked(monkeypatch):
    r = _upload(_client(monkeypatch))
    assert r.status_code == 200
    data = r.get_json()
    assert _leaks(json.dumps(data)) == []
    assert data["privacy"]["masked"] is True
    assert data["clinical_draft"]["source"]["name"] == "uploaded_note.jpg"
    assert all(m in data["ordered_text"] for m in MEDICAL)  # medical content is kept


def test_report_and_regenerate_do_not_leak_and_keep_total_counts(monkeypatch):
    client = _client(monkeypatch)
    draft = _upload(client).get_json()["clinical_draft"]
    report = client.post("/api/report-text", json=draft).get_json()["text"]
    assert _leaks(report) == []

    # the reviewer fixes a word and adds a NEW identifier; re-generate runs on already-masked text
    text = draft["source"]["text"] + "\nMr Venkat Raman aged 50, Ph 9123456780"
    again = client.post("/api/summarize", json={"ocr_text": text, "source_name": "venkat_raman.jpg"}).get_json()
    blob = json.dumps(again)
    assert "Venkat" not in blob and "9123456780" not in blob and "venkat_raman" not in blob
    counts = again["privacy"]["counts"]
    assert counts["PATIENT_NAME"] >= 2 and counts["PHONE"] >= 2 and counts["RECORD_ID"] >= 4  # earlier masking still counted
    assert _leaks(client.post("/api/report-text", json=again).get_json()["text"]) == []


def test_the_masking_switch_is_not_controlled_by_the_request(monkeypatch):
    client = _client(monkeypatch)
    r = client.post("/api/summarize", json={"ocr_text": "Mr Venkat Raman aged 50", "mask": False})
    assert r.get_json()["privacy"]["masked"] is True and "Venkat" not in json.dumps(r.get_json())
