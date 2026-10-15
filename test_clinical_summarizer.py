"""Run with:  python -m pytest -q   (pip install pytest)"""
from types import SimpleNamespace as NS

from datetime import datetime

from clinical_summarizer import (
    approve_report, build_clinical_note, extract_facts, format_report_text,
    ordered_text_from_vision, parse_summarize_payload, validate_report,
)


# ---- the five tests from the original zip ---------------------------------
def test_dose_units_and_medication_are_preserved():
    report = build_clinical_note("Digoxin 0.125 mg tablet once daily", 0.92)
    med = [f for f in report["facts"] if f["type"] == "medication"]
    assert med and med[0]["value"] == "Digoxin"
    assert med[0]["dose"] == "0.125 mg"
    assert "0.125 mg" in report["summary"]


def test_negation_is_flagged_and_source_traceable():
    facts = extract_facts("No fever. Patient reports cough.")
    neg = [f for f in facts if f["type"] == "negation"]
    assert neg and neg[0]["negated"] is True
    assert neg[0]["source_span"]["text"] == "No fever"


def test_low_ocr_confidence_warns():
    report = build_clinical_note("Patient age 70 years", 0.4)
    assert any("below 0.85" in w for w in report["review_flags"])


def test_empty_input_rejected():
    try:
        build_clinical_note("   ")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_does_not_claim_diagnosis():
    report = build_clinical_note("Patient reports dizziness.")
    assert report["status"] == "draft"
    assert "diagnosis" in " ".join(report["review_flags"]).lower()


# ---- regression tests for the fixes ---------------------------------------
def test_registration_number_is_not_a_negation():
    facts = extract_facts("REG. No. 58368")
    assert not [f for f in facts if f["type"] == "negation"]


def test_each_drug_gets_its_own_dose():
    facts = extract_facts("Metformin 500 mg twice daily, aspirin 75 mg")
    doses = {f["value"].lower(): f["dose"] for f in facts if f["type"] == "medication"}
    assert doses == {"metformin": "500 mg", "aspirin": "75 mg"}


def test_dose_written_before_drug():
    facts = extract_facts("500 mg metformin")
    assert [f["dose"] for f in facts if f["type"] == "medication"] == ["500 mg"]


def test_negation_stops_at_but():
    facts = extract_facts("No fever but cough")
    sym = {f["value"]: f["negated"] for f in facts if f["type"] == "symptom_or_finding"}
    assert sym == {"No fever": True, "cough": False}


def test_age_with_aged_and_percent_unit():
    assert [f["value"] for f in extract_facts("aged 18 years") if f["type"] == "age"] == [18]
    assert [f["value"] for f in extract_facts("Aged 18") if f["type"] == "age"] == [18]
    facts = extract_facts("Glucose 5% solution")
    assert facts[0]["value"] == ["5%"]


def test_lab_terms_are_found():
    values = [f["value"] for f in extract_facts("Indicate CKD\nRaised s. creatinine\nBlood urea")]
    assert any("CKD" in v for v in values)
    assert any("creatinine" in v for v in values)
    assert any("urea" in v for v in values)


# ---- reading order ---------------------------------------------------------
def _word(text, x, y, h=20):
    verts = [NS(x=x, y=y), NS(x=x + 10 * len(text), y=y), NS(x=x + 10 * len(text), y=y + h), NS(x=x, y=y + h)]
    return NS(symbols=[NS(text=c) for c in text], bounding_box=NS(vertices=verts))


def _annotation(words_in_vision_order):
    para = NS(words=words_in_vision_order)
    return NS(pages=[NS(blocks=[NS(paragraphs=[para])])])


def test_reading_order_rebuilds_lines():
    # Vision returned the right-hand words first; positions say they are on two lines.
    words = [_word("18", 200, 102), _word("years", 260, 99), _word("Date:", 20, 160),
             _word("aged", 130, 100), _word("08/07/2024", 90, 158), _word("Mr", 20, 101)]
    text = ordered_text_from_vision(_annotation(words))
    assert text == "Mr aged 18 years\nDate: 08/07/2024"


def test_reading_order_attaches_punctuation_and_handles_empty():
    words = [_word("Sir", 20, 50), _word(",", 55, 50)]
    assert ordered_text_from_vision(_annotation(words)) == "Sir,"
    assert ordered_text_from_vision(NS(pages=[])) == ""


# ---- input checks and review flow -----------------------------------------
def test_payload_checks():
    assert parse_summarize_payload({"ocr_text": " hi ", "ocr_confidence": 0.5, "source_name": "a.jpg"}) == ("hi", 0.5, "a.jpg")
    for bad in (None, [], {}, {"ocr_text": "  "}, {"ocr_text": "x", "ocr_confidence": 2},
                {"ocr_text": "x", "ocr_confidence": True}, {"ocr_text": "x" * 50001}):
        try:
            parse_summarize_payload(bad)
            assert False, f"should reject {bad!r}"[:80]
        except ValueError:
            pass


def test_validate_and_approve():
    draft = build_clinical_note("Aspirin 75 mg daily")
    assert validate_report(draft)["valid"] is True
    approved = approve_report(draft, " Dr. Roy ")
    assert approved["persisted"] is False and approved["report"]["approved_by"] == "Dr. Roy"
    assert draft["status"] == "draft"  # the original is not changed
    assert validate_report(approved["report"])["valid"] is False  # no pre-approved reports
    try:
        approve_report(draft, "  ")
        assert False
    except ValueError:
        pass


# ---- human-readable final report -------------------------------------------
NOW = datetime(2026, 10, 15, 2, 10)
NOTE = ("Referring Mr X aged 18 years. Metformin 500 mg twice daily, aspirin 75 mg\n"
        "No fever but cough\nIndicate CKD, raised s. creatinine\nDate: 08/07/2024\nREG. No. 58368")


def test_report_text_has_all_sections_and_content():
    text = format_report_text(build_clinical_note(NOTE, 0.8, "note.jpg"), NOW)
    for heading in ("CLINICAL NOTE SUMMARY REPORT", "1. SUMMARY", "2. PATIENT DETAILS", "3. MEDICATIONS",
                    "4. SYMPTOMS AND FINDINGS", "5. OTHER MEASUREMENTS AND FACTS",
                    "6. NEGATION STATEMENTS IN THE NOTE", "7. PLEASE VERIFY BEFORE USE",
                    "8. PRIVACY MASKING", "9. ORIGINAL OCR TEXT", "REVIEW SIGN-OFF"):
        assert heading in text, heading
    assert "DRAFT - NOT APPROVED" in text and "15-10-2026 02:10" in text and "note.jpg" in text
    assert "Age: 18 years" in text and "08/07/2024" in text
    assert "1. Metformin" in text and "Dose:   500 mg" in text and "2. aspirin" in text and "Dose:   75 mg" in text
    assert "below 0.85" in text
    assert "Reviewed by: ____" in text
    assert text.count("No fever") >= 1 and "Written as negated" in text
    assert all(len(line) <= 90 for line in text.splitlines())


def test_report_text_after_approval_and_after_edits():
    approved = approve_report(build_clinical_note(NOTE, 0.95, "note.jpg"), "Dr R")["report"]
    approved["summary"] = "Edited by reviewer."          # the reviewer's edit must appear
    text = format_report_text(approved, NOW)
    assert "APPROVED by Dr R" in text and "Edited by reviewer." in text
    assert "Reviewed and approved by: Dr R" in text and "Reviewed by: ____" not in text
    assert "below 0.85" not in text


def test_report_text_survives_empty_or_broken_reports():
    assert "None found in the text." in format_report_text(build_clinical_note("Thank you"), NOW)
    broken = {"summary": 5, "clinical_report": {"medications": ["x", None, {"value": "aspirin"}]},
              "review_flags": [1, "ok"], "source": "oops"}
    text = format_report_text(broken, NOW)
    assert "aspirin" in text and "not found - check the source image" in text
    # no clinical_report: rebuilt from the flat facts list
    flat = build_clinical_note("Aspirin 75 mg")
    del flat["clinical_report"]
    assert "Dose:   75 mg" in format_report_text(flat, NOW)
    try:
        format_report_text(["not", "a", "dict"])
        assert False
    except ValueError:
        pass


def test_no_as_a_label_is_not_a_negation():
    for label in ("AR NO: 5089836", "REG. No. 58368", "UHID No A1234", "Bill No 77", "No. of tablets 3", "OP NO.: 4411"):
        assert not [f for f in extract_facts(label) if f["type"] == "negation"], label
    assert [f for f in extract_facts("No fever") if f["type"] == "negation"]
    assert [f for f in extract_facts("Fever: no.") if f["type"] == "negation"]
    assert [f for f in extract_facts("Patient denies pain") if f["type"] == "negation"]
    assert [f for f in extract_facts("No. fever noted") if f["type"] == "negation"]


def test_report_privacy_section():
    off = format_report_text(build_clinical_note(NOTE, 0.8), NOW)
    assert "Automatic masking was OFF" in off and "9. ORIGINAL OCR TEXT" in off
    priv = {"masked": True, "counts": {"PATIENT_NAME": 2, "PHONE": 1}, "uncertain": ["Line 3: check this."]}
    on = format_report_text(build_clinical_note(NOTE, 0.8, "uploaded_note.jpg", priv), NOW)
    assert "Automatic masking was ON" in on and "PATIENT_NAME" in on and "Line 3: check this." in on
    assert "personal details masked" in on and "Automatic masking was OFF" not in on
    assert "Not recorded" in format_report_text({"summary": "x"}, NOW)
