"""
clinical_summarizer.py - turns OCR text into a structured, review-first clinical draft.

Adapted from the HT-24 Clinical Note Summarizer (app/pipeline.py and the
validate/approve logic in app/main.py). Pure standard-library Python: no extra
packages are needed.

It does NOT diagnose, recommend treatment, or store anything. Every output is a
DRAFT that a clinician must check against the original image.

Changes from the original zip are marked with "# FIX:" so they are easy to review.

Sections
    1. Reading order    - rebuild lines from Google Vision word positions
    2. Fact extraction  - medications, doses, dates, ages, symptoms, negations
    3. Report building  - build_clinical_note()
    4. Validate/approve - the review workflow (session only, nothing is saved)
"""
import re
import textwrap
from datetime import datetime
from statistics import median
from typing import Optional

# ---------------------------------------------------------------------------
# 1. Reading order (new): Vision returns text in block order, not page order
# ---------------------------------------------------------------------------
_ATTACH_TO_PREVIOUS = set(",.;:!?)")


def ordered_text_from_vision(annotation) -> str:
    """
    Rebuild the text in natural reading order (top to bottom, left to right)
    from the bounding box of every word in a Vision `full_text_annotation`.

    Words are grouped into a line when their vertical centres are close
    (within ~60% of the median word height), then sorted left to right.
    Limits: strongly rotated pages and multi-column layouts can still mix up.
    """
    lines_out = []
    for page in getattr(annotation, "pages", []) or []:
        items = []
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    text = "".join(sym.text for sym in word.symbols)
                    verts = word.bounding_box.vertices
                    xs = [v.x for v in verts]
                    ys = [v.y for v in verts]
                    if not text or not xs or not ys:
                        continue
                    items.append({
                        "text": text,
                        "x0": min(xs),
                        "yc": (min(ys) + max(ys)) / 2.0,
                        "h": max(max(ys) - min(ys), 1),
                    })
        if not items:
            continue

        tolerance = 0.6 * median(i["h"] for i in items)
        items.sort(key=lambda i: i["yc"])
        lines, current = [], [items[0]]
        for item in items[1:]:
            line_yc = sum(i["yc"] for i in current) / len(current)
            if abs(item["yc"] - line_yc) <= tolerance:
                current.append(item)
            else:
                lines.append(current)
                current = [item]
        lines.append(current)

        for line in lines:
            line.sort(key=lambda i: i["x0"])
            parts = []
            for w in line:
                if parts and w["text"] and all(c in _ATTACH_TO_PREVIOUS for c in w["text"]):
                    parts[-1] += w["text"]
                else:
                    parts.append(w["text"])
            lines_out.append(" ".join(parts))
    return "\n".join(lines_out)


# ---------------------------------------------------------------------------
# 2. Fact extraction (from the zip's pipeline.py)
# ---------------------------------------------------------------------------
# FIX: "no" used as a label is not a negation: "Reg. No. 58368", "AR NO: 5089836",
# "UHID No A1234", "No. of tablets". "Fever: no." and "No fever" are still negations.
NEGATION_RE = re.compile(
    r"\b(?:no|not|denies|denied|without|negative|none|never)\b"
    r"(?!\s*[.:#-]*\s*(?:[A-Za-z]*\d|of\b|\[))",  # \[ = a masked value like [RECORD_ID]
    re.I,
)
# FIX (new): negation does not carry across "but / however / although ..."
CONTRAST_RE = re.compile(r"\b(?:but|however|although|though|except|yet)\b", re.I)

MEDICATION_ACTIONS = {
    "start": r"\b(start(?:ed)?|begin(?:ning)?|commence(?:d)?)\b",
    "stop": r"\b(stop(?:ped)?|discontinue(?:d)?|cease(?:d)?)\b",
    "increase": r"\b(increase(?:d)?|raise(?:d)?|up[- ]titrate(?:d)?)\b",
    "decrease": r"\b(decrease(?:d)?|reduce(?:d)?|lower(?:ed)?|down[- ]titrate(?:d)?)\b",
    "continue": r"\b(continue(?:d)?|maintain(?:ed)?)\b",
    "hold": r"\b(hold|held|withhold|withheld)\b",
}

# FIX: "%" never matched in the original (\b after "%" fails); lab units like
# "mg/dL" were cut to "mg". Both handled here.
DOSE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:(?:mcg|µg|mg|g|kg|ml|L|units?|IU|mmol|mol|tablets?|capsules?)"
    r"(?:\s?/\s?(?:dL|L|mL|min|day|kg))?\b|%)",
    re.I,
)
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|(?:\d{1,2}\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b",
    re.I,
)
# FIX: also understands "aged 18" (the original needed the word "years").
AGE_RE = re.compile(
    r"\b(?:aged?\s*[:=]?\s*(\d{1,3})\b|(\d{1,3})\s*(?:years?(?:\s*old)?|yrs?|yo)\b)", re.I
)

# FIX: the zip knew only 13 drugs. Edit this list freely; it is the one place to extend.
MEDICATIONS = [
    "digoxin", "warfarin", "aspirin", "paracetamol", "acetaminophen", "ibuprofen",
    "metformin", "insulin", "amoxicillin", "penicillin", "atorvastatin", "lisinopril",
    "amlodipine", "omeprazole",
    # added
    "dolo", "crocin", "pantoprazole", "telmisartan", "losartan", "furosemide",
    "clopidogrel", "azithromycin", "ceftriaxone", "cetirizine", "prednisolone",
    "levothyroxine", "glimepiride", "ramipril", "metoprolol", "atenolol",
    "salbutamol", "diclofenac", "ranitidine", "ondansetron",
]
MED_RE = re.compile(r"\b(?:" + "|".join(MEDICATIONS) + r")\b", re.I)

# FIX: added renal/lab terms and common conditions; the zip listed symptoms only.
SYMPTOM_RE = re.compile(
    r"\b(?:pain|fever|cough|breathlessness|nausea|vomiting|dizziness|swelling|symptoms?"
    r"|ckd|creatinine|urea|hba1c|haemoglobin|hemoglobin|raised|elevated"
    r"|hypertension|diabetes|anaemia|anemia|fatigue|headache|rash|infection)\b",
    re.I,
)


def _sentences(text: str):
    return [s.strip() for s in re.split(r"\n+|;|(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]


def _negation(sentence: str) -> bool:
    return bool(NEGATION_RE.search(sentence))


def _source_span(text: str, phrase: str):
    idx = text.lower().find(phrase.lower())
    if idx < 0:
        return None
    return {"start": idx, "end": idx + len(phrase), "text": text[idx:idx + len(phrase)]}


def _clause_bounds(sentence: str):
    """Split a sentence at contrast words so 'no fever but cough' is two clauses."""
    bounds, start = [], 0
    for m in CONTRAST_RE.finditer(sentence):
        bounds.append((start, m.start()))
        start = m.end()
    bounds.append((start, len(sentence)))
    return bounds


def _dose_for_medication(sentence, med, prev_end, next_start, dose_first):
    """
    FIX: link each drug to its own dose. The zip gave every drug in a sentence the
    first dose found, so 'metformin 500 mg, aspirin 75 mg' gave aspirin 500 mg.
    """
    if dose_first:  # layout like "500 mg metformin"
        found = list(DOSE_RE.finditer(sentence, prev_end, med.start()))
        return found[-1].group(0) if found else None
    m = DOSE_RE.search(sentence, med.end(), next_start)  # layout like "metformin 500 mg"
    return m.group(0) if m else None


def _action_for(segment: str) -> str:
    return next(
        (a for a, p in MEDICATION_ACTIONS.items() if re.search(p, segment, re.I)),
        "unspecified",
    )


def extract_facts(ocr_text: str):
    """Extract literal cues with source offsets. No diagnosis or clinical assumptions."""
    facts = []
    for sentence in _sentences(ocr_text):
        # FIX: drop the trailing full stop so the traced span reads "No fever", not "No fever."
        shown = sentence.rstrip(".!? ")
        common = {
            "source_text": shown,
            "source_span": _source_span(ocr_text, shown),
            "negated": _negation(sentence),
        }

        meds = list(MED_RE.finditer(sentence))
        dose_matches = list(DOSE_RE.finditer(sentence))
        if meds:
            dose_first = bool(dose_matches) and dose_matches[0].start() < meds[0].start()
            for i, m in enumerate(meds):
                prev_end = meds[i - 1].end() if i else 0
                next_start = meds[i + 1].start() if i + 1 < len(meds) else len(sentence)
                facts.append({
                    **common, "type": "medication", "value": m.group(0),
                    "dose": _dose_for_medication(sentence, m, prev_end, next_start, dose_first),
                    "action": _action_for(sentence[prev_end:next_start]),
                })
        elif dose_matches:
            facts.append({**common, "type": "dose_or_measurement",
                          "value": [m.group(0) for m in dose_matches], "action": "unspecified"})

        if NEGATION_RE.search(sentence):
            facts.append({**common, "type": "negation", "value": shown})

        # FIX: negation is decided per clause, not per sentence.
        for start, end in _clause_bounds(sentence):
            clause = sentence[start:end].strip(" ,.;:!?")
            if clause and SYMPTOM_RE.search(clause):
                facts.append({
                    "source_text": clause,
                    "source_span": _source_span(ocr_text, clause),
                    "negated": _negation(clause),
                    "type": "symptom_or_finding",
                    "value": clause,
                })

    for m in AGE_RE.finditer(ocr_text):
        facts.append({"type": "age", "value": int(m.group(1) or m.group(2)),
                      "source_text": m.group(0),
                      "source_span": {"start": m.start(), "end": m.end(), "text": m.group(0)},
                      "negated": False})
    for m in DATE_RE.finditer(ocr_text):
        facts.append({"type": "date", "value": m.group(0), "source_text": m.group(0),
                      "source_span": {"start": m.start(), "end": m.end(), "text": m.group(0)},
                      "negated": False})

    # Stable de-duplication while preserving source order.
    out, seen = [], set()
    for f in facts:
        key = (f.get("type"), str(f.get("value")), f.get("source_text"))
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# 3. Report building (from the zip's pipeline.py, unchanged in structure)
# ---------------------------------------------------------------------------
OCR_CONFIDENCE_THRESHOLD = 0.85  # a review heuristic, not a validated clinical threshold


def build_clinical_note(ocr_text: str, ocr_confidence: Optional[float] = None,
                        source_name: Optional[str] = None, privacy: Optional[dict] = None):
    if not ocr_text or not ocr_text.strip():
        raise ValueError("OCR text is required.")
    text = ocr_text.strip()
    facts = extract_facts(text)
    warnings = [
        "This is a draft generated from OCR text, not medical advice or a diagnosis.",
        "A clinician must verify all medication names, doses, units, dates, negations, "
        "and patient details against the source image.",
    ]
    if ocr_confidence is not None and ocr_confidence < OCR_CONFIDENCE_THRESHOLD:
        warnings.append(
            f"OCR confidence is below {OCR_CONFIDENCE_THRESHOLD}; carefully compare the "
            "transcription with the original image."
        )
    if not facts:
        warnings.append("No structured clinical facts were confidently recognized; review the raw OCR text.")

    # Concise summary is extractive: it does not add inferred clinical meaning.
    summary = " ".join(_sentences(text)[:3])
    known = ("age", "date", "medication", "symptom_or_finding", "negation", "dose_or_measurement")
    return {
        "schema_version": "1.0",
        "status": "draft",
        # "text" (new) keeps the reviewed OCR text so the final report can print it.
        "source": {"type": "existing_ocr", "name": source_name,
                   "ocr_confidence": ocr_confidence, "text": text},
        # "privacy" (new): whether personal details were masked, and what to double-check.
        # Holds counts and notes only, never the original values.
        "privacy": privacy if privacy is not None else {"masked": False, "counts": {}, "uncertain": []},
        "summary": summary,
        "clinical_report": {
            "patient_details": [f for f in facts if f["type"] in ("age", "date")],
            "medications": [f for f in facts if f["type"] == "medication"],
            "symptoms_and_findings": [f for f in facts if f["type"] == "symptom_or_finding"],
            "negations": [f for f in facts if f["type"] == "negation"],
            "other_measurements": [f for f in facts if f["type"] == "dose_or_measurement"],
            "other_facts": [f for f in facts if f["type"] not in known],
        },
        "facts": facts,
        "review_flags": warnings,
        "limitations": [
            "Rule-based extraction may miss or misclassify handwritten/OCR content.",
            "Medication-dose association is heuristic; confirm each association manually.",
            "No persistence, authentication, or electronic health record integration is configured.",
        ],
    }


# ---------------------------------------------------------------------------
# 4. Input checks, validate and approve (from the zip's main.py, as plain functions)
# ---------------------------------------------------------------------------
def parse_summarize_payload(data):
    """Check a request body like the zip's SummarizeRequest. Returns (text, confidence, name)."""
    if not isinstance(data, dict):
        raise ValueError("Send a JSON object with an 'ocr_text' field.")
    text = data.get("ocr_text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("OCR text is required.")
    text = text.strip()
    if len(text) > 50000:
        raise ValueError("OCR text is too long (maximum 50,000 characters).")
    conf = data.get("ocr_confidence")
    if conf is not None:
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
            raise ValueError("OCR confidence must be a number between 0 and 1.")
    name = data.get("source_name")
    if name is not None:
        if not isinstance(name, str) or len(name) > 255:
            raise ValueError("Source name must be text of at most 255 characters.")
        name = name.strip() or None
    return text, conf, name


def validate_report(report):
    issues = []
    if not isinstance(report, dict):
        return {"valid": False, "issues": ["Report must be a JSON object."]}
    if not isinstance(report.get("summary", ""), str):
        issues.append("Summary must be text.")
    if not isinstance(report.get("facts", []), list):
        issues.append("Facts must be a list.")
    if report.get("status") == "approved":
        issues.append("Approval is managed through the review UI; do not submit a pre-approved report.")
    return {"valid": not issues, "issues": issues}


def approve_report(report, approved_by):
    """Marks a report approved in the RESPONSE only. Nothing is stored anywhere."""
    if not isinstance(report, dict):
        raise ValueError("Report must be a JSON object.")
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise ValueError("Reviewer name or initials are required.")
    if len(approved_by) > 120:
        raise ValueError("Reviewer name is too long (maximum 120 characters).")
    approved = dict(report)
    approved["status"] = "approved"
    approved["approved_by"] = approved_by.strip()
    approved["approval_notice"] = "Local response only; no database or audit log is configured."
    return {"report": approved, "persisted": False}


# ---------------------------------------------------------------------------
# 5. Human-readable final report (new)
# ---------------------------------------------------------------------------
_WIDTH = 78
_GROUP_KEYS = ("patient_details", "medications", "symptoms_and_findings",
               "negations", "other_measurements", "other_facts")


def _as_dicts(items):
    return [f for f in (items if isinstance(items, list) else []) if isinstance(f, dict)]


def _grouped(report):
    """Facts grouped by section. Falls back to the flat 'facts' list if the
    reviewer edited the JSON and removed 'clinical_report'."""
    cr = report.get("clinical_report")
    if isinstance(cr, dict):
        return {k: _as_dicts(cr.get(k)) for k in _GROUP_KEYS}
    facts = _as_dicts(report.get("facts"))
    known = ("age", "date", "medication", "symptom_or_finding", "negation", "dose_or_measurement")

    def pick(*types):
        return [f for f in facts if f.get("type") in types]

    return {
        "patient_details": pick("age", "date"),
        "medications": pick("medication"),
        "symptoms_and_findings": pick("symptom_or_finding"),
        "negations": pick("negation"),
        "other_measurements": pick("dose_or_measurement"),
        "other_facts": [f for f in facts if f.get("type") not in known],
    }


def _wrap(text, indent="", bullet=""):
    return textwrap.fill(str(text), width=_WIDTH, initial_indent=indent + bullet,
                         subsequent_indent=indent + " " * len(bullet))


def _section(title, lines):
    return ["", title, "-" * len(title)] + (lines or ["  None found in the text."])


def _join(value):
    return ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)


def _masked(privacy):
    return isinstance(privacy, dict) and privacy.get("masked") is True


def _privacy_lines(privacy):
    if not isinstance(privacy, dict):
        return ["  Not recorded for this report."]
    if not _masked(privacy):
        return ["  Automatic masking was OFF. This report may contain the patient's name, address,",
                "  phone number and other personal details."]
    counts = privacy.get("counts") if isinstance(privacy.get("counts"), dict) else {}
    lines = ["  Automatic masking was ON. Items replaced by tags such as [PATIENT_NAME]:"]
    lines += [f"    {k:<14} {v}" for k, v in counts.items()] or ["    nothing needed masking"]
    notes = [t for t in (privacy.get("uncertain") or []) if isinstance(t, str)]
    if notes:
        lines += ["", "  Please double-check:"] + [_wrap(t, "  ", "[ ] ") for t in notes]
    lines += ["", _wrap("Masking uses fixed rules, so a name written without a title or label, or split "
                        "over two lines, can be missed. Read the text before sharing it.", "  ")]
    return lines


def format_report_text(report, generated_at=None):
    """
    Turn a draft/approved report (the JSON dict) into a plain-text clinical report
    that a person can read, print or paste. Works on whatever the reviewer edited.
    """
    if not isinstance(report, dict):
        raise ValueError("Report must be a JSON object.")
    when = (generated_at or datetime.now()).strftime("%d-%m-%Y %H:%M")
    source = report.get("source") if isinstance(report.get("source"), dict) else {}
    approved = report.get("status") == "approved"
    reviewer = str(report.get("approved_by") or "").strip()
    groups = _grouped(report)

    conf = source.get("ocr_confidence")
    if isinstance(conf, (int, float)) and not isinstance(conf, bool):
        conf_text = f"{conf}"
        if conf < OCR_CONFIDENCE_THRESHOLD:
            conf_text += f"  (below {OCR_CONFIDENCE_THRESHOLD}: compare carefully with the original image)"
    else:
        conf_text = "not available"

    def row(label, value):
        return f"{label:<16}{value}"

    out = ["CLINICAL NOTE SUMMARY REPORT", "=" * 28, ""]
    out.append(row("Status:", f"APPROVED by {reviewer or 'reviewer'}" if approved
                   else "DRAFT - NOT APPROVED"))
    out.append(row("Source image:", source.get("name") or "not stated"))
    out.append(row("OCR confidence:", conf_text))
    out.append(row("Report created:", when))

    # 1. Summary
    summary = str(report.get("summary") or "").strip()
    out += _section("1. SUMMARY", [_wrap(summary, "  ")] if summary else [])

    # 2. Patient details
    lines = []
    for f in groups["patient_details"]:
        if f.get("type") == "age":
            lines.append(f"  Age: {f.get('value')} years   (written as \"{f.get('source_text', '')}\")")
        elif f.get("type") == "date":
            lines.append(f"  Date written in the note: {f.get('value')}")
        else:
            lines.append(_wrap(f"{f.get('type')}: {_join(f.get('value'))}", "  "))
    out += _section("2. PATIENT DETAILS", lines)

    # 3. Medications
    lines = []
    for i, f in enumerate(groups["medications"], 1):
        action = f.get("action")
        lines.append(f"  {i}. {f.get('value')}")
        lines.append(f"     Dose:   {f.get('dose') or 'not found - check the source image'}")
        lines.append(f"     Action: {'not stated' if action in (None, '', 'unspecified') else action}")
        if f.get("negated"):
            lines.append("     Note:   the sentence contains a negation word - check its meaning.")
        lines.append(_wrap(f"From: \"{f.get('source_text', '')}\"", "     "))
    out += _section("3. MEDICATIONS", lines)

    # 4. Symptoms and findings
    lines = []
    for i, f in enumerate(groups["symptoms_and_findings"], 1):
        lines.append(_wrap(_join(f.get("value")), "  ", f"{i}. "))
        if f.get("negated"):
            lines.append("     Written as negated in the note (for example \"No ...\").")
    out += _section("4. SYMPTOMS AND FINDINGS", lines)

    # 5. Other measurements and facts
    lines = []
    for f in groups["other_measurements"]:
        lines.append(_wrap(f"{_join(f.get('value'))}   (from: \"{f.get('source_text', '')}\")", "  ", "- "))
    for f in groups["other_facts"]:
        lines.append(_wrap(f"{f.get('type')}: {_join(f.get('value'))}", "  ", "- "))
    out += _section("5. OTHER MEASUREMENTS AND FACTS", lines)

    # 6. Negation statements
    lines = [_wrap(f"\"{_join(f.get('value'))}\"", "  ", "- ") for f in groups["negations"]]
    out += _section("6. NEGATION STATEMENTS IN THE NOTE", lines)

    # 7. To verify
    lines = [_wrap(t, "  ", "[ ] ") for t in report.get("review_flags", []) if isinstance(t, str)]
    limits = [t for t in report.get("limitations", []) if isinstance(t, str)]
    if limits:
        lines += ["", "  Limits of this tool:"] + [_wrap(t, "  ", "- ") for t in limits]
    out += _section("7. PLEASE VERIFY BEFORE USE", lines)

    # 8. Privacy masking
    out += _section("8. PRIVACY MASKING", _privacy_lines(report.get("privacy")))

    # 9. Original OCR text (already masked when masking was on)
    original = str(source.get("text") or "").strip()
    if original:
        title = "9. OCR TEXT USED (reading order, personal details masked)" \
            if _masked(report.get("privacy")) else "9. ORIGINAL OCR TEXT (reading order)"
        out += _section(title, ["  " + ln for ln in original.splitlines()])

    # Sign-off
    out += ["", "=" * 28, "REVIEW SIGN-OFF", "-" * 15]
    if approved:
        out.append(_wrap(f"Reviewed and approved by: {reviewer or 'reviewer'}. Approval was made in this "
                         "session only; this app stores no record.", ""))
    else:
        out += ["Reviewed by: ______________________    Date: ______________"]
    out += ["", _wrap("This report was prepared from OCR text. It is not a diagnosis or medical advice. "
                      "A clinician must verify every medication, dose, date and negation against the "
                      "original image.", "")]
    return "\n".join(out) + "\n"
