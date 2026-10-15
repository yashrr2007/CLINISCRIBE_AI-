"""Run with:  python -m pytest -q      (all names and numbers below are invented)"""
import json

from phi_masker import mask_filename, mask_text

# An invented referral note in the style of the real ones (scattered identifiers + medical content).
NOTE = """Date: 08/07/2024
Name: Ramesh Kumar
Address: 12/3 Gandhi Road, Anna Nagar, Chennai 600040
Ph: 98765 43210
AR NO: 5089836
Dear Dr Harkishan Singh Sir,
Referring Mr Ramesh Kumar aged 45 years for your kind consultation.
Ramesh complains of cough, no fever but breathlessness. S/o Suresh Kumar.
Indicate CKD (Raised s. creatinine 2.1 mg/dL, Blood urea 58 mg/dL)
Metformin 500 mg twice daily, aspirin 75 mg once daily. BP 120/80, dose 1-0-1.
Referred to Apollo Hospital. Email: ramesh.k@example.com  Aadhaar: 2345 6789 0123
Thanking you
Regards
kifal
DR. K. ANIL ROY
MBBS, PGDHA, DPH
REG. No. 58368"""

SECRETS = ["Ramesh", "Kumar", "Suresh", "Harkishan", "Singh", "Anil", "kifal", "Gandhi", "Anna Nagar",
           "Chennai", "600040", "98765", "43210", "5089836", "58368", "08/07/2024", "Apollo",
           "ramesh.k@example.com", "2345 6789 0123", "12/3"]
MEDICAL = ["cough", "no fever but breathlessness", "CKD", "creatinine 2.1 mg/dL", "Blood urea 58 mg/dL",
           "Metformin 500 mg twice daily", "aspirin 75 mg once daily", "BP 120/80", "dose 1-0-1",
           "aged 45 years", "MBBS, PGDHA, DPH"]


def test_full_note_removes_every_identifier_and_keeps_every_medical_fact():
    r = mask_text(NOTE)
    for s in SECRETS:
        assert s not in r.text, f"leaked: {s}"
    for m in MEDICAL:
        assert m in r.text, f"lost medical content: {m}"


def test_summary_and_notes_never_contain_originals():
    r = mask_text(NOTE)
    blob = json.dumps({"counts": r.counts, "uncertain": r.uncertain})
    for s in SECRETS:
        assert s not in blob, f"original value in summary: {s}"
    assert r.counts["PATIENT_NAME"] >= 2 and r.counts["PHONE"] == 1 and r.counts["DATE"] == 1
    assert r.counts["ADDRESS"] >= 1 and r.counts["FACILITY"] == 1 and r.counts["EMAIL"] == 1
    assert r.counts["GOV_ID"] == 1 and r.counts["RECORD_ID"] >= 2 and r.counts["RELATIVE"] == 1
    assert r.total == sum(r.counts.values())


def test_masking_twice_changes_nothing():
    once = mask_text(NOTE)
    assert mask_text(once.text).text == once.text


# ---- names ----------------------------------------------------------------------
def test_same_person_same_tag_and_different_person_new_tag():
    t = mask_text("Referring Mr Ramesh Kumar for review. Ramesh has cough.\n"
                  "Dr Anil Roy saw him. Later Dr Meena Iyer reviewed. Dr Roy agreed.").text
    assert t.count("[PATIENT_NAME]") == 2
    assert t.count("[CLINICIAN]") == 2 and t.count("[CLINICIAN_2]") == 1
    assert "Roy" not in t and "Iyer" not in t and "Ramesh" not in t


def test_titles_labels_and_relatives():
    cases = {
        "MR PARDAL ACKNOOR AGED 18": "MR [PATIENT_NAME] AGED 18",
        "Patient name: Lakshmi Devi": "Patient name: [PATIENT_NAME]",
        "Baby of Anitha, born preterm": "Baby of [PATIENT_NAME], born preterm",
        "D/o Ravi Shankar": "D/o [RELATIVE]",
        "Father name: Ravi Shankar": "Father name: [RELATIVE]",
        "Consultant: Dr S Rao": "Consultant: Dr [CLINICIAN]",
        "Patient: Ramesh Kumar": "Patient: [PATIENT_NAME]",
    }
    for src, want in cases.items():
        assert mask_text(src).text == want, src


def test_clinical_shorthand_is_not_mistaken_for_names():
    for line in ("Patient: Stable", "Patient ID is not shown", "w/o fever", "C/o Chest pain", "H/o Diabetes", "R/O Dengue", "Mother: HTN",
                 "Father: Diabetes", "Drug name: Metformin", "Name: Creatinine"):
        assert mask_text(line).text == line, line


def test_name_wrapped_over_two_lines_is_flagged():
    r = mask_text("Referring Mr Ramesh\nKumar has a cough")
    assert any("rest of the name" in u for u in r.uncertain)
    # when the wrapped part sits right before "aged 40" it is now masked outright
    r = mask_text("Referring Mr Ramesh\nKumar aged 40")
    assert "Ramesh" not in r.text and "Kumar" not in r.text


def test_signature_line_after_regards_is_masked_and_flagged():
    r = mask_text("Please review.\nThanking you\nRegards\nkifal")
    assert r.text.endswith("[CLINICIAN]") and "kifal" not in r.text
    assert any("signature" in u for u in r.uncertain)
    assert mask_text("Thanking you\nStart metformin").text.endswith("Start metformin")


# ---- numbers ----------------------------------------------------------------------
def test_phones_emails_government_ids():
    r = mask_text("Mobile: +91-9876543210. Landline 044-24567890. Contact: 98765 43210.\n"
                  "PAN ABCDE1234F, Aadhaar 234567890123, mail a.b@x.co.in")
    for s in ("9876543210", "24567890", "98765", "ABCDE1234F", "234567890123", "a.b@x.co.in"):
        assert s not in r.text, s
    assert r.counts["PHONE"] == 3 and r.counts["GOV_ID"] == 2 and r.counts["EMAIL"] == 1


def test_record_labels_keep_the_label_and_hide_the_number():
    assert mask_text("AR NO: 5089836").text == "AR NO: [RECORD_ID]"
    assert mask_text("REG. No. 58368").text == "REG. No. [RECORD_ID]"
    assert mask_text("UHID: AB-12345").text == "UHID: [RECORD_ID]"
    assert mask_text("Policy No: P/2024/7788").text == "Policy No: [GOV_ID]"
    assert mask_text("Lab No: Pending").text == "Lab No: Pending"


def test_dates_dob_and_age():
    t = mask_text("DOB: 12/03/1980. Seen on 8 Jul 2024 and 15-10-2026. Aged 95, patient 101 years old.").text
    assert "[DOB]" in t and t.count("[DATE]") == 2 and t.count("[AGE_90+]") == 2
    assert "1980" not in t and "2024" not in t and "95" not in t and "101" not in t
    kept = mask_text("DOB: 12/03/1980. Seen 08/07/2024.", mask_dates=False).text
    assert "[DOB]" in kept and "08/07/2024" in kept


def test_clinical_numbers_are_left_alone():
    for line in ("BP 120/80 mmHg", "Dose 1-0-1 for 10 days", "Platelets 150000", "HbA1c 7.2 %",
                 "Aged 45 years", "Creatinine 2.1 mg/dL on 2 occasions", "May 10 mg daily"):
        assert mask_text(line).text == line, line
    assert any("6 or more digits" in u for u in mask_text("Platelets 150000").uncertain)


# ---- places and organisations ----------------------------------------------------
def test_address_and_facility_rules_do_not_eat_clinical_text():
    t = mask_text("Road traffic accident 2 days ago, flat affect\nadmitted to hospital\n"
                  "Seen at City General Hospital\nFlat 4B, Park Street, Kochi").text
    assert "Road traffic accident 2 days ago, flat affect" in t and "admitted to hospital" in t
    assert "City General" not in t and "Park Street" not in t
    assert "Seen at [FACILITY]" in t


def test_edge_cases():
    assert mask_text("").text == "" and mask_text("   ").text == "   "
    assert mask_text(None).text == ""
    assert mask_text("No fever, no cough").text == "No fever, no cough"
    assert mask_text("Line one\r\nMr Arun Nair\r\n").text == "Line one\nMr [PATIENT_NAME]\n"


def test_filename_masking():
    assert mask_filename("Ramesh_Kumar_prescription.JPG") == "uploaded_note.jpg"
    assert mask_filename("scan") == "uploaded_note" and mask_filename(None) == "uploaded_note"


def test_email_is_masked_as_one_unit_and_tags_follow_reading_order():
    r = mask_text("Email: ramesh.k@example.com\nMr Ramesh Kumar\nDear Dr Meena Iyer,\n"
                  "Regards\nkifal\nDR. K. ANIL ROY")
    assert r.text.splitlines()[0] == "Email: [EMAIL]" and r.counts["EMAIL"] == 1
    assert "Dr [CLINICIAN]," in r.text and "DR. [CLINICIAN_2]" in r.text
    assert r.text.splitlines()[4] == "[CLINICIAN_3]"


# ---- gaps found while testing (all names and numbers invented) --------------------------
def test_ids_written_in_spaced_groups_are_fully_masked():
    for text, secret in [("Aadhaar: 1234 5678 9012", "5678"), ("Aadhaar No 1234 5678 9012", "9012"),
                         ("UID 2345 6789 0123", "6789"), ("UHID: AB 123 456", "456"),
                         ("Policy No: LIC 123 456 789", "789"), ("Voter ID: ABC 1234567", "1234567"),
                         ("PAN abcde1234f", "abcde1234f")]:
        assert secret not in mask_text(text).text, text


def test_bare_name_before_aged_is_masked_without_a_title():
    for text in ["Pardal Acknoor aged 18 years", "Pt Ravi Iyer aged 40", "Referring Pardal Acknoor aged 18",
                 "PARDAL ACKNOOR aged: 18"]:
        out = mask_text(text).text
        assert "Pardal" not in out and "Acknoor" not in out and "Ravi" not in out and "Iyer" not in out \
            and "PARDAL" not in out and "ACKNOOR" not in out, out
        assert "aged" in out.lower() and "[PATIENT_NAME]" in out, out
    # the same name later in the note is masked too
    assert "Acknoor" not in mask_text("Pardal Acknoor aged 18.\nAcknoor has cough.").text


def test_before_aged_rule_leaves_clinical_words_alone():
    for text in ["Male aged 45", "Female aged 30 years", "Child aged 5", "Boy aged 7", "Patient aged 60",
                 "CKD aged 70", "Known case of Diabetes aged 60", "Elderly woman aged 80"]:
        assert mask_text(text).text == text, text
