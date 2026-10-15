"""
Handwritten image -> text -> clinical draft, using Google Cloud Vision and Flask.

Files needed in the same folder:
    app.py                    (this file)
    clinical_summarizer.py    (the summarizer module)

Run:
    pip install flask google-cloud-vision python-dotenv
    python app.py
Then open http://127.0.0.1:5000 in your browser.

Credentials:
    Put your service-account key at credentials/vision-key.json
    or set GOOGLE_APPLICATION_CREDENTIALS in a .env file.

What it does:
    1. Reads handwriting with Google Cloud Vision (as before).
    2. Rebuilds the text in reading order (top to bottom, left to right).
    3. Builds a structured clinical DRAFT (medications, doses, dates, age, findings).
    4. Lets a reviewer edit, validate and approve it. Nothing is stored.
    5. Produces a plain-text FINAL REPORT that a person can read, copy or download.

Privacy: personal details (names, addresses, phone numbers, ID numbers, dates ...) are masked
automatically on every upload by phi_masker.py. Turn it off only with MASK_PHI=0 in .env.
NOTE: the photo itself is still sent to Google Cloud Vision to be read. Masking applies to
everything after that step (what you see, summarise, copy, download).
"""

import os
import re
import uuid
from collections import Counter

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from google.cloud import vision
from werkzeug.utils import secure_filename

from clinical_summarizer import (
    approve_report,
    build_clinical_note,
    format_report_text,
    ordered_text_from_vision,
    parse_summarize_payload,
    validate_report,
)
from phi_masker import mask_filename, mask_text

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "tif", "gif", "webp"}
MAX_UPLOAD_MB = 10

# Privacy masking is ON unless you set MASK_PHI=0 in .env. MASK_DATES=0 keeps ordinary dates
# (date of birth is still masked).
MASK_PHI = os.getenv("MASK_PHI", "1").strip() != "0"
MASK_DATES = os.getenv("MASK_DATES", "1").strip() != "0"

# Locate the credentials file. Make the path absolute so it works no matter
# which folder you start the server from.
cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.path.join(
    "credentials", "vision-key.json"
)
if not os.path.isabs(cred_path):
    cred_path = os.path.join(BASE_DIR, cred_path)

if os.path.exists(cred_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
else:
    print(f"[WARNING] Credentials file not found at: {cred_path}")
    print("          Put your key there or set GOOGLE_APPLICATION_CREDENTIALS in .env")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

_client = None


def get_client():
    """Create the Vision client once and reuse it."""
    global _client
    if _client is None:
        _client = vision.ImageAnnotatorClient()
    return _client


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def handwritten_to_text(image_path, language_hints=None):
    """
    Send an image to Google Cloud Vision.
    Returns (text, average_confidence, text_in_reading_order).
    """
    with open(image_path, "rb") as f:
        image = vision.Image(content=f.read())

    context = vision.ImageContext(language_hints=language_hints or [])
    response = get_client().document_text_detection(image=image, image_context=context)

    if response.error.message:
        raise RuntimeError(response.error.message)

    annotation = response.full_text_annotation
    text = annotation.text.strip() if annotation and annotation.text else ""

    # Average word confidence (0 to 1), when available
    confidences = [
        word.confidence
        for page in annotation.pages
        for block in page.blocks
        for paragraph in block.paragraphs
        for word in paragraph.words
    ]
    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else None

    ordered = ordered_text_from_vision(annotation) if text else ""

    return text, avg_conf, ordered


def clean_text(text):
    """Join broken lines and remove extra spaces for the NLP step."""
    return " ".join(text.split())


_MASK_TAG = re.compile(
    r"\[(PATIENT_NAME|RELATIVE|CLINICIAN|FACILITY|ADDRESS|PHONE|EMAIL|DOB|DATE|RECORD_ID|GOV_ID|"
    r"OTHER_ID|AGE_90\+)(?:_\d+)?\]"
)


def protect(text):
    """
    Remove personal details from `text` when masking is on.
    Returns (safe_text, privacy). `privacy` holds counts and notes, never the original values.
    """
    if not MASK_PHI:
        return text, {"masked": False, "counts": {}, "uncertain": []}
    result = mask_text(text, mask_dates=MASK_DATES)
    # Count the tags that are in the final text, not just the ones added by this pass: after
    # "Re-generate" the text is already masked, and the report must still show everything that was.
    counts = dict(sorted(Counter(_MASK_TAG.findall(result.text)).items()))
    return result.text, {"masked": True, "counts": counts, "uncertain": result.uncertain}


# ---------------------------------------------------------------------------
# Web page
# ---------------------------------------------------------------------------
PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Handwriting to Clinical Draft</title>
  <style>
    body { font-family: Arial, sans-serif; background:#f4f6f8; margin:0; padding:24px; }
    .card { max-width:720px; margin:0 auto 20px; background:#fff; padding:24px;
            border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.1); }
    h1 { margin-top:0; font-size:22px; }
    h2 { margin:0; font-size:19px; }
    h3 { margin:18px 0 6px; font-size:15px; }
    label { display:block; margin:14px 0 6px; font-weight:bold; }
    input[type=text], input[type=file] { width:100%; box-sizing:border-box; }
    input[type=text] { padding:8px; border:1px solid #ccc; border-radius:6px; }
    button { margin-top:16px; margin-right:8px; padding:10px 20px; border:0; border-radius:6px;
             background:#1a73e8; color:#fff; font-size:15px; cursor:pointer; }
    button.secondary { background:#e9eff7; color:#1c3555; }
    button:disabled { opacity:.55; cursor:not-allowed; }
    img { max-width:100%; margin-top:14px; border-radius:6px; display:none; }
    textarea { width:100%; height:200px; margin-top:6px; padding:10px;
               box-sizing:border-box; border:1px solid #ccc; border-radius:6px; font-family:inherit; }
    textarea.short { height:90px; }
    textarea.json { height:260px; font-family:Consolas,monospace; font-size:13px; }
    #status, .muted { color:#555; }
    #status { margin-top:12px; }
    .error { color:#c62828; }
    .hidden { display:none; }
    .head { display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap; }
    .pill { background:#fff1cf; color:#805400; border-radius:99px; padding:6px 10px;
            font-size:12px; font-weight:bold; }
    .pill.ok { background:#d8f5e4; color:#17603a; }
    .warning { border-left:4px solid #d99a18; background:#fff8e8; padding:9px 12px;
               margin:10px 0; border-radius:4px; font-size:14px; }
    .fact { border:1px solid #e1e6ee; border-radius:6px; padding:8px 10px; margin:6px 0; font-size:14px; }
    .fact small { color:#667085; display:block; margin-top:3px; }
    .tag { background:#fde8e8; color:#b42318; border-radius:4px; padding:1px 6px;
           font-size:11px; font-weight:bold; margin-left:6px; }
    .privacy { border-left:4px solid #2e7d32; background:#eef8f0; padding:9px 12px; margin:10px 0;
               border-radius:4px; font-size:14px; }
    .privacy.off { border-left-color:#c62828; background:#fdecec; }
    textarea.report { height:520px; font-family:Consolas,"Courier New",monospace; font-size:13px;
                      line-height:1.45; background:#fcfcfd; white-space:pre; overflow:auto; }
    .notice { background:#fffaf0; border:1px solid #f0dca9; border-radius:8px; padding:12px; font-size:14px; line-height:1.5; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Handwriting to Text</h1>
    <div id="maskBanner"></div>

    <label for="image">Choose an image of handwriting</label>
    <input type="file" id="image" accept="image/*">

    <label for="lang">Language hints (optional, e.g. en or en,ta)</label>
    <input type="text" id="lang" value="en">

    <img id="preview" alt="Preview">
    <button id="go">Extract text</button>
    <div id="status"></div>

    <textarea id="result" placeholder="Extracted text appears here" readonly></textarea>
  </div>

  <div class="card hidden" id="review">
    <div class="head">
      <h2>Clinical draft</h2>
      <span id="pill" class="pill">DRAFT - NOT APPROVED</span>
    </div>

    <label for="ordered">Text in reading order (editable)</label>
    <p class="muted" style="margin:0">Lines are rebuilt from word positions on the page. Correct any
      OCR mistakes here, then click Re-generate.</p>
    <textarea id="ordered"></textarea>
    <button id="regen" class="secondary">Re-generate draft</button>

    <div id="privacy"></div>
    <div id="flags"></div>

    <label for="summary">Concise summary (editable)</label>
    <textarea id="summary" class="short"></textarea>

    <h3>Findings</h3>
    <div id="facts"></div>

    <label for="reportJson">Structured report JSON (editable)</label>
    <textarea id="reportJson" class="json" spellcheck="false"></textarea>

    <label for="reviewer">Reviewer name / initials</label>
    <input type="text" id="reviewer" maxlength="120" placeholder="Required for approval">

    <button id="validate" class="secondary">Validate edits</button>
    <button id="showFinal" class="secondary">Show final report</button>
    <button id="approve">Approve reviewed draft</button>
    <p id="feedback" class="muted"></p>
  </div>

  <div class="card hidden" id="final">
    <div class="head">
      <h2>Final report</h2>
      <span id="finalPill" class="pill">DRAFT - NOT APPROVED</span>
    </div>
    <p class="muted">Readable version of the reviewed draft. It reflects your edits to the summary
      and the JSON. Approve the draft first if you want the sign-off line filled in.</p>
    <textarea id="finalText" class="report" readonly spellcheck="false"></textarea>
    <button id="copyReport" class="secondary">Copy report</button>
    <button id="downloadReport">Download as .txt</button>
    <p id="finalFeedback" class="muted"></p>
  </div>

  <div class="card notice">
    <strong>Safety:</strong> This tool only organizes OCR text. It does not diagnose, recommend
    treatment, or replace clinician verification. Handwriting OCR can change drug names, doses and
    dates. Check every field against the original image. Personal details are masked automatically, but the photo itself is sent to Google Cloud to be read, before any masking. Use real patient images only if your organization has approved that.
  </div>

  <script>
    const $ = id => document.getElementById(id);
    const fileInput = $("image"), preview = $("preview"), statusEl = $("status");
    const result = $("result"), btn = $("go");
    let latest = null;       // the draft currently on screen
    let sourceName = null;   // file name of the image
    let confidence = null;   // average OCR confidence

    fetch("/health").then(r => r.json()).then(h => {
      $("maskBanner").className = h.mask_phi ? "privacy" : "privacy off";
      $("maskBanner").textContent = h.mask_phi
        ? "Privacy masking is ON: names, addresses, phone numbers, ID numbers and dates are removed automatically."
        : "Privacy masking is OFF (MASK_PHI=0). Personal details will be visible.";
    }).catch(() => {});

    fileInput.addEventListener("change", () => {
      const f = fileInput.files[0];
      if (f) { preview.src = URL.createObjectURL(f); preview.style.display = "block"; }
    });

    async function post(url, body) {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      const data = await res.json().catch(() => ({ error: "Unexpected server response" }));
      if (!res.ok) throw new Error(data.error || "Request failed");
      return data;
    }

    // ---- show a draft on the page (text is always set with textContent / .value) ----
    const GROUPS = [
      ["patient_details", "Patient details"],
      ["medications", "Medications"],
      ["symptoms_and_findings", "Symptoms and findings"],
      ["negations", "Negations"],
      ["other_measurements", "Other measurements"],
      ["other_facts", "Other"]
    ];

    function describe(f) {
      let main = f.type.replace(/_/g, " ") + ": " + (Array.isArray(f.value) ? f.value.join(", ") : f.value);
      if (f.type === "medication") main += "  |  dose: " + (f.dose || "not found") + "  |  action: " + f.action;
      return main;
    }

    function showDraft(draft) {
      latest = draft;
      $("final").classList.add("hidden");   // an older final report would be out of date
      $("review").classList.remove("hidden");
      $("pill").textContent = "DRAFT - NOT APPROVED";
      $("pill").className = "pill";
      $("summary").value = draft.summary;
      $("reportJson").value = JSON.stringify(draft, null, 2);
      $("feedback").textContent = "Review every field against the original image before approval.";

      const p = draft.privacy || {};
      $("privacy").replaceChildren();
      const bar = document.createElement("div");
      const kinds = Object.entries(p.counts || {}).map(([k, v]) => k + " x" + v).join(", ");
      bar.className = p.masked ? "privacy" : "privacy off";
      bar.textContent = p.masked
        ? "Privacy masking ON. " + (kinds ? "Masked: " + kinds + "." : "Nothing needed masking.")
        : "Privacy masking is OFF. Personal details may be visible in this draft.";
      $("privacy").appendChild(bar);
      (p.uncertain || []).forEach(t => {
        const d = document.createElement("div");
        d.className = "warning";
        d.textContent = t;
        $("privacy").appendChild(d);
      });

      $("flags").replaceChildren();
      (draft.review_flags || []).forEach(t => {
        const d = document.createElement("div");
        d.className = "warning";
        d.textContent = t;
        $("flags").appendChild(d);
      });

      $("facts").replaceChildren();
      let any = false;
      GROUPS.forEach(([key, title]) => {
        const items = (draft.clinical_report || {})[key] || [];
        if (!items.length) return;
        any = true;
        const h = document.createElement("h3");
        h.textContent = title;
        $("facts").appendChild(h);
        items.forEach(f => {
          const box = document.createElement("div");
          box.className = "fact";
          box.appendChild(document.createTextNode(describe(f)));
          if (f.negated && f.type !== "negation") {
            const tag = document.createElement("span");
            tag.className = "tag";
            tag.textContent = "NEGATED";
            box.appendChild(tag);
          }
          const src = document.createElement("small");
          src.textContent = "from: " + f.source_text;
          box.appendChild(src);
          $("facts").appendChild(box);
        });
      });
      if (!any) {
        const p = document.createElement("p");
        p.className = "muted";
        p.textContent = "No structured facts were recognized. Review the text above.";
        $("facts").appendChild(p);
      }
    }

    // ---- 1. extract text from the image ----
    btn.addEventListener("click", async () => {
      const f = fileInput.files[0];
      if (!f) { statusEl.textContent = "Please choose an image first."; return; }

      const form = new FormData();
      form.append("image", f);
      form.append("lang", $("lang").value);

      btn.disabled = true;
      statusEl.className = "";
      statusEl.textContent = "Reading handwriting...";
      result.value = "";
      $("review").classList.add("hidden");
      $("final").classList.add("hidden");

      try {
        const res = await fetch("/extract-text", { method: "POST", body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Request failed");
        result.value = data.text || "(No text detected)";
        // use the server's masked name: the real file name may contain the patient's name
        sourceName = data.clinical_draft ? data.clinical_draft.source.name : null;
        confidence = data.confidence;
        statusEl.textContent = data.confidence !== null
          ? "Done. Average confidence: " + data.confidence
          : "Done.";
        if (data.clinical_draft) {
          $("ordered").value = data.ordered_text;
          showDraft(data.clinical_draft);
        }
      } catch (err) {
        statusEl.className = "error";
        statusEl.textContent = "Error: " + err.message;
      } finally {
        btn.disabled = false;
      }
    });

    // ---- 2. re-generate after the reviewer corrects the text ----
    $("regen").addEventListener("click", async () => {
      try {
        const d = await post("/api/summarize", {
          ocr_text: $("ordered").value,
          ocr_confidence: confidence,
          source_name: sourceName
        });
        // show the masked text, so an identifier typed in during review does not stay on screen
        if (d.source && typeof d.source.text === "string") $("ordered").value = d.source.text;
        showDraft(d);
      } catch (e) { $("feedback").textContent = "Could not generate: " + e.message; }
    });

    // ---- 3. validate and approve ----
    function currentReport() {
      let report;
      try {
        report = JSON.parse($("reportJson").value);
      } catch (e) {
        // The JSON box is damaged (cut short, or a typing mistake). Do not fail: rebuild it from
        // the last draft the server produced. The summary the reviewer edited is kept.
        if (!latest) throw e;
        report = JSON.parse(JSON.stringify(latest));
        $("reportJson").value = JSON.stringify(report, null, 2);
        $("feedback").textContent = "The JSON box had an error (" + e.message +
          "), so it was reset to the last generated draft. Your summary edits were kept.";
      }
      report.summary = $("summary").value;
      return report;
    }

    $("validate").addEventListener("click", async () => {
      try {
        const r = await post("/api/validate", currentReport());
        $("feedback").textContent = r.valid
          ? "Structure looks valid. Clinical accuracy still needs manual review."
          : r.issues.join(" ");
      } catch (e) { $("feedback").textContent = "Validation failed: " + e.message; }
    });

    $("approve").addEventListener("click", async () => {
      const name = $("reviewer").value.trim();
      if (!name) { $("feedback").textContent = "Enter reviewer name or initials before approval."; return; }
      try {
        const report = currentReport();
        const v = await post("/api/validate", report);
        if (!v.valid) { $("feedback").textContent = v.issues.join(" "); return; }
        const r = await post("/api/approve", { report: report, approved_by: name });
        $("reportJson").value = JSON.stringify(r.report, null, 2);
        $("pill").textContent = "APPROVED IN THIS SESSION";
        $("pill").className = "pill ok";
        $("feedback").textContent = r.report.approval_notice +
          " Copy or download the final report below; this app does not save it.";
        await showFinalReport();
      } catch (e) { $("feedback").textContent = "Could not approve: " + e.message; }
    });

    // ---- 4. readable final report ----
    async function showFinalReport() {
      try {
        const r = await post("/api/report-text", currentReport());
        $("finalText").value = r.text;
        const ok = $("pill").className.indexOf("ok") >= 0;
        $("finalPill").textContent = ok ? "APPROVED IN THIS SESSION" : "DRAFT - NOT APPROVED";
        $("finalPill").className = ok ? "pill ok" : "pill";
        $("finalFeedback").textContent = "";
        $("final").classList.remove("hidden");
        $("final").scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (e) { $("feedback").textContent = "Could not build the report: " + e.message; }
    }

    $("showFinal").addEventListener("click", showFinalReport);

    $("copyReport").addEventListener("click", async () => {
      const text = $("finalText").value;
      try {
        await navigator.clipboard.writeText(text);
      } catch (e) {                      // older browsers / non-secure pages
        $("finalText").select();
        document.execCommand("copy");
      }
      $("finalFeedback").textContent = "Report copied.";
    });

    $("downloadReport").addEventListener("click", () => {
      const blob = new Blob([$("finalText").value], { type: "text/plain;charset=utf-8" });
      const base = (sourceName || "note").replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9_-]+/g, "_");
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "clinical_report_" + base + ".txt";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(a.href);
      $("finalFeedback").textContent = "Downloaded " + a.download;
    });
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    # The main UI is cliniscribe.html (same folder as app.py). If that file is missing, fall
    # back to the simple built-in page so the app still opens.
    ui_path = os.path.join(BASE_DIR, "cliniscribe.html")
    if os.path.exists(ui_path):
        with open(ui_path, "rb") as f:
            resp = Response(f.read(), mimetype="text/html")
        resp.headers["Cache-Control"] = "no-store"   # always pick up the latest file
        return resp
    return Response(PAGE, mimetype="text/html")


@app.route("/classic")
def classic():
    # The original simple page, kept as a backup.
    # Sent as-is (not through Jinja) so the page's JavaScript braces are never misread.
    return Response(PAGE, mimetype="text/html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "credentials_found": os.path.exists(cred_path),
                    "mask_phi": MASK_PHI})


@app.route("/extract-text", methods=["POST"])
def extract_text():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded. Send it in the 'image' field."}), 400

    if not allowed_file(file.filename):
        return jsonify(
            {"error": "Unsupported file type. Use: " + ", ".join(sorted(ALLOWED_EXTENSIONS))}
        ), 400

    # Optional language hints, e.g. "en" or "en,ta"
    lang_field = request.form.get("lang", "").strip()
    hints = [h.strip() for h in lang_field.split(",") if h.strip()]

    # Save with a unique name so two uploads never overwrite each other
    safe_name = secure_filename(file.filename)
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{safe_name}")
    file.save(path)

    try:
        text, confidence, ordered = handwritten_to_text(path, hints)

        # Mask first. The raw Vision text comes back in scrambled block order, where a name can be
        # split into separate pieces, so it cannot be masked reliably: when masking is on, only the
        # reading-order text is ever sent to the browser.
        base = ordered if ordered.strip() else text
        safe_text, privacy = protect(base)
        source_name = mask_filename(file.filename) if MASK_PHI else safe_name

        # Clinical draft built from the (masked) text. If this step fails, the plain OCR result
        # is still returned.
        draft, draft_error = None, None
        if safe_text.strip():
            try:
                draft = build_clinical_note(safe_text, confidence, source_name, privacy)
            except Exception as e:  # keep OCR usable even if the summarizer has a problem
                draft_error = str(e)

        shown = safe_text if MASK_PHI else text
        return jsonify(
            {
                "text": shown,
                "cleaned_text": clean_text(shown),
                "confidence": confidence,
                "ordered_text": safe_text,
                "clinical_draft": draft,
                "clinical_draft_error": draft_error,
                "privacy": privacy,
            }
        )
    except Exception as e:
        return jsonify({"error": f"Vision API error: {e}"}), 500
    finally:
        # Delete the uploaded image after processing
        if os.path.exists(path):
            os.remove(path)


# ---- clinical summarizer API (same endpoints as the original HT-24 service) ----
@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    try:
        text, confidence, name = parse_summarize_payload(request.get_json(silent=True))
        safe_text, privacy = protect(text)
        if MASK_PHI and name:
            name = mask_filename(name)
        return jsonify(build_clinical_note(safe_text, confidence, name, privacy))
    except ValueError as e:
        return jsonify({"error": str(e)}), 422


@app.route("/api/validate", methods=["POST"])
def api_validate():
    return jsonify(validate_report(request.get_json(silent=True)))


@app.route("/api/approve", methods=["POST"])
def api_approve():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object with 'report' and 'approved_by'."}), 422
    try:
        return jsonify(approve_report(data.get("report"), data.get("approved_by")))
    except ValueError as e:
        return jsonify({"error": str(e)}), 422


@app.route("/api/report-text", methods=["POST"])
def api_report_text():
    """Readable plain-text version of a (possibly edited) report."""
    try:
        return jsonify({"text": format_report_text(request.get_json(silent=True))})
    except ValueError as e:
        return jsonify({"error": str(e)}), 422


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File too large. Maximum size is {MAX_UPLOAD_MB} MB."}), 413


# ---------------------------------------------------------------------------
# Start the server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # debug=True is for development only. Turn it off in production.
    app.run(host="127.0.0.1", port=5000, debug=True)
