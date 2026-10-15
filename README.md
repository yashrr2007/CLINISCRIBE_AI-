# CliniScribe AI

Turns a photo of a handwritten clinical note into a reviewed, structured report.

**Flow:** photo -> Google Cloud Vision OCR -> privacy masking -> rule-based clinical draft -> human review -> approval -> readable report (.txt).

## Run
```
pip install -r requirements.txt
copy .env.example .env      # then set your Google credentials (Windows)
python app.py
```
Open http://127.0.0.1:5000 in your browser (do not open the HTML file directly).
In Settings, click "Test Google Vision connection" to check your setup.

## Files
- `app.py` - Flask server and API
- `cliniscribe.html` - the web interface (plain HTML/CSS/JS, works offline)
- `phi_masker.py` - rule-based removal of personal details
- `clinical_summarizer.py` - rule-based clinical extraction and report text
- tests: `test_*.py` (run `python -m pytest -q`)

## Privacy and limits
- Nothing is stored: uploads are deleted after processing; results live in browser memory only.
- The photo itself is sent to Google Cloud Vision before masking. Do not use real patient data without your organisation's approval.
- Masking and extraction are rule-based (no ML model), so unusual phrasing can be missed. A person must review every report before it is used.
- Use invented notes for demos and tests.

## Security
Never commit `.env` or `credentials/`. If a key is ever committed or shared, delete it in Google Cloud Console and create a new one.
