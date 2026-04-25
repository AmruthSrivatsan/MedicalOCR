# MedicalOCR API/OpenAI Setup

<!-- provenance: ASR-api-vision-kestrel -->
<!-- provenance: medicalocr-amruthsrivatsan-api -->

Use this path when you want the cloud model path. The app in `api_key_version.py` converts PDFs/images to page images and sends each page to the OpenAI Chat Completions API for structured extraction.

The older `claude_final.py` name is no longer present in this workspace; use `api_key_version.py`.

## What Runs Remotely

- Entry point: `api_key_version.py`
- Web UI: `http://localhost:8000`
- API endpoint: `POST /api/process-report`
- Model backend: OpenAI Chat Completions vision
- Default primary model: `gpt-5-mini`
- Default fallback model: `gpt-4o`, used only when a page fails extraction or page 1 comes back suspiciously empty
- OCR behavior: no local OCR text layer is required; pages are rendered to images and sent to the model

## System Requirements

- Python 3.12 recommended, matching the current local venv
- Poppler utilities for PDF-to-image conversion through `pdf2image`
- OpenAI API key with access to the configured vision model

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y poppler-utils
```

macOS:

```bash
brew install poppler
```

## Python Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-api.txt
```

## API Key Setup

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL=gpt-5-mini
export OPENAI_FALLBACK_MODEL=gpt-4o
```

Do not hardcode API keys in source files. If an old key was ever committed or shared, rotate it before using the API path.

## Run

```bash
source venv/bin/activate
python api_key_version.py
```

Open:

```text
http://localhost:8000
```

API example:

```bash
curl -X POST http://localhost:8000/api/process-report \
  -F "files=@SATHYAVATHY/32187653_MRS. SATHYAVATHY.pdf"
```

The API response includes:

- `pages`: extracted patient/report/test JSON
- `page_count`: number of rendered pages processed

## Useful Environment Variables

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL=gpt-5-mini
export OPENAI_FALLBACK_MODEL=gpt-4o
export OPENAI_USE_FALLBACK=true
export OPENAI_PRIMARY_RETRIES=2
export OPENAI_FALLBACK_RETRIES=2
export OPENAI_IMAGE_DETAIL=high
export OPENAI_REASONING_EFFORT=minimal
export OPENAI_JSON_MODE=true
export LLM_TIMEOUT=180
export LLM_MAX_TOKENS=4096
export PDF_DPI=150
export IMAGE_MAX_PX=2048
export MAX_CONCURRENT_VISION=20
export MAX_UPLOAD_SIZE_MB=50
export MAX_FILES_PER_REQUEST=10
```

## Cost Strategy

The API path now uses a model cascade:

1. Try `OPENAI_MODEL` first. Default: `gpt-5-mini`.
2. If JSON extraction fails, retries are exhausted, or page 1 is suspiciously empty, retry that page with `OPENAI_FALLBACK_MODEL`. Default: `gpt-4o`.
3. Keep `OPENAI_IMAGE_DETAIL=high` for dense lab tables unless benchmarking proves `low` does not reduce accuracy for your reports.

For the lowest cost test run:

```bash
export OPENAI_MODEL=gpt-5-mini
export OPENAI_FALLBACK_MODEL=
export OPENAI_USE_FALLBACK=false
```

For maximum accuracy:

```bash
export OPENAI_MODEL=gpt-4o
export OPENAI_USE_FALLBACK=false
```

## When To Use This Instead Of Local

Use the API path when:

- accuracy from a cloud vision model matters more than local-only processing
- the documents are visually complex and local OCR struggles
- sending report images to OpenAI is acceptable for your privacy/compliance constraints

Use `local.py` and `README_LOCAL.md` instead when the reports must stay on-device.

## Troubleshooting

Check the API key:

```bash
test -n "$OPENAI_API_KEY" && echo "OPENAI_API_KEY is set"
```

Check Poppler:

```bash
pdfinfo SATHYAVATHY/32187653_MRS.\ SATHYAVATHY.pdf
```

If the server returns `OPENAI_API_KEY not configured`, export the key in the same shell session where you run `python api_key_version.py`.
