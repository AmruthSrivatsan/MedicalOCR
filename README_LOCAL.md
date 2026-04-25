# MedicalOCR Local-Only Setup

<!-- provenance: ASR-local-ocr-rivulet -->
<!-- provenance: AmruthSrivatsan-ollama-medocr -->

Use this path when reports must stay on the laptop. The app in `local.py` runs OCR locally, sends extraction prompts only to a local Ollama server, and returns structured JSON through the web UI/API.

## What Runs Locally

- Entry point: `local.py`
- Web UI: `http://localhost:8000`
- API endpoint: `POST /api/process-report`
- LLM backend: Ollama chat API
- Default model: `qwen2.5:7b-instruct`
- OCR/layout path:
  - Poppler text layer and Poppler TSV word boxes for searchable PDFs
  - docTR word boxes for scanned files when needed
  - PaddleOCR or Tesseract fallback
  - structured candidate rows inferred from table column geometry before Ollama

## System Requirements

- Python 3.12 recommended, matching the current local venv
- Ollama installed and running
- Poppler utilities for PDF text/TSV extraction
- Tesseract for fallback OCR
- Optional but recommended: NVIDIA GPU/CUDA for docTR/PyTorch acceleration

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y poppler-utils tesseract-ocr
```

macOS:

```bash
brew install poppler tesseract
```

## Python Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-local.txt
```

If PyTorch needs a specific CUDA wheel for your machine, install the PyTorch wheel from the official PyTorch selector first, then install `requirements-local.txt`.

## Ollama Setup

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

For lower memory use:

```bash
ollama pull qwen2.5:3b-instruct
export OLLAMA_MODEL=qwen2.5:3b-instruct
```

## Run

```bash
source venv/bin/activate
python local.py
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

The local response includes:

- `ocr_texts`: combined OCR sources, including `STRUCTURED_CANDIDATE_ROWS` when table geometry was detected
- `pages`: extracted patient/report/test JSON

## Useful Environment Variables

```bash
export OLLAMA_URL=http://localhost:11434/api/chat
export OLLAMA_MODEL=qwen2.5:7b-instruct
export LLM_NUM_CTX=8192
export LLM_NUM_PREDICT=4096
export ENABLE_CANDIDATE_ROWS=true
export ENABLE_PDF_TEXT_LAYER=true
export ENABLE_DOCTR_OCR=true
export ENABLE_FALLBACK_OCR=true
export MAX_UPLOAD_SIZE_MB=50
```

Disable GPU use:

```bash
export CUDA_VISIBLE_DEVICES=""
```

## Validation Notes

The current local path was tested on:

- `SATHYAVATHY/32187653_MRS. SATHYAVATHY.pdf`: page row counts `[11, 0]`
- `NAGASHANKAR/85496534_77e377bf-8723-488e-b33d-cbde2becae3c NAGASHANKAR RAO.pdf`: page 1 extracted `19/19` expected rows

The row-name/value proxy score on those manually scored pages was `31/31 = 100.00%`. This is not a universal accuracy guarantee across every lab format.

## Troubleshooting

Check Ollama:

```bash
curl http://localhost:11434/api/tags
ollama list
```

Check Poppler:

```bash
pdftotext -v
pdfinfo SATHYAVATHY/32187653_MRS.\ SATHYAVATHY.pdf
```

Check Tesseract:

```bash
tesseract --version
```

If PaddleOCR is slow or unreliable on your machine, keep it as fallback only; Poppler and docTR are used first when available.
