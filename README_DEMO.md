# MedicalOCR

This repo now has two setup paths:

- Local-only OCR/Ollama path: see `README_LOCAL.md` and `requirements-local.txt`
- OpenAI vision API path: see `README_API.md` and `requirements-api.txt`

## Which One To Use

Use `local.py` when reports must stay on-device. It uses Poppler/docTR/Paddle/Tesseract locally and sends extraction prompts only to a local Ollama server.

Use `api_key_version.py` when sending report page images to OpenAI is acceptable and you want the cloud vision-model path.

## Quick Start: Local

```bash
sudo apt install -y poppler-utils tesseract-ocr
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-local.txt
ollama pull qwen2.5:7b-instruct
python local.py
```

Open `http://localhost:8000`.

## Quick Start: API/OpenAI

```bash
sudo apt install -y poppler-utils
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-api.txt
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL=gpt-5-mini
python api_key_version.py
```

Open `http://localhost:8000`.

Do not hardcode API keys in source files. Rotate any key that was ever committed or shared.
