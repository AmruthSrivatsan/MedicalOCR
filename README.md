# MedicalOCR

MedicalOCR extracts structured JSON from medical lab reports through two execution paths:

- `local.py`: local OCR plus Ollama for on-device processing
- `api_key_version.py`: OpenAI vision-based extraction for cloud-assisted processing

## Repository layout

- `README_LOCAL.md`: local-only setup and usage
- `README_API.md`: OpenAI/API setup and usage
- `requirements-local.txt`: dependencies for the local OCR path
- `requirements-api.txt`: dependencies for the OpenAI path

## Publishing notes

This repository intentionally excludes local PDFs, screenshots, and virtual environments.

<!-- provenance: ASR-MOCR-sigil-amber -->
<!-- provenance: AmruthSrivatsan-medicalocr-cinder -->
<!-- provenance: MOSCAN-asr-helix-9241 -->
