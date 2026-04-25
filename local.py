#!/usr/bin/env python3
"""
Medical Report OCR + JSON Extractor with Beautiful Modern UI.
Consumer-friendly interface with real-time progress updates.

Provenance markers:
- ASR-MOCR-sigil-lattice
- medocr-amruthsrivatsan-verde
- AS-initials-trace-4812
"""

import argparse
import csv
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import asyncio

# GPU optimization
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.7)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}

OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OCR_TIMEOUT: int = int(os.getenv("OCR_TIMEOUT", "300"))
LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "300"))
SAVE_OCR_TEXT_FILES: bool = env_bool("SAVE_OCR_TEXT_FILES", False)

LLM_NUM_CTX: int = int(os.getenv("LLM_NUM_CTX", "8192"))
LLM_NUM_PREDICT: int = int(os.getenv("LLM_NUM_PREDICT", "4096"))

DOCTR_TRY_COLUMN_SPLIT: bool = env_bool("DOCTR_TRY_COLUMN_SPLIT", True)
DOCTR_Y_TOL: float = float(os.getenv("DOCTR_Y_TOL", "0.012"))
GROUND_WINDOW_LINES: int = int(os.getenv("GROUND_WINDOW_LINES", "16"))

ENABLE_PREPROCESSING: bool = env_bool("ENABLE_PREPROCESSING", True)
ENABLE_FALLBACK_OCR: bool = env_bool("ENABLE_FALLBACK_OCR", True)
ENABLE_PDF_TEXT_LAYER: bool = env_bool("ENABLE_PDF_TEXT_LAYER", True)
ENABLE_DOCTR_OCR: bool = env_bool("ENABLE_DOCTR_OCR", True)
ENABLE_CANDIDATE_ROWS: bool = env_bool("ENABLE_CANDIDATE_ROWS", True)

MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MAX_FILES_PER_REQUEST: int = int(os.getenv("MAX_FILES_PER_REQUEST", "10"))
ALLOWED_EXTENSIONS: set = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("medical_ocr_ui")

# Non-functional provenance strings used to identify verbatim or near-verbatim copies.
PROVENANCE_MARKERS = (
    "ASR-MOCR-sigil-lattice",
    "medocr-amruthsrivatsan-verde",
    "AS-initials-trace-4812",
)

if torch.cuda.is_available():
    logger.info("GPU: %s", torch.cuda.get_device_name(0))

# Schemas
class PatientDetails(BaseModel):
    full_name: Optional[str] = None
    age: Optional[str] = None
    sex: Optional[str] = None
    date_of_birth: Optional[str] = None
    patient_id: Optional[str] = None
    other_ids: Optional[Any] = None

class ReportMetadata(BaseModel):
    report_type: Optional[str] = None
    report_date: Optional[str] = None
    sample_collection_date: Optional[str] = None
    referring_doctor: Optional[str] = None
    lab_name: Optional[str] = None
    lab_address: Optional[str] = None
    hospital_name: Optional[str] = None
    accession_number: Optional[str] = None

class ReferenceRangeEntry(BaseModel):
    population: Optional[str] = None
    range: Optional[str] = None

class TestEntry(BaseModel):
    panel: Optional[str] = None
    test_name: Optional[str] = None
    method: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    reference_ranges: Optional[List[ReferenceRangeEntry]] = None
    flag: Optional[str] = None
    specimen_type: Optional[str] = None
    remarks: Optional[str] = None
    page_index: Optional[int] = None
    computed_flag: Optional[str] = None
    position_label: Optional[str] = None
    position_ratio: Optional[float] = None
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    value_num: Optional[float] = None

class PageMetadata(BaseModel):
    page_index: Optional[int] = None
    patient_details: PatientDetails = Field(default_factory=PatientDetails)
    report_metadata: ReportMetadata = Field(default_factory=ReportMetadata)
    tests: List[TestEntry] = Field(default_factory=list)
    flags: List[str] = Field(default_factory=list)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS

@dataclass
class TempResources:
    temp_dir: str
    def cleanup(self) -> None:
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning("Failed to cleanup temp dir %s: %s", self.temp_dir, e)

@dataclass
class CandidateTestRow:
    panel: Optional[str] = None
    test_name: Optional[str] = None
    method: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    reference_ranges: Optional[List[Dict[str, Optional[str]]]] = None
    specimen_type: Optional[str] = None
    row_text: Optional[str] = None
    source: Optional[str] = None

    def as_prompt_line(self, index: int) -> str:
        parts = [f"ROW {index}"]
        for key in ["panel", "test_name", "method", "value", "unit", "reference_range", "specimen_type"]:
            val = getattr(self, key)
            if val:
                parts.append(f"{key}={val}")
        if self.reference_ranges:
            rr = "; ".join(
                f"{r.get('population')}: {r.get('range')}" if r.get("population") else str(r.get("range"))
                for r in self.reference_ranges
                if r.get("range")
            )
            if rr:
                parts.append(f"reference_ranges=[{rr}]")
        if self.row_text:
            parts.append(f"evidence={self.row_text}")
        return " | ".join(parts)

def load_report_inputs(inputs: List[str]) -> Tuple[List[str], TempResources]:
    if not inputs:
        raise ValueError("No input paths provided")
    temp_dir = tempfile.mkdtemp(prefix="medical_inputs_")
    staged: List[str] = []
    for s in inputs:
        p = Path(s)
        if not p.exists():
            logger.error("Input path does not exist: %s", p)
            continue
        if not (is_pdf(p) or is_image(p)):
            logger.warning("Unsupported file type (skipping): %s", p)
            continue
        staged.append(str(p))
    if not staged:
        raise RuntimeError("No valid PDF/image inputs were provided.")
    return staged, TempResources(temp_dir=temp_dir)

# Preprocessing
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available")

def detect_skew_angle(image):
    if not CV2_AVAILABLE:
        return 0.0
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return 0.0
    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        if abs(angle) < 45:
            angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess_image_for_ocr(image_path_or_pil):
    if not CV2_AVAILABLE or not ENABLE_PREPROCESSING:
        return image_path_or_pil
    try:
        if isinstance(image_path_or_pil, str):
            img = cv2.imread(image_path_or_pil)
        else:
            img = cv2.cvtColor(np.array(image_path_or_pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angle = detect_skew_angle(gray)
        if abs(angle) > 0.5:
            logger.info("Deskewing by %.2f degrees", angle)
            gray = rotate_image(gray, -angle)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return Image.fromarray(binary)
    except Exception as e:
        logger.warning("Preprocessing failed: %s", e)
        return image_path_or_pil

# Fallback OCR
_TESSERACT_AVAILABLE = None
_PADDLEOCR_AVAILABLE = None
_PADDLEOCR_MODEL = None

def check_tesseract():
    global _TESSERACT_AVAILABLE
    if _TESSERACT_AVAILABLE is None:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            _TESSERACT_AVAILABLE = True
        except Exception:
            _TESSERACT_AVAILABLE = False
    return _TESSERACT_AVAILABLE

def check_paddleocr():
    global _PADDLEOCR_AVAILABLE
    if _PADDLEOCR_AVAILABLE is None:
        try:
            from paddleocr import PaddleOCR
            _PADDLEOCR_AVAILABLE = True
        except Exception:
            _PADDLEOCR_AVAILABLE = False
    return _PADDLEOCR_AVAILABLE

def ocr_with_tesseract(image_path_or_pil):
    import pytesseract
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil)
    else:
        img = image_path_or_pil
    config = '--psm 6'
    text = pytesseract.image_to_string(img, config=config)
    return text.strip()

def ocr_with_paddleocr(image_path_or_pil):
    global _PADDLEOCR_MODEL
    from paddleocr import PaddleOCR
    if _PADDLEOCR_MODEL is None:
        try:
            _PADDLEOCR_MODEL = PaddleOCR(
                lang="en",
                use_textline_orientation=True,
                use_doc_orientation_classify=True,
            )
        except TypeError:
            _PADDLEOCR_MODEL = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    tmp_path: Optional[str] = None
    try:
        if not isinstance(image_path_or_pil, str):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tmp_path = tf.name
            image_path_or_pil.save(tmp_path)
            image_path_or_pil = tmp_path
        result = _PADDLEOCR_MODEL.ocr(image_path_or_pil)
        lines: List[str] = []
        if result:
            first = result[0] if isinstance(result, list) else result
            if isinstance(first, dict):
                lines.extend(str(t) for t in first.get("rec_texts", []) if str(t).strip())
            elif isinstance(first, list):
                for line in first:
                    try:
                        lines.append(str(line[1][0]))
                    except Exception:
                        continue
        return "\n".join(lines)
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def ocr_image_with_fallback(image_path_or_pil) -> str:
    if not ENABLE_FALLBACK_OCR:
        if ENABLE_PREPROCESSING and CV2_AVAILABLE:
            processed = preprocess_image_for_ocr(image_path_or_pil)
            return ocr_with_tesseract(processed) if check_tesseract() else "[OCR not available]"
        return "[OCR fallback disabled]"
    
    if check_paddleocr():
        try:
            processed = preprocess_image_for_ocr(image_path_or_pil)
            text = ocr_with_paddleocr(processed)
            if len(text.strip()) > 50:
                return text
        except Exception as e:
            logger.warning(f"PaddleOCR failed: {e}")
    
    if check_tesseract():
        try:
            processed = preprocess_image_for_ocr(image_path_or_pil)
            text = ocr_with_tesseract(processed)
            if len(text.strip()) > 50:
                return text
        except Exception as e:
            logger.warning(f"Tesseract failed: {e}")
    
    return "[OCR failed]"

def ocr_pdf_with_fallback(pdf_path: str) -> List[str]:
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        logger.error(f"pdf2image failed: {e}")
        return ["[PDF conversion failed]"]
    page_texts = []
    for img in images:
        text = ocr_image_with_fallback(img)
        page_texts.append(text)
    return page_texts

def _text_quality(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]", text or ""))

def _is_useful_text(text: str, min_chars: int = 80) -> bool:
    return _text_quality(text) >= min_chars

def _run_text_command(cmd: List[str], timeout: int = OCR_TIMEOUT) -> str:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return ""
    except Exception as e:
        logger.debug("Text command failed (%s): %s", " ".join(cmd), e)
        return ""
    if proc.returncode != 0:
        logger.debug("Text command returned %s: %s", proc.returncode, proc.stderr.strip())
        return ""
    return proc.stdout.replace("\x0c", "").strip()

def _pdf_page_count(pdf_path: str) -> Optional[int]:
    out = _run_text_command(["pdfinfo", pdf_path], timeout=30)
    m = re.search(r"^Pages:\s*(\d+)\s*$", out, flags=re.MULTILINE)
    return int(m.group(1)) if m else None

def _pdftotext_page(pdf_path: str, page_number: int, mode: str) -> str:
    flag = "-layout" if mode == "layout" else "-raw"
    return _run_text_command(
        ["pdftotext", flag, "-enc", "UTF-8", "-nopgbrk", "-f", str(page_number), "-l", str(page_number), pdf_path, "-"]
    )

def _pdf_text_layer_pages(pdf_path: str, mode: str = "layout") -> List[str]:
    if not ENABLE_PDF_TEXT_LAYER:
        return []
    page_count = _pdf_page_count(pdf_path)
    if not page_count:
        return []
    pages = [_pdftotext_page(pdf_path, i, mode) for i in range(1, page_count + 1)]
    useful = sum(1 for p in pages if _is_useful_text(p))
    if useful:
        logger.info("Poppler %s text layer: %s/%s useful page(s)", mode, useful, page_count)
    return pages

def _pdftotext_tsv_page(pdf_path: str, page_number: int) -> List[Dict[str, Any]]:
    if not ENABLE_PDF_TEXT_LAYER or not ENABLE_CANDIDATE_ROWS:
        return []
    out = _run_text_command(["pdftotext", "-tsv", "-f", str(page_number), "-l", str(page_number), pdf_path, "-"])
    if not out:
        return []
    words: List[Dict[str, Any]] = []
    try:
        reader = csv.DictReader(io.StringIO(out), delimiter="\t")
        for row in reader:
            if row.get("level") != "5":
                continue
            text = (row.get("text") or "").strip()
            if not text or text == "###PAGE###":
                continue
            try:
                x0 = float(row.get("left") or 0.0)
                y0 = float(row.get("top") or 0.0)
                width = float(row.get("width") or 0.0)
                height = float(row.get("height") or 0.0)
                conf = float(row.get("conf") or -1.0)
            except ValueError:
                continue
            if width <= 0 or height <= 0:
                continue
            x1 = x0 + width
            y1 = y0 + height
            words.append({
                "text": text,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "xc": (x0 + x1) / 2.0,
                "yc": (y0 + y1) / 2.0,
                "h": max(1e-6, height),
                "confidence": conf,
            })
    except Exception as e:
        logger.debug("Failed parsing pdftotext TSV for %s page %s: %s", pdf_path, page_number, e)
        return []
    return words

def _pdf_tsv_words_pages(pdf_path: str) -> List[List[Dict[str, Any]]]:
    if not ENABLE_CANDIDATE_ROWS:
        return []
    page_count = _pdf_page_count(pdf_path)
    if not page_count:
        return []
    return [_pdftotext_tsv_page(pdf_path, i) for i in range(1, page_count + 1)]

def _pdf_candidate_blocks_pages(pdf_path: str) -> List[str]:
    word_pages = _pdf_tsv_words_pages(pdf_path)
    blocks: List[str] = []
    for words in word_pages:
        rows = _extract_table_rows_from_words(words, source_name="POPPLER_TSV")
        blocks.append(_rows_to_candidate_block(rows, "POPPLER_TSV"))
    return blocks

def _plain(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def _line_text(line_words: List[Dict[str, Any]]) -> str:
    return " ".join(str(w["text"]) for w in sorted(line_words, key=lambda w: w["x0"])).strip()

def _find_header_line(lines: List[List[Dict[str, Any]]]) -> Tuple[Optional[int], Dict[str, float]]:
    best_i: Optional[int] = None
    best_score = 0
    best_anchors: Dict[str, float] = {}
    for i, line in enumerate(lines):
        text = _plain(_line_text(line))
        if not text:
            continue
        has_test = any(k in text for k in [
            "investigation", "test name", "test", "analysis", "examination",
            "description", "component", "analyte", "parameter", "particulars",
        ])
        has_value = any(k in text for k in ["observed value", "result", "results", "value"])
        has_unit = any(k in text for k in ["unit", "units"])
        has_ref = any(k in text for k in ["biological ref", "bio ref", "reference", "interval", "range"])
        has_specimen = "specimen" in text
        score = sum([has_test, has_value, has_unit, has_ref, has_specimen])
        if not (score >= 3 and has_value and (has_test or has_ref)):
            continue

        anchors: Dict[str, float] = {}
        words = sorted(line, key=lambda w: w["x0"])
        for j, w in enumerate(words):
            token = _plain(str(w["text"]))
            x = float(w["x0"])
            if token in {
                "investigation", "test", "tests", "parameter", "particulars",
                "analysis", "examination", "description", "component", "analyte",
            } and "test_name" not in anchors:
                anchors["test_name"] = x
            if token in {"observed", "result", "results", "value"} and "value" not in anchors:
                anchors["value"] = x
            if token in {"unit", "units"} and "unit" not in anchors:
                anchors["unit"] = x
            if token in {"biological", "bio", "reference", "ref", "interval", "range"} and "reference_range" not in anchors:
                anchors["reference_range"] = x
            if token == "specimen" and "specimen_type" not in anchors:
                anchors["specimen_type"] = x
        if "test_name" not in anchors and words:
            anchors["test_name"] = float(words[0]["x0"])
        if "value" in anchors and "reference_range" in anchors and anchors["reference_range"] < anchors["value"]:
            anchors["reference_range"] = max(float(w["x0"]) for w in words if _plain(str(w["text"])) in {"ref", "interval", "range", "reference"})
        if score > best_score:
            best_i = i
            best_score = score
            best_anchors = anchors
    return best_i, best_anchors

def _column_boundaries(anchors: Dict[str, float]) -> List[Tuple[str, float, float]]:
    ordered = sorted(
        ((name, x) for name, x in anchors.items() if name in {"test_name", "value", "unit", "reference_range", "specimen_type"}),
        key=lambda item: item[1],
    )
    if not ordered:
        return []
    cols: List[Tuple[str, float, float]] = []

    def boundary_before_next(x: float, next_x: float) -> float:
        gap = max(0.0, next_x - x)
        return next_x - max(gap * 0.10, 0.015 if next_x <= 2.0 else 4.0)

    for i, (name, x) in enumerate(ordered):
        left = -float("inf") if i == 0 else boundary_before_next(ordered[i - 1][1], x)
        right = float("inf") if i == len(ordered) - 1 else boundary_before_next(x, ordered[i + 1][1])
        cols.append((name, left, right))
    return cols

def _assign_line_to_columns(line: List[Dict[str, Any]], columns: List[Tuple[str, float, float]]) -> Dict[str, str]:
    buckets: Dict[str, List[str]] = {name: [] for name, _, _ in columns}
    for w in sorted(line, key=lambda ww: ww["x0"]):
        xc = float(w["xc"])
        assigned = None
        for name, left, right in columns:
            if left <= xc < right:
                assigned = name
                break
        if assigned is None and columns:
            finite_cols = [(name, left, right) for name, left, right in columns if abs(left) != float("inf") and abs(right) != float("inf")]
            if finite_cols:
                assigned = min(finite_cols, key=lambda c: abs(((c[1] + c[2]) / 2.0) - xc))[0]
            else:
                assigned = min(columns, key=lambda c: abs(float(c[1] if c[1] != -float("inf") else c[2]) - xc))[0]
        if assigned:
            buckets.setdefault(assigned, []).append(str(w["text"]))
    return {k: " ".join(v).strip() for k, v in buckets.items() if " ".join(v).strip()}

def _looks_like_method(text: str) -> bool:
    return bool(re.search(r"\bmethod\b|^\s*\([^)]{2,80}\)\s*$", text or "", flags=re.IGNORECASE))

def _clean_method_text(text: str) -> str:
    text = re.sub(r"^\s*method\s*[:\-]?\s*", "", text or "", flags=re.IGNORECASE).strip()
    text = text.strip("() ")
    return text

def _is_stop_table_line(text: str) -> bool:
    return bool(re.search(
        r"\b(comments?|verified by|verified date|important instructions|end of report|page \d+ of \d+|"
        r"total bilirubin in neonates|alp in paediatric|reference range\s+iu/l|teitz)\b",
        text or "",
        flags=re.IGNORECASE,
    ))

def _is_panel_line(text: str, cols: Dict[str, str]) -> bool:
    if not text or cols.get("value") or cols.get("unit") or cols.get("reference_range"):
        return False
    if _looks_like_method(text):
        return False
    if re.search(r"\d", text):
        return False
    alpha = re.sub(r"[^A-Za-z]+", "", text)
    if len(alpha) < 6:
        return False
    titleish = sum(1 for w in re.findall(r"[A-Za-z]+", text) if w[:1].isupper())
    words = re.findall(r"[A-Za-z]+", text)
    return text.upper() == text or (words and titleish >= max(1, len(words) - 1))

def _is_value_like(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    if re.search(r"\d", text):
        return True
    norm = re.sub(r"[^a-z]+", "", text.lower())
    return norm in {
        "nil", "negative", "positive", "pos", "neg", "absent", "present",
        "detected", "notdetected", "reactive", "nonreactive", "trace",
    }

def _parse_reference_entries(text: Optional[str]) -> List[Dict[str, Optional[str]]]:
    if not text:
        return []
    cleaned = re.sub(r"\s+", " ", text).strip(" ;,")
    if not cleaned:
        return []
    entries: List[Dict[str, Optional[str]]] = []
    range_piece = r"(?:[<>]?\s*\d+(?:\.\d+)?\s*(?:[-–]|\bto\b)\s*[<>]?\s*\d+(?:\.\d+)?\s*%?|[<>]?\s*\d+(?:\.\d+)?\s*%?|negative|positive|not\s+detected|detected|non[-\s]?reactive|reactive)"
    pattern = re.compile(
        rf"(?P<pop>[A-Za-z0-9><][A-Za-z0-9 /<>.,()-]{{0,45}}?)\s*:\s*(?P<range>{range_piece})",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(cleaned):
        pop = re.sub(r"\s+", " ", match.group("pop")).strip(" :-")
        rr = re.sub(r"\s+", " ", match.group("range")).strip().replace("–", "-")
        if rr:
            entries.append({"population": pop or None, "range": rr})
    if entries:
        return entries
    if re.search(range_piece, cleaned, flags=re.IGNORECASE):
        entries.append({"population": None, "range": cleaned.replace("–", "-")})
    return entries

def _append_reference(row: CandidateTestRow, ref_text: Optional[str]) -> None:
    entries = _parse_reference_entries(ref_text)
    if not entries:
        return
    row.reference_ranges = row.reference_ranges or []
    existing = {(r.get("population"), r.get("range")) for r in row.reference_ranges}
    for entry in entries:
        key = (entry.get("population"), entry.get("range"))
        if key not in existing:
            row.reference_ranges.append(entry)
            existing.add(key)
    if row.reference_ranges:
        adult = next((r for r in row.reference_ranges if r.get("population") and "adult" in r["population"].lower()), None)
        primary = adult or (row.reference_ranges[0] if not row.reference_range else None)
        if primary:
            row.reference_range = f"{primary.get('population')}: {primary.get('range')}" if primary.get("population") else primary.get("range")

def _rows_to_candidate_block(rows: List[CandidateTestRow], source_name: str) -> str:
    if not rows:
        return ""
    lines = [
        f"Candidate rows from {source_name} table geometry. Prefer these rows for test/value/unit/reference/specimen alignment, but verify against OCR text.",
    ]
    for i, row in enumerate(rows, start=1):
        lines.append(row.as_prompt_line(i))
    return "\n".join(lines)

def _extract_table_rows_from_words(words: List[Dict[str, Any]], source_name: str = "WORD_GEOMETRY") -> List[CandidateTestRow]:
    if not ENABLE_CANDIDATE_ROWS or len(words) < 12:
        return []
    lines = _cluster_words_into_lines(words)
    header_i, anchors = _find_header_line(lines)
    columns = _column_boundaries(anchors)
    if header_i is None or len(columns) < 3 or "test_name" not in anchors or "value" not in anchors:
        return []

    rows: List[CandidateTestRow] = []
    current_panel: Optional[str] = None
    current_row: Optional[CandidateTestRow] = None
    blank_like = 0
    for line in lines[header_i + 1:]:
        line_text = _line_text(line)
        if not line_text:
            blank_like += 1
            continue
        if _is_stop_table_line(line_text):
            break
        cols = _assign_line_to_columns(line, columns)
        test_text = cols.get("test_name", "").strip()
        value_text = cols.get("value", "").strip()
        unit_text = cols.get("unit", "").strip()
        ref_text = cols.get("reference_range", "").strip()
        specimen_text = cols.get("specimen_type", "").strip()

        if current_row and (_looks_like_method(test_text) or _looks_like_method(line_text)) and not _is_value_like(value_text):
            method = _clean_method_text(test_text if _looks_like_method(test_text) else line_text)
            if method:
                current_row.method = method
            _append_reference(current_row, ref_text)
            continue

        if _is_panel_line(line_text, cols):
            current_panel = line_text.strip()
            current_row = None
            continue

        if current_row and not value_text and ref_text:
            _append_reference(current_row, ref_text)
            continue

        if test_text and _is_value_like(value_text):
            row = CandidateTestRow(
                panel=current_panel,
                test_name=re.sub(r"\s+", " ", test_text).strip(),
                value=value_text,
                unit=unit_text or None,
                specimen_type=specimen_text or None,
                row_text=line_text,
                source=source_name,
            )
            _append_reference(row, ref_text)
            rows.append(row)
            current_row = row
            blank_like = 0
            continue

        if current_row and test_text and not value_text and re.fullmatch(r"\([^)]{2,100}\)", test_text.strip()):
            current_row.method = _clean_method_text(test_text)
            continue

        blank_like += 1
        if blank_like > 12 and rows:
            break

    return rows

def _dedupe_sources(sources: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    seen: set = set()
    for name, text in sources:
        text = (text or "").strip()
        if not text:
            continue
        is_candidate_block = name.startswith("STRUCTURED_CANDIDATE_ROWS")
        if not is_candidate_block and not _is_useful_text(text):
            continue
        key = normalize_spaces(text)[:2000]
        if key in seen:
            continue
        seen.add(key)
        out.append((name, text))
    return out

def _combine_ocr_sources(page_index: int, sources: List[Tuple[str, str]]) -> str:
    sources = _dedupe_sources(sources)
    if not sources:
        return "[OCR failed]"
    parts = [
        f"PAGE {page_index + 1}",
        "Use STRUCTURED_CANDIDATE_ROWS first for test/value/unit/reference/specimen alignment when present. Use POPPLER_LAYOUT next because it comes from the PDF text layer and preserves columns.",
    ]
    for name, text in sources:
        parts.append(f"\n--- OCR_SOURCE: {name} ---\n{text.strip()}")
    return "\n".join(parts).strip()

# docTR OCR
_DOCTR_MODEL = None

def get_doctr_model():
    global _DOCTR_MODEL
    if _DOCTR_MODEL is not None:
        return _DOCTR_MODEL
    logger.info("Loading docTR OCR model...")
    try:
        from doctr.models import ocr_predictor
    except Exception as e:
        logger.error(f"docTR import failed: {e}")
        _DOCTR_MODEL = None
        return None
    try:
        _DOCTR_MODEL = ocr_predictor(pretrained=True, assume_straight_pages=True, straighten_pages=True)
        logger.info("Loaded docTR OCR predictor")
        return _DOCTR_MODEL
    except TypeError:
        try:
            _DOCTR_MODEL = ocr_predictor(pretrained=True)
            logger.info("Loaded docTR OCR predictor (basic)")
            return _DOCTR_MODEL
        except Exception as e:
            logger.error(f"Failed to load docTR: {e}")
            _DOCTR_MODEL = None
            return None
    except Exception as e:
        logger.error(f"Failed to load docTR: {e}")
        _DOCTR_MODEL = None
        return None

def _doctr_words_from_exported_page(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    words: List[Dict[str, Any]] = []
    for block in page.get("blocks") or []:
        for line in block.get("lines") or []:
            for w in line.get("words") or []:
                txt = (w.get("value") or "").strip()
                if not txt:
                    continue
                geom = w.get("geometry")
                if not geom or len(geom) != 2:
                    continue
                (x0, y0), (x1, y1) = geom
                x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
                xc = (x0 + x1) / 2.0
                yc = (y0 + y1) / 2.0
                h = max(1e-6, y1 - y0)
                words.append({"text": txt, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "xc": xc, "yc": yc, "h": h})
    return words

def _safe_median(xs: List[float], default: float) -> float:
    xs = [x for x in xs if isinstance(x, (int, float))]
    if not xs:
        return default
    xs = sorted(xs)
    n = len(xs)
    m = n // 2
    if n % 2:
        return float(xs[m])
    return float((xs[m - 1] + xs[m]) / 2.0)

def _cluster_words_into_lines(words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not words:
        return []
    med_h = _safe_median([w["h"] for w in words], default=0.015)
    y_tol = max(DOCTR_Y_TOL, 0.60 * med_h)
    words_sorted = sorted(words, key=lambda w: (w["yc"], w["x0"]))
    lines: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_y: Optional[float] = None
    for w in words_sorted:
        if current_y is None:
            current = [w]
            current_y = w["yc"]
            continue
        if abs(w["yc"] - current_y) <= y_tol:
            current.append(w)
            current_y = (current_y * (len(current) - 1) + w["yc"]) / float(len(current))
        else:
            current.sort(key=lambda ww: ww["x0"])
            lines.append(current)
            current = [w]
            current_y = w["yc"]
    if current:
        current.sort(key=lambda ww: ww["x0"])
        lines.append(current)
    lines.sort(key=lambda ln: sum(w["yc"] for w in ln) / max(1, len(ln)))
    return lines

def _detect_two_columns(words: List[Dict[str, Any]]) -> Optional[float]:
    if not DOCTR_TRY_COLUMN_SPLIT or len(words) < 60:
        return None
    xcs = sorted(w["xc"] for w in words)
    gaps = [b - a for a, b in zip(xcs, xcs[1:]) if b > a]
    median_gap = _safe_median(gaps, default=0.0)
    max_gap = 0.0
    split_x = None
    for a, b in zip(xcs, xcs[1:]):
        gap = b - a
        if gap > max_gap:
            max_gap = gap
            split_x = (a + b) / 2.0
    if split_x is None or max_gap < 0.10:
        return None
    if median_gap and max_gap < 8 * median_gap:
        return None
    if not (0.25 < split_x < 0.75):
        return None
    left = [w for w in words if w["xc"] < split_x]
    right = [w for w in words if w["xc"] >= split_x]
    if len(left) < 0.25 * len(words) or len(right) < 0.25 * len(words):
        return None
    return split_x

def _linearize_page_from_words(words: List[Dict[str, Any]]) -> str:
    if not words:
        return ""
    split_x = _detect_two_columns(words)
    columns = [words] if split_x is None else [
        [w for w in words if w["xc"] < split_x],
        [w for w in words if w["xc"] >= split_x],
    ]
    out_lines: List[str] = []
    for ci, col_words in enumerate(columns):
        lines = _cluster_words_into_lines(col_words)
        for ln in lines:
            s = " ".join(w["text"] for w in ln).strip()
            if s:
                out_lines.append(s)
        if ci == 0 and len(columns) > 1:
            out_lines.append("")
    text = "\n".join(out_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def _ocr_doctr_on_document(doc: Any) -> List[str]:
    page_texts, _ = _ocr_doctr_on_document_with_candidates(doc)
    return page_texts

def _ocr_doctr_on_document_with_candidates(doc: Any) -> Tuple[List[str], List[str]]:
    model = get_doctr_model()
    if model is None:
        return [], []
    result = model(doc)
    exported = result.export()
    pages = exported.get("pages") or []
    page_texts: List[str] = []
    candidate_blocks: List[str] = []
    for p in pages:
        words = _doctr_words_from_exported_page(p)
        txt = _linearize_page_from_words(words)
        page_texts.append(txt)
        rows = _extract_table_rows_from_words(words, source_name="DOCTR")
        candidate_blocks.append(_rows_to_candidate_block(rows, "DOCTR"))
    return page_texts, candidate_blocks

def run_doctr_ocr_for_inputs(input_paths: List[str]) -> Tuple[List[str], str]:
    from doctr.io import DocumentFile
    report_label = ", ".join(str(Path(p).name) for p in input_paths)
    all_page_texts: List[str] = []
    model = None
    doctr_failed = not ENABLE_DOCTR_OCR
    if doctr_failed:
        logger.warning("docTR disabled, using alternate OCR sources")
    for p in input_paths:
        path = Path(p)
        logger.info(f"OCR input: {path}")
        page_texts = []
        if is_pdf(path):
            layout_pages = _pdf_text_layer_pages(str(path), mode="layout")
            raw_pages = _pdf_text_layer_pages(str(path), mode="raw")
            candidate_blocks = _pdf_candidate_blocks_pages(str(path))
            if layout_pages and all(_is_useful_text(t) for t in layout_pages):
                for i, layout_text in enumerate(layout_pages):
                    sources: List[Tuple[str, str]] = []
                    if i < len(candidate_blocks):
                        sources.append(("STRUCTURED_CANDIDATE_ROWS", candidate_blocks[i]))
                    sources.append(("POPPLER_LAYOUT", layout_text))
                    if i < len(raw_pages):
                        sources.append(("POPPLER_RAW", raw_pages[i]))
                    page_texts.append(_combine_ocr_sources(len(all_page_texts) + i, sources))
                all_page_texts.extend(page_texts)
                continue

            doctr_pages: List[str] = []
            doctr_candidate_blocks: List[str] = []
            if not doctr_failed:
                try:
                    model = model or get_doctr_model()
                    if model is None:
                        doctr_failed = True
                        raise RuntimeError("docTR model unavailable")
                    doc = DocumentFile.from_pdf(str(path))
                    doctr_pages, doctr_candidate_blocks = _ocr_doctr_on_document_with_candidates(doc)
                except Exception as e:
                    logger.warning(f"docTR failed on {path}: {e}")
                    doctr_failed = True

            if doctr_pages and any(_is_useful_text(t) for t in doctr_pages):
                max_pages = max(len(doctr_pages), len(layout_pages), len(raw_pages), len(candidate_blocks), len(doctr_candidate_blocks))
                for i in range(max_pages):
                    sources: List[Tuple[str, str]] = []
                    if i < len(candidate_blocks):
                        sources.append(("STRUCTURED_CANDIDATE_ROWS", candidate_blocks[i]))
                    if i < len(doctr_candidate_blocks):
                        sources.append(("STRUCTURED_CANDIDATE_ROWS_DOCTR", doctr_candidate_blocks[i]))
                    if i < len(layout_pages):
                        sources.append(("POPPLER_LAYOUT", layout_pages[i]))
                    if i < len(raw_pages):
                        sources.append(("POPPLER_RAW", raw_pages[i]))
                    if i < len(doctr_pages):
                        sources.append(("DOCTR", doctr_pages[i]))
                    page_texts.append(_combine_ocr_sources(len(all_page_texts) + i, sources))
            if not page_texts or all(not _is_useful_text(t) for t in page_texts):
                logger.info(f"Using fallback OCR for PDF: {path}")
                fallback_pages = ocr_pdf_with_fallback(str(path))
                page_texts = [
                    _combine_ocr_sources(len(all_page_texts) + i, [("PADDLE_OR_TESSERACT", text)])
                    for i, text in enumerate(fallback_pages)
                ]
        elif is_image(path):
            sources: List[Tuple[str, str]] = []
            if not doctr_failed:
                try:
                    model = model or get_doctr_model()
                    if model is None:
                        doctr_failed = True
                        raise RuntimeError("docTR model unavailable")
                    img = Image.open(str(path)).convert("RGB")
                    doc = DocumentFile.from_images([img])
                    doctr_texts, doctr_candidate_blocks = _ocr_doctr_on_document_with_candidates(doc)
                    if doctr_candidate_blocks:
                        sources.append(("STRUCTURED_CANDIDATE_ROWS_DOCTR", doctr_candidate_blocks[0]))
                    if doctr_texts:
                        sources.append(("DOCTR", doctr_texts[0]))
                except Exception as e:
                    logger.warning(f"docTR failed on {path}: {e}")
                    doctr_failed = True
            if not sources or not any(_is_useful_text(text) for _, text in sources):
                logger.info(f"Using fallback OCR for image: {path}")
                sources.append(("PADDLE_OR_TESSERACT", ocr_image_with_fallback(str(path))))
            page_texts = [_combine_ocr_sources(len(all_page_texts), sources)]
        all_page_texts.extend(page_texts)
    return all_page_texts, report_label

# LLM utilities
def _ollama_chat_request(payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    url = OLLAMA_URL.rstrip("/")
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def clean_llm_output(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```json(.*?)```", r"\1", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(.*?)```", r"\1", text, flags=re.DOTALL)
    return text.strip()

def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty LLM output")
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")
    brace = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace += 1
        elif text[i] == "}":
            brace -= 1
            if brace == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Unbalanced braces")

def _coerce_llm_json_to_page_metadata(json_obj: Dict[str, Any], page_index: int) -> Dict[str, Any]:
    """Accept direct PageMetadata JSON or a full-report/pages JSON shape."""
    if any(k in json_obj for k in ("patient_details", "tests", "flags")) and "pages" not in json_obj:
        out = dict(json_obj)
        out.setdefault("page_index", page_index)
        out.setdefault("patient_details", {})
        out.setdefault("report_metadata", {})
        out.setdefault("tests", [])
        out.setdefault("flags", [])
        return out

    pages = json_obj.get("pages") or []
    page_obj: Dict[str, Any] = {}
    if isinstance(pages, list):
        for p in pages:
            if isinstance(p, dict) and p.get("page_index") == page_index:
                page_obj = p
                break
            if isinstance(p, dict) and p.get("page_number") in {page_index, page_index + 1}:
                page_obj = p
                break
        if not page_obj and pages and isinstance(pages[0], dict):
            page_obj = pages[0]

    rm_in = json_obj.get("report_metadata") or page_obj.get("report_metadata") or {}
    patient_in = page_obj.get("patient_details") or json_obj.get("patient_details") or {}
    if isinstance(rm_in, dict) and not patient_in and isinstance(rm_in.get("patient"), dict):
        patient_in = rm_in.get("patient") or {}

    dates = rm_in.get("dates") if isinstance(rm_in, dict) else {}
    dates = dates or {}
    report_metadata = {
        "report_type": rm_in.get("report_type") if isinstance(rm_in, dict) else None,
        "report_date": (dates.get("report_date") or rm_in.get("report_date") or rm_in.get("reported")) if isinstance(rm_in, dict) else None,
        "sample_collection_date": (dates.get("collection_date") or rm_in.get("sample_collection_date") or rm_in.get("collection")) if isinstance(rm_in, dict) else None,
        "referring_doctor": (rm_in.get("referring_doctor") or rm_in.get("doctor_name")) if isinstance(rm_in, dict) else None,
        "lab_name": (rm_in.get("lab_name") or rm_in.get("hospital_name")) if isinstance(rm_in, dict) else None,
        "lab_address": (rm_in.get("lab_address") or rm_in.get("hospital_address")) if isinstance(rm_in, dict) else None,
        "hospital_name": rm_in.get("hospital_name") if isinstance(rm_in, dict) else None,
        "accession_number": (rm_in.get("accession_number") or rm_in.get("lab_id")) if isinstance(rm_in, dict) else None,
    }

    tests_out: List[Dict[str, Any]] = []
    for item in page_obj.get("tests") or json_obj.get("tests") or []:
        if not isinstance(item, dict):
            continue
        rrs = item.get("reference_ranges")
        if isinstance(rrs, str):
            rrs = [{"population": None, "range": rrs}]
        tests_out.append({
            "panel": item.get("panel"),
            "test_name": item.get("test_name") or item.get("investigation"),
            "method": item.get("method"),
            "value": item.get("value") or item.get("observed_value") or item.get("result"),
            "unit": item.get("unit") or item.get("units"),
            "reference_range": item.get("reference_range") or item.get("bio_ref_interval"),
            "reference_ranges": rrs,
            "flag": item.get("flag"),
            "specimen_type": item.get("specimen_type") or item.get("specimen"),
            "remarks": item.get("remarks"),
            "page_index": page_index,
        })

    flags = page_obj.get("flags") or page_obj.get("comments_and_flags") or json_obj.get("flags") or []
    if not isinstance(flags, list):
        flags = [str(flags)]

    return {
        "page_index": page_index,
        "patient_details": {
            "full_name": patient_in.get("full_name") or patient_in.get("name"),
            "age": patient_in.get("age"),
            "sex": patient_in.get("sex") or patient_in.get("gender"),
            "date_of_birth": patient_in.get("date_of_birth") or patient_in.get("dob"),
            "patient_id": patient_in.get("patient_id") or patient_in.get("lab_id") or patient_in.get("uhid"),
            "other_ids": patient_in.get("other_ids"),
        },
        "report_metadata": report_metadata,
        "tests": tests_out,
        "flags": flags,
    }

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()

def normalize_alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def value_in_ocr(val: Optional[str], text_norm: str) -> bool:
    if not val:
        return False
    v = normalize_spaces(val)
    return bool(v) and v in text_norm

def loose_value_in_ocr(val: Optional[str], text: str) -> bool:
    if not val:
        return False
    if value_in_ocr(val, normalize_spaces(text)):
        return True
    v = normalize_alnum(val)
    t = normalize_alnum(text)
    return bool(v) and v in t

def grounded_field_value(field_name: str, val: str, local_text: str) -> bool:
    if loose_value_in_ocr(val, local_text):
        return True
    if field_name in {"value", "reference_range"}:
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", val.replace(",", ""))
        local_nums = set(re.findall(r"[-+]?\d+(?:\.\d+)?", local_text.replace(",", "")))
        return bool(nums) and all(n in local_nums for n in nums)
    return False

_KNOWN_HEADER_LABELS = [
    "Hosp. UHID", "Reg. Date", "Collection", "Collected", "Received", "Report Status",
    "Report", "Reported", "Print", "Age/Gender", "Gender", "Age", "Collected At",
    "Collected at", "Processed at", "Referral Dr", "Ref By", "A/c Status", "Lab No.",
    "Lab. Id", "Lab Id", "Name", "Bed",
]

def _clean_header_value(value: str) -> str:
    value = (value or "").strip()
    for label in _KNOWN_HEADER_LABELS:
        value = re.split(rf"\s{{2,}}{re.escape(label)}\s*:", value, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    value = re.sub(r"\s+", " ", value).strip()
    return value.strip(" :")

def _extract_header_value(text: str, label_patterns: List[str]) -> Optional[str]:
    for line in text.splitlines():
        for label in label_patterns:
            m = re.search(rf"(?:^|\s{{2,}}){label}\s*:\s*(.+)$", line, flags=re.IGNORECASE)
            if m:
                value = _clean_header_value(m.group(1))
                return value or None
    return None

def _split_age_gender(age_gender: str) -> Tuple[Optional[str], Optional[str]]:
    raw = _clean_header_value(age_gender)
    sex = None
    if re.search(r"\bfemale\b|\bF\b", raw, flags=re.IGNORECASE):
        sex = "Female"
    elif re.search(r"\bmale\b|\bM\b", raw, flags=re.IGNORECASE):
        sex = "Male"
    age = re.sub(r"\s*/?\s*\b(Female|Male|F|M)\b", "", raw, flags=re.IGNORECASE).strip(" /")
    return (age or None), sex

def apply_regex_metadata(page_meta: PageMetadata, page_text: str) -> PageMetadata:
    pd = page_meta.patient_details
    rm = page_meta.report_metadata

    lab_id = _extract_header_value(page_text, [r"Lab\.?\s*Id", r"Lab\s*No\.?"])
    if lab_id and not re.search(r"\d{1,2}[-/][A-Za-z0-9]{2,}", lab_id):
        pd.patient_id = pd.patient_id or lab_id
        rm.accession_number = lab_id

    name = _extract_header_value(page_text, [r"(?:Patient\s+)?Name(?!\s*(?:of|test|lab|hospital))"])
    if name:
        pd.full_name = re.sub(r"\s+\.$", ".", name).strip()

    age_gender = _extract_header_value(page_text, [r"Age\s*/\s*Gender", r"Age\s*/\s*Sex"])
    if age_gender:
        age, sex = _split_age_gender(age_gender)
        if age:
            pd.age = age
        if sex:
            pd.sex = sex
    else:
        age = _extract_header_value(page_text, [r"Age"])
        sex = _extract_header_value(page_text, [r"Gender", r"Sex"])
        if age:
            pd.age = age
        if sex:
            _, parsed_sex = _split_age_gender(sex)
            pd.sex = parsed_sex or sex

    report_date = _extract_header_value(page_text, [r"Report(?!\s*Status)", r"Reported"])
    if report_date and re.search(r"\d", report_date):
        rm.report_date = report_date
    collection_date = _extract_header_value(page_text, [r"Collection", r"Collected(?!\s*At)"])
    if collection_date and re.search(r"\d", collection_date):
        rm.sample_collection_date = collection_date
    ref_doc = _extract_header_value(page_text, [r"Referral\s*Dr", r"Ref\s*By"])
    if ref_doc and ref_doc != "-":
        rm.referring_doctor = ref_doc
    elif ref_doc == "-":
        rm.referring_doctor = None

    if rm.accession_number and re.search(r"\d{1,2}[-/][A-Za-z0-9]{2,}|\bHosp\.", rm.accession_number, flags=re.IGNORECASE):
        rm.accession_number = pd.patient_id
    return page_meta

def build_llm_prompt_for_page(page_text: str, page_index: int, report_context: Optional[Dict] = None) -> List[Dict[str, str]]:
    schema = {
        "page_index": page_index,
        "patient_details": {
            "full_name": "string or null",
            "age": "string or null",
            "sex": "string or null",
            "date_of_birth": "string or null",
            "patient_id": "string or null",
            "other_ids": "object/array/string or null",
        },
        "report_metadata": {
            "report_type": "string or null",
            "report_date": "string or null",
            "sample_collection_date": "string or null",
            "referring_doctor": "string or null",
            "lab_name": "string or null",
            "lab_address": "string or null",
            "hospital_name": "string or null",
            "accession_number": "string or null",
        },
        "tests": [
            {
                "panel": "string or null",
                "test_name": "string",
                "method": "string or null",
                "value": "string or null",
                "unit": "string or null",
                "reference_range": "string or null",
                "reference_ranges": [{"population": "string or null", "range": "string"}],
                "flag": "string or null",
                "specimen_type": "string or null",
                "remarks": "string or null",
                "page_index": page_index,
            }
        ],
        "flags": ["string"],
    }

    example_ocr = """--- OCR_SOURCE: STRUCTURED_CANDIDATE_ROWS ---
Candidate rows from POPPLER_TSV table geometry. Prefer these rows for test/value/unit/reference/specimen alignment, but verify against OCR text.
ROW 1 | panel=LIVER FUNCTION TEST | test_name=TOTAL PROTEIN | method=Biuret | value=7.1 | unit=g/dL | reference_range=Adults: 6.7-8.7 | specimen_type=Serum | reference_ranges=[Umbilical cord: 4.8-8.8; Premature: 3.6-6.0; New born: 4.6-7.0; Adults: 6.7-8.7] | evidence=TOTAL PROTEIN 7.1 g/dL Umbilical cord : 4.8-8.8 Serum
ROW 2 | panel=LIVER FUNCTION TEST | test_name=ALBUMIN | method=BCP | value=3.9 | unit=g/L | reference_range=Adults: 3.5-5.2 | specimen_type=Serum | reference_ranges=[Adults: 3.5-5.2; Newborn: 2.8-4.4] | evidence=ALBUMIN 3.9 g/L Adults : 3.5-5.2 Serum

--- OCR_SOURCE: POPPLER_LAYOUT ---
Investigation                Observed Value   Unit   Biological Ref. Interval        Specimen
LIVER FUNCTION TEST
TOTAL PROTEIN                7.1              g/dL   Umbilical cord : 4.8-8.8       Serum
Method:Biuret                                      Premature : 3.6-6.0
                                                   New born : 4.6-7.0
                                                   Adults : 6.7-8.7
ALBUMIN                      3.9              g/L    Adults : 3.5-5.2               Serum
Method:BCP                                         Newborn : 2.8-4.4"""

    example_json = {
        "page_index": page_index,
        "patient_details": {"full_name": None, "age": None, "sex": None},
        "report_metadata": {},
        "tests": [
            {
                "panel": "LIVER FUNCTION TEST",
                "test_name": "TOTAL PROTEIN",
                "method": "Biuret",
                "value": "7.1",
                "unit": "g/dL",
                "reference_range": "Adults : 6.7-8.7",
                "reference_ranges": [
                    {"population": "Umbilical cord", "range": "4.8-8.8"},
                    {"population": "Premature", "range": "3.6-6.0"},
                    {"population": "New born", "range": "4.6-7.0"},
                    {"population": "Adults", "range": "6.7-8.7"},
                ],
                "flag": None,
                "specimen_type": "Serum",
                "page_index": page_index,
            },
            {
                "panel": "LIVER FUNCTION TEST",
                "test_name": "ALBUMIN",
                "method": "BCP",
                "value": "3.9",
                "unit": "g/L",
                "reference_range": "Adults : 3.5-5.2",
                "reference_ranges": [
                    {"population": "Adults", "range": "3.5-5.2"},
                    {"population": "Newborn", "range": "2.8-4.4"},
                ],
                "flag": None,
                "specimen_type": "Serum",
                "page_index": page_index,
            },
        ],
        "flags": [],
    }

    system_msg = f"""You extract structured data from medical lab report OCR text.

The OCR text may contain multiple OCR_SOURCE blocks. Prefer STRUCTURED_CANDIDATE_ROWS when present for test/value/unit/reference/specimen alignment because it is built from word coordinates. Verify candidate rows against POPPLER_LAYOUT or the other OCR sources. Use POPPLER_LAYOUT next because it is the embedded PDF text layer and preserves columns. Use DOCTR, PADDLE, or TESSERACT blocks only to fill gaps or resolve obvious OCR mistakes.

Return ONLY one valid JSON object matching this schema:
{json.dumps(schema, indent=2)}

Rules:
1. Extract only values visible in the OCR text. Do not invent missing values.
2. Extract every real lab test row with a result value. Do not extract section headers, notes, reference-table-only rows, page numbers, or lab addresses as tests.
3. Keep method text in "method"; never merge it into "test_name".
4. Capture all population-specific reference ranges in "reference_ranges". For "reference_range", choose the Adult/Adults range when present for an adult patient; otherwise choose the most patient-relevant range if obvious; otherwise choose the first range printed for that test.
5. Preserve comparator ranges like "<0.9" and ">59".
6. Use "specimen_type" for specimen column values such as Serum, Plasma, EDTA, Urine.
7. "flag" is only a printed marker such as H, L, HH, LL, HIGH, LOW, or CRITICAL. Do not compute flags.
8. If candidate rows are present, use them as the row list unless the full OCR text clearly shows a real test row that candidates missed or split incorrectly.
9. Use null when uncertain.

Example OCR:
{example_ocr}

Example JSON:
{json.dumps(example_json, indent=2)}
"""

    context_note = ""
    if report_context:
        context_note = "\nREPORT CONTEXT FROM EARLIER PAGES:\n" + json.dumps(report_context, indent=2)

    user_msg = f"Extract page_index {page_index}.{context_note}\n\nOCR TEXT:\n{page_text}\n"
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]

# Validation & grading
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

def _parse_value_num(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    m = _NUM_RE.search(value.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def _parse_reference_range(rr: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not rr:
        return None, None
    rr_norm = normalize_spaces(rr)
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*[-–]\s*([-+]?\d+(?:\.\d+)?)", rr_norm)
    if m:
        try:
            a, b = float(m.group(1)), float(m.group(2))
            return (a, b) if a <= b else (b, a)
        except Exception:
            return None, None
    m = re.search(r"(?:<=|<)\s*([-+]?\d+(?:\.\d+)?)", rr_norm)
    if m:
        try:
            return None, float(m.group(1))
        except Exception:
            return None, None
    m = re.search(r"(?:>=|>)\s*([-+]?\d+(?:\.\d+)?)", rr_norm)
    if m:
        try:
            return float(m.group(1)), None
        except Exception:
            return None, None
    return None, None

def enrich_tests_with_computed_flags(page_meta: PageMetadata) -> PageMetadata:
    page_flags = list(page_meta.flags or [])
    for t in page_meta.tests:
        if not t.reference_range and t.reference_ranges:
            for rr in t.reference_ranges:
                if rr.range:
                    t.reference_range = f"{rr.population}: {rr.range}" if rr.population else rr.range
                    break
        v = _parse_value_num(t.value)
        lo, hi = _parse_reference_range(t.reference_range)
        t.value_num = v
        t.ref_low, t.ref_high = lo, hi
        if v is None:
            continue
        if lo is None and hi is not None:
            t.computed_flag = "HIGH" if v > hi else "NORMAL"
            continue
        if lo is not None and hi is None:
            t.computed_flag = "LOW" if v < lo else "NORMAL"
            continue
        if lo is None or hi is None or hi == lo:
            continue
        if v < lo:
            t.computed_flag = "LOW"
        elif v > hi:
            t.computed_flag = "HIGH"
        else:
            t.computed_flag = "NORMAL"
        t.position_ratio = max(0.0, min(1.0, (v - lo) / (hi - lo)))
        if t.position_ratio < 0.15:
            t.position_label = "VERY_LOW"
        elif t.position_ratio < 0.4:
            t.position_label = "LOW"
        elif t.position_ratio <= 0.6:
            t.position_label = "MID"
        elif t.position_ratio <= 0.85:
            t.position_label = "HIGH"
        else:
            t.position_label = "VERY_HIGH"
        if lo > 0 and hi > 0 and v > 5 * hi:
            msg = f"Suspicious value for {t.test_name or ''}: {v} vs range {lo}-{hi}".strip()
            page_flags.append(msg)
    page_meta.flags = sorted(set(page_flags))
    return page_meta

# Fuzzy matching & grounding
import difflib

_PARENS_RE = re.compile(r"\([^)]*\)")
_NONALNUM_RE = re.compile(r"[^a-z0-9]+")

def norm_key(s: str) -> str:
    s = s.lower().strip()
    s = _PARENS_RE.sub(" ", s)
    s = _NONALNUM_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def best_line_match_index(name: str, lines: List[str]) -> Optional[int]:
    nk = norm_key(name)
    if not nk:
        return None
    keys = [norm_key(l) for l in lines]
    for i, k in enumerate(keys):
        if nk and nk in k:
            return i
    toks = nk.split()
    if len(toks) >= 2:
        prefix2 = " ".join(toks[:2])
        for i, k in enumerate(keys):
            if prefix2 in k:
                return i
    best_i = None
    best_score = 0.0
    for i, k in enumerate(keys):
        if not k:
            continue
        score = difflib.SequenceMatcher(a=nk, b=k).ratio()
        if score > best_score:
            best_score = score
            best_i = i
    if best_i is not None and best_score >= 0.78:
        return best_i
    return None

NON_TEST_NAME_RE = re.compile(
    r"\b("
    r"alc status|report status|test report|test name results|units bio\. ref|page\s*\d+|"
    r"differential leucocyte count|absolute leucocyte count|comment|note|interpretation|"
    r"important instructions|end of report|method\s*:"
    r")\b", re.IGNORECASE
)

def is_probably_not_a_test(t: TestEntry) -> bool:
    name = (t.test_name or "").strip()
    if not name or NON_TEST_NAME_RE.search(name):
        return True
    if (not t.value or str(t.value).strip() == "") and \
       (not t.unit or str(t.unit).strip() == "") and \
       (not t.reference_range or str(t.reference_range).strip() == ""):
        return True
    return False

def clean_test_entry_units(t: TestEntry) -> None:
    if not t.value:
        return
    v = t.value.strip()
    v_lower = v.lower()
    if t.unit:
        u = t.unit.strip()
        u_lower = u.lower()
        if v_lower.endswith(" " + u_lower):
            v = v[: -len(u) - 1].strip()
        elif v_lower.endswith(u_lower):
            v = v[: -len(u)].strip()
    if (not t.unit) and v.endswith("%"):
        v = v[:-1].rstrip()
        t.unit = "%"
    t.value = v

def clean_test_entry(t: TestEntry) -> None:
    if t.test_name:
        name = re.sub(r"\bmethod\s*[:\-].*$", "", t.test_name, flags=re.IGNORECASE).strip()
        if t.method:
            method_norm = norm_key(t.method)
            name_norm = norm_key(name)
            if method_norm and method_norm in name_norm:
                name = re.sub(re.escape(t.method), "", name, flags=re.IGNORECASE).strip(" -:;")
        t.test_name = re.sub(r"\s+", " ", name).strip() or t.test_name
    if t.method:
        t.method = re.sub(r"^\s*method\s*[:\-]\s*", "", t.method, flags=re.IGNORECASE).strip()
    if t.reference_ranges == []:
        t.reference_ranges = None
    clean_test_entry_units(t)

def enforce_grounding(page_meta: PageMetadata, page_text: str) -> PageMetadata:
    lines = page_text.splitlines()
    flags: List[str] = list(page_meta.flags or [])
    
    pd = page_meta.patient_details
    for field_name in ["full_name", "age", "sex", "date_of_birth", "patient_id"]:
        val = getattr(pd, field_name)
        if val and not loose_value_in_ocr(val, page_text):
            setattr(pd, field_name, None)
            flags.append(f"auto_null_{field_name}")
    
    rm = page_meta.report_metadata
    for field_name in ["report_type", "report_date", "sample_collection_date", "referring_doctor",
                       "lab_name", "lab_address", "hospital_name", "accession_number"]:
        val = getattr(rm, field_name)
        if val and not loose_value_in_ocr(val, page_text):
            setattr(rm, field_name, None)
            flags.append(f"auto_null_{field_name}")
    
    grounded_tests: List[TestEntry] = []
    for t in page_meta.tests or []:
        if t.page_index is None:
            t.page_index = page_meta.page_index
        clean_test_entry(t)
        if is_probably_not_a_test(t):
            flags.append(f"dropped_non_test_{(t.test_name or '')!r}")
            continue
        if not t.test_name:
            flags.append("dropped_test_without_name")
            continue
        base = best_line_match_index(t.test_name, lines)
        if base is None:
            has_grounded_value = any(
                isinstance(getattr(t, fname, None), str)
                and grounded_field_value(fname, getattr(t, fname), page_text)
                for fname in ["value", "reference_range", "unit", "method", "specimen_type"]
            )
            if not has_grounded_value:
                flags.append(f"dropped_test_{t.test_name!r}")
                continue
            flags.append(f"ungrounded_name_{t.test_name!r}")
            local_text = page_text
        else:
            end = min(len(lines), base + max(1, GROUND_WINDOW_LINES))
            local_text = "\n".join(lines[base:end])
        for fname in ["value", "unit", "reference_range", "flag", "method", "specimen_type", "remarks"]:
            val = getattr(t, fname, None)
            if isinstance(val, str) and val.strip():
                if not grounded_field_value(fname, val, local_text):
                    setattr(t, fname, None)
                    flags.append(f"auto_null_{fname}_{t.test_name!r}")
        if t.reference_ranges:
            kept_ranges: List[ReferenceRangeEntry] = []
            for rr in t.reference_ranges:
                rr_text = f"{rr.population or ''} {rr.range or ''}".strip()
                if not rr.range or grounded_field_value("reference_range", rr_text, local_text):
                    kept_ranges.append(rr)
            t.reference_ranges = kept_ranges or None
        clean_test_entry(t)
        grounded_tests.append(t)
    
    if not grounded_tests:
        flags.append("no_tests_on_page")
    
    page_meta.tests = grounded_tests
    page_meta.flags = sorted(set(flags))
    return page_meta

def call_ollama_for_page(page_index: int, page_text: str, report_context: Optional[Dict] = None) -> PageMetadata:
    messages = build_llm_prompt_for_page(page_text, page_index, report_context)
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        content: str = ""
        try:
            logger.info(f"Calling Ollama for page {page_index} (attempt {attempt})...")
            payload = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0, "num_ctx": LLM_NUM_CTX, "num_predict": LLM_NUM_PREDICT},
            }
            data = _ollama_chat_request(payload, timeout=LLM_TIMEOUT)
            content = data.get("message", {}).get("content", "") if isinstance(data, dict) else ""
            content = clean_llm_output(content)
            json_obj = extract_json_from_text(content)
            coerced = _coerce_llm_json_to_page_metadata(json_obj, page_index)
            page_meta = PageMetadata.model_validate(coerced)
            page_meta.page_index = page_index
            for t in page_meta.tests:
                if t.page_index is None:
                    t.page_index = page_index
            page_meta = enforce_grounding(page_meta, page_text)
            page_meta = apply_regex_metadata(page_meta, page_text)
            page_meta = enrich_tests_with_computed_flags(page_meta)
            return page_meta
        except Exception as e:
            logger.warning(f"Ollama failed page {page_index} attempt {attempt}: {e}")
            messages = messages + [{"role": "user", "content": "Previous response invalid. Return ONLY valid JSON."}]
    logger.error(f"Ollama failed for page {page_index}")
    return apply_regex_metadata(
        PageMetadata(page_index=page_index, flags=[f"LLM_failed_after_{max_retries}_attempts"]),
        page_text,
    )

# FastAPI with beautiful UI
app = FastAPI(title="Medical Report OCR", version="2.0")

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Report OCR & Analysis</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            animation: fadeIn 0.6s ease-in;
        }
        
        .header h1 {
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.95;
        }
        
        .card {
            background: white;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 24px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            animation: slideUp 0.6s ease-out;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 12px;
            padding: 60px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            position: relative;
            overflow: hidden;
        }
        
        .upload-area:hover {
            border-color: #764ba2;
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        }
        
        .upload-area.dragover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-color: white;
        }
        
        .upload-area.dragover * {
            color: white !important;
        }
        
        .upload-icon {
            font-size: 4em;
            margin-bottom: 16px;
            color: #667eea;
            animation: bounce 2s infinite;
        }
        
        .upload-text {
            font-size: 1.3em;
            color: #2d3748;
            margin-bottom: 8px;
            font-weight: 600;
        }
        
        .upload-hint {
            font-size: 0.95em;
            color: #718096;
        }
        
        .file-input {
            display: none;
        }
        
        .selected-file {
            margin-top: 16px;
            padding: 12px 20px;
            background: #f0fff4;
            border: 1px solid #9ae6b4;
            border-radius: 8px;
            color: #22543d;
            font-weight: 500;
            animation: fadeIn 0.3s ease-in;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 16px 40px;
            font-size: 1.1em;
            font-weight: 600;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            display: inline-block;
            margin-top: 20px;
        }
        
        .btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
            box-shadow: 0 4px 15px rgba(229, 62, 62, 0.4);
            margin-left: 12px;
        }
        
        .btn-secondary:hover:not(:disabled) {
            box-shadow: 0 6px 20px rgba(229, 62, 62, 0.6);
        }
        
        .btn-reset {
            background: linear-gradient(135deg, #718096 0%, #4a5568 100%);
            box-shadow: 0 4px 15px rgba(113, 128, 150, 0.4);
        }
        
        .btn-reset:hover:not(:disabled) {
            box-shadow: 0 6px 20px rgba(113, 128, 150, 0.6);
        }
        
        .button-group {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            align-items: center;
            margin-top: 20px;
        }
        
        .status {
            margin-top: 20px;
            padding: 16px;
            border-radius: 8px;
            font-weight: 500;
            display: none;
            animation: fadeIn 0.3s ease-in;
        }
        
        .status.info {
            background: #ebf8ff;
            color: #2c5282;
            border: 1px solid #90cdf4;
        }
        
        .status.success {
            background: #f0fff4;
            color: #22543d;
            border: 1px solid #9ae6b4;
        }
        
        .status.error {
            background: #fff5f5;
            color: #742a2a;
            border: 1px solid #fc8181;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 12px;
            display: none;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.3s ease;
            animation: pulse 1.5s ease-in-out infinite;
        }
        
        .output-section {
            display: none;
            animation: fadeIn 0.6s ease-in;
        }
        
        .output-label {
            font-size: 1.2em;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .output-box {
            background: #1a202c;
            color: #e2e8f0;
            padding: 24px;
            border-radius: 12px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9em;
            line-height: 1.6;
            max-height: 600px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.3);
        }
        
        .output-box::-webkit-scrollbar {
            width: 12px;
        }
        
        .output-box::-webkit-scrollbar-track {
            background: #2d3748;
            border-radius: 6px;
        }
        
        .output-box::-webkit-scrollbar-thumb {
            background: #667eea;
            border-radius: 6px;
        }
        
        .json-key {
            color: #81e6d9;
        }
        
        .json-string {
            color: #fbd38d;
        }
        
        .json-number {
            color: #b794f4;
        }
        
        .json-boolean {
            color: #fc8181;
        }
        
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes bounce {
            0%, 100% {
                transform: translateY(0);
            }
            50% {
                transform: translateY(-10px);
            }
        }
        
        @keyframes spin {
            to {
                transform: rotate(360deg);
            }
        }
        
        @keyframes pulse {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.7;
            }
        }
        
        @keyframes float {
            0%, 100% {
                transform: translateY(0px);
            }
            50% {
                transform: translateY(-5px);
            }
        }
        
        .btn-reset {
            animation: float 3s ease-in-out infinite;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 8px;
        }
        
        .badge-success {
            background: #c6f6d5;
            color: #22543d;
        }
        
        .badge-info {
            background: #bee3f8;
            color: #2c5282;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Medical Report OCR & Analysis</h1>
            <p>Upload your medical report (PDF or Image) for instant OCR and structured data extraction</p>
        </div>
        
        <div class="card">
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📄</div>
                <div class="upload-text">Click to select or drag & drop your report</div>
                <div class="upload-hint">Supports PDF, PNG, JPG, JPEG, TIFF (Max 50MB)</div>
                <input type="file" id="fileInput" class="file-input" accept=".pdf,.png,.jpg,.jpeg,.tiff,.tif" multiple>
            </div>
            <div id="selectedFile" class="selected-file" style="display:none;"></div>
            <div class="button-group">
                <button id="processBtn" class="btn" style="display:none;">🚀 Process Report</button>
                <button id="cancelBtn" class="btn btn-secondary" style="display:none;">⛔ Cancel Processing</button>
                <button id="resetBtn" class="btn btn-reset" style="display:none;">🔄 Start Over</button>
            </div>
            <div id="status" class="status"></div>
            <div id="progressBar" class="progress-bar">
                <div id="progressFill" class="progress-fill"></div>
            </div>
        </div>
        
        <div id="ocrSection" class="card output-section">
            <div class="output-label">
                📝 Raw OCR Text
                <span class="badge badge-info">Extracted Text</span>
            </div>
            <div id="ocrOutput" class="output-box"></div>
        </div>
        
        <div id="jsonSection" class="card output-section">
            <div class="output-label">
                🔬 Structured JSON Data
                <span class="badge badge-success">Parsed Results</span>
            </div>
            <div id="jsonOutput" class="output-box"></div>
        </div>
    </div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const selectedFileDiv = document.getElementById('selectedFile');
        const processBtn = document.getElementById('processBtn');
        const cancelBtn = document.getElementById('cancelBtn');
        const resetBtn = document.getElementById('resetBtn');
        const statusDiv = document.getElementById('status');
        const progressBar = document.getElementById('progressBar');
        const progressFill = document.getElementById('progressFill');
        const ocrSection = document.getElementById('ocrSection');
        const jsonSection = document.getElementById('jsonSection');
        const ocrOutput = document.getElementById('ocrOutput');
        const jsonOutput = document.getElementById('jsonOutput');
        
        let selectedFiles = null;
        let abortController = null;
        let isProcessing = false;
        
        // Reset everything to initial state
        function resetAll() {
            selectedFiles = null;
            abortController = null;
            isProcessing = false;
            
            // Reset file input
            fileInput.value = '';
            selectedFileDiv.style.display = 'none';
            selectedFileDiv.textContent = '';
            
            // Hide buttons
            processBtn.style.display = 'none';
            cancelBtn.style.display = 'none';
            resetBtn.style.display = 'none';
            processBtn.disabled = false;
            
            // Hide status and progress
            statusDiv.style.display = 'none';
            statusDiv.textContent = '';
            progressBar.style.display = 'none';
            progressFill.style.width = '0%';
            
            // Hide outputs
            ocrSection.style.display = 'none';
            jsonSection.style.display = 'none';
            ocrOutput.textContent = '';
            jsonOutput.innerHTML = '';
            
            // Reset upload area
            uploadArea.classList.remove('dragover');
        }
        
        // Click to upload
        uploadArea.addEventListener('click', () => fileInput.click());
        
        // File selection
        fileInput.addEventListener('change', (e) => {
            selectedFiles = e.target.files;
            if (selectedFiles.length > 0) {
                const fileNames = Array.from(selectedFiles).map(f => f.name).join(', ');
                selectedFileDiv.textContent = `✅ Selected: ${fileNames}`;
                selectedFileDiv.style.display = 'block';
                processBtn.style.display = 'inline-block';
            }
        });
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            selectedFiles = e.dataTransfer.files;
            if (selectedFiles.length > 0) {
                const fileNames = Array.from(selectedFiles).map(f => f.name).join(', ');
                selectedFileDiv.textContent = `Selected: ${fileNames}`;
                selectedFileDiv.style.display = 'block';
                processBtn.style.display = 'inline-block';
            }
        });
        
        // Process button
        processBtn.addEventListener('click', async () => {
            if (!selectedFiles || selectedFiles.length === 0) {
                showStatus('error', 'Please select a file first');
                return;
            }
            
            isProcessing = true;
            processBtn.disabled = true;
            cancelBtn.style.display = 'inline-block';
            resetBtn.style.display = 'none';
            ocrSection.style.display = 'none';
            jsonSection.style.display = 'none';
            progressBar.style.display = 'block';
            progressFill.style.width = '30%';
            
            showStatus('info', '<div class="spinner"></div> Step 1/2: Running OCR on your document...');
            
            const formData = new FormData();
            for (let file of selectedFiles) {
                formData.append('files', file);
            }
            
            // Create abort controller for cancellation
            abortController = new AbortController();
            
            try {
                const response = await fetch('/api/process-report', {
                    method: 'POST',
                    body: formData,
                    signal: abortController.signal
                });
                
                if (!response.ok) {
                    throw new Error(await response.text());
                }
                
                progressFill.style.width = '70%';
                showStatus('info', '<div class="spinner"></div> Step 2/2: Extracting structured data with AI...');
                
                const data = await response.json();
                
                progressFill.style.width = '100%';
                
                // Display OCR text
                const ocrText = data.ocr_texts.join('\\n\\n========== PAGE BREAK ==========\\n\\n');
                ocrOutput.textContent = ocrText;
                ocrSection.style.display = 'block';
                
                // Display JSON with syntax highlighting
                const formattedJson = JSON.stringify(data.pages, null, 2);
                jsonOutput.innerHTML = syntaxHighlight(formattedJson);
                jsonSection.style.display = 'block';
                
                const testCount = data.pages.reduce((sum, p) => sum + (p.tests?.length || 0), 0);
                showStatus('success', ` Processing complete! Extracted ${data.pages.length} page(s) with ${testCount} test results.`);
                progressBar.style.display = 'none';
                
                // Show reset button after success
                cancelBtn.style.display = 'none';
                resetBtn.style.display = 'inline-block';
                
            } catch (error) {
                if (error.name === 'AbortError') {
                    showStatus('error', ' Processing cancelled by user');
                } else {
                    showStatus('error', ' Error: ' + error.message);
                }
                progressBar.style.display = 'none';
                cancelBtn.style.display = 'none';
                resetBtn.style.display = 'inline-block';
            } finally {
                isProcessing = false;
                processBtn.disabled = false;
                abortController = null;
            }
        });
        
        // Cancel button
        cancelBtn.addEventListener('click', () => {
            if (abortController && isProcessing) {
                abortController.abort();
                showStatus('info', 'Cancelling processing...');
                cancelBtn.disabled = true;
            }
        });
        
        // Reset button
        resetBtn.addEventListener('click', () => {
            resetAll();
            showStatus('info', ' Ready for a new report');
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 2000);
        });
        
        function showStatus(type, message) {
            statusDiv.className = `status ${type}`;
            statusDiv.innerHTML = message;
            statusDiv.style.display = 'block';
        }
        
        function syntaxHighlight(json) {
            json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
                let cls = 'json-number';
                if (/^"/.test(match)) {
                    if (/:$/.test(match)) {
                        cls = 'json-key';
                    } else {
                        cls = 'json-string';
                    }
                } else if (/true|false/.test(match)) {
                    cls = 'json-boolean';
                } else if (/null/.test(match)) {
                    cls = 'json-null';
                }
                return '<span class="' + cls + '">' + match + '</span>';
            });
        }
    </script>
</body>
</html>"""

@app.post("/api/process-report", response_class=JSONResponse)
async def api_process_report(files: List[UploadFile] = File(...)):
    """Process medical report: OCR + JSON extraction"""
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Max {MAX_FILES_PER_REQUEST} files per request.")

    temp_dir = tempfile.mkdtemp(prefix="medical_api_")
    paths: List[str] = []
    try:
        # Save uploaded files
        for f in files:
            suffix = Path(f.filename or "").suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
            out_path = Path(temp_dir) / f"upload_{len(paths)}{suffix}"
            size = 0
            with out_path.open("wb") as dst:
                while True:
                    chunk = await f.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_UPLOAD_SIZE_BYTES:
                        raise HTTPException(status_code=413, detail=f"File '{f.filename}' exceeds {MAX_UPLOAD_SIZE_MB} MB.")
                    dst.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail=f"File '{f.filename}' is empty.")
            paths.append(str(out_path))
        
        # Run OCR
        loop = asyncio.get_running_loop()
        staged, _ = await loop.run_in_executor(None, lambda: load_report_inputs(paths))
        ocr_texts, _ = await loop.run_in_executor(None, lambda: run_doctr_ocr_for_inputs(staged))
        
        # Extract JSON with LLM
        pages_meta: List[PageMetadata] = []
        report_context: Dict[str, Any] = {}
        
        for i, page_text in enumerate(ocr_texts):
            try:
                context_snapshot = dict(report_context)
                page_meta = await loop.run_in_executor(
                    None,
                    lambda i=i, page_text=page_text, context_snapshot=context_snapshot: call_ollama_for_page(
                        i,
                        page_text,
                        context_snapshot,
                    ),
                )
                page_meta = enrich_tests_with_computed_flags(page_meta)
                if page_meta.patient_details.full_name:
                    report_context["patient_name"] = page_meta.patient_details.full_name
                if page_meta.patient_details.age:
                    report_context["patient_age"] = page_meta.patient_details.age
                if page_meta.patient_details.sex:
                    report_context["patient_sex"] = page_meta.patient_details.sex
                if page_meta.report_metadata.model_dump(exclude_none=True):
                    report_context["report_metadata"] = page_meta.report_metadata.model_dump(exclude_none=True)
                pages_meta.append(page_meta)
            except Exception as e:
                logger.error(f"LLM extraction failed for page {i}: {e}")
                # Add empty page metadata on error
                pages_meta.append(PageMetadata(
                    page_index=i,
                    flags=[f"extraction_error: {str(e)}"]
                ))
        
        return JSONResponse(content={
            "ocr_texts": ocr_texts,
            "pages": [p.model_dump(exclude_none=True) for p in pages_meta]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"API error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    finally:
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Medical Report OCR")
    print("="*60)
    print("\nStarting server...")
    print(" Open your browser to: http://localhost:8000")
    print("\n Features:")
    print("   • Multi-engine OCR (docTR + PaddleOCR + Tesseract)")
    print("   •structured data extraction")
    print("   • Conservative validation & grounding")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
