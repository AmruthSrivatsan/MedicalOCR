#!/usr/bin/env python3
import asyncio
import base64
import io
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

# ── Config ────────────────────────────────────────────────────────────────────
def env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL:   str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_FALLBACK_MODEL: str = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o")
OPENAI_USE_FALLBACK: bool = env_bool("OPENAI_USE_FALLBACK", True)
OPENAI_IMAGE_DETAIL: str = os.getenv("OPENAI_IMAGE_DETAIL", "high")
OPENAI_REASONING_EFFORT: str = os.getenv("OPENAI_REASONING_EFFORT", "minimal")
OPENAI_JSON_MODE: bool = env_bool("OPENAI_JSON_MODE", True)

LLM_TIMEOUT:    int = int(os.getenv("LLM_TIMEOUT",    "180"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
OPENAI_PRIMARY_RETRIES: int = int(os.getenv("OPENAI_PRIMARY_RETRIES", "2"))
OPENAI_FALLBACK_RETRIES: int = int(os.getenv("OPENAI_FALLBACK_RETRIES", "2"))

# Resolution for PDF→PNG conversion.  150 dpi is a safer default for dense
# lab-report tables with small fonts. Lower to 100 only after benchmarking.
PDF_DPI: int = int(os.getenv("PDF_DPI", "150"))

# Maximum long-edge pixel size per page image sent to the API.
# Vision models tile images; keeping this ≤ 2048 avoids unnecessary tokens.
IMAGE_MAX_PX: int = int(os.getenv("IMAGE_MAX_PX", "2048"))

# Maximum simultaneous OpenAI vision calls.  Set high — 429s are handled by
# retry with Retry-After; this cap only protects against truly massive batches.
MAX_CONCURRENT_VISION: int = int(os.getenv("MAX_CONCURRENT_VISION", "20"))

MAX_UPLOAD_SIZE_MB:    int = int(os.getenv("MAX_UPLOAD_SIZE_MB",    "50"))
MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MAX_FILES_PER_REQUEST: int = int(os.getenv("MAX_FILES_PER_REQUEST", "10"))
ALLOWED_EXTENSIONS: set = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMAGE_EXTS:         set = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("medical_vision.quartz")

# ── Pydantic schemas (identical to OCR version for API compatibility) ─────────

class PatientDetails(BaseModel):
    full_name:   Optional[str] = None
    age:         Optional[str] = None
    sex:         Optional[str] = None
    date_of_birth: Optional[str] = None
    patient_id:  Optional[str] = None
    other_ids:   Optional[Any] = None

class ReportMetadata(BaseModel):
    report_type:            Optional[str] = None
    report_date:            Optional[str] = None
    sample_collection_date: Optional[str] = None
    referring_doctor:       Optional[str] = None
    lab_name:               Optional[str] = None
    lab_address:            Optional[str] = None
    hospital_name:          Optional[str] = None
    accession_number:       Optional[str] = None

class ReferenceRangeEntry(BaseModel):
    population: Optional[str] = None   # e.g. "Adults", "Newborn", "> 3 years"
    range:      Optional[str] = None   # e.g. "6.7-8.7", "<0.9"

class TestEntry(BaseModel):
    panel:            Optional[str]   = None
    test_name:        Optional[str]   = None
    method:           Optional[str]   = None   # e.g. "Biuret", "UV WITH P5P"
    value:            Optional[str]   = None
    unit:             Optional[str]   = None
    reference_range:  Optional[str]   = None   # first/primary — backward compat
    reference_ranges: Optional[List[ReferenceRangeEntry]] = None  # ALL ranges
    flag:             Optional[str]   = None
    specimen_type:    Optional[str]   = None
    remarks:          Optional[str]   = None
    page_index:       Optional[int]   = None
    computed_flag:    Optional[str]   = None
    position_label:   Optional[str]   = None
    position_ratio:   Optional[float] = None
    ref_low:          Optional[float] = None
    ref_high:         Optional[float] = None
    value_num:        Optional[float] = None

class PageMetadata(BaseModel):
    page_index:      Optional[int]    = None
    patient_details: PatientDetails   = Field(default_factory=PatientDetails)
    report_metadata: ReportMetadata   = Field(default_factory=ReportMetadata)
    tests:           List[TestEntry]  = Field(default_factory=list)
    flags:           List[str]        = Field(default_factory=list)


# ── Image utilities ────────────────────────────────────────────────────────────

def _resize_if_needed(img: Image.Image, max_px: int = IMAGE_MAX_PX) -> Image.Image:
    """Downscale so the longest edge ≤ max_px (preserving aspect ratio)."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_px:
        return img
    scale = max_px / longest
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _pil_to_base64_png(img: Image.Image) -> str:
    """Convert a PIL image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def file_to_page_images(path: Path) -> List[Image.Image]:
    """
    Convert a PDF or image file into a list of PIL Images (one per page).
    PDFs are rasterised at PDF_DPI; images are loaded directly.
    Guarded against decompression bombs (PIL default: 178M pixels).
    """
    # Raise the pixel limit slightly for large medical scans but cap it.
    Image.MAX_IMAGE_PIXELS = 300_000_000  # ~17000×17000 — well above any real scan
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(str(path), dpi=PDF_DPI)
            logger.info(f"PDF '{path.name}' → {len(pages)} page image(s) at {PDF_DPI} dpi")
            return pages
        except Exception as e:
            raise RuntimeError(f"pdf2image failed on '{path.name}': {e}") from e
    elif suffix in IMAGE_EXTS:
        img = Image.open(str(path)).convert("RGB")
        return [img]
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def pages_to_base64(pages: List[Image.Image]) -> List[str]:
    """Resize and base64-encode each page image."""
    result = []
    for i, page in enumerate(pages):
        resized = _resize_if_needed(page)
        b64 = _pil_to_base64_png(resized)
        logger.info(
            f"Page {i+1}: {page.size[0]}×{page.size[1]}px → "
            f"{resized.size[0]}×{resized.size[1]}px, "
            f"{len(b64)//1024} KB base64"
        )
        result.append(b64)
    return result

# ── OpenAI Vision API call ────────────────────────────────────────────────────

class RateLimitError(Exception):
    """Raised on HTTP 429; carries the server-requested retry delay (seconds)."""
    def __init__(self, retry_after: float = 10.0):
        self.retry_after = retry_after
        super().__init__(f"Rate limited — retry after {retry_after}s")


async def _call_openai_vision(messages: List[Dict[str, Any]], timeout: int, model: Optional[str] = None) -> str:
    """POST to OpenAI Chat Completions (vision-compatible) and return text."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    model_name = model or OPENAI_MODEL
    payload = {
        "model":       model_name,
        "messages":    messages,
        "max_completion_tokens": LLM_MAX_TOKENS,
    }
    if not model_name.startswith("gpt-5"):
        payload["temperature"] = 0.0
    if OPENAI_JSON_MODE:
        payload["response_format"] = {"type": "json_object"}
    if model_name.startswith("gpt-5") and OPENAI_REASONING_EFFORT:
        payload["reasoning_effort"] = OPENAI_REASONING_EFFORT
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type":  "application/json",
    }
    loop = asyncio.get_running_loop()
    resp = await loop.run_in_executor(
        None,
        lambda: requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        ),
    )
    if resp.status_code == 429:
        # Respect the Retry-After header; fall back to 15 s if absent.
        retry_after = float(resp.headers.get("Retry-After", 15))
        raise RateLimitError(retry_after=retry_after)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _clean_output(text: str) -> str:
    """Strip reasoning blocks and markdown fences from LLM output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```json(.*?)```", r"\1", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(.*?)```",     r"\1", text, flags=re.DOTALL)
    return text.strip()


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract the first complete JSON object from text."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")
    brace = 0
    for i in range(start, len(text)):
        if   text[i] == "{": brace += 1
        elif text[i] == "}":
            brace -= 1
            if brace == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Unbalanced braces in model output")


# ── JSON → PageMetadata coercion ──────────────────────────────────────────────

def _patient_is_adult(age: Optional[str]) -> bool:
    if not age:
        return True
    m = re.search(r"\d+(?:\.\d+)?", str(age))
    if not m:
        return True
    try:
        years = float(m.group(0))
    except Exception:
        return True
    return years >= 18

def _format_reference_range(entry: Dict[str, Any]) -> Optional[str]:
    rr = entry.get("range") if isinstance(entry, dict) else None
    if not rr:
        return None
    pop = entry.get("population")
    return f"{pop}: {rr}" if pop else str(rr)

def _choose_primary_reference_range(rrs: List[Dict[str, Any]], patient_age: Optional[str]) -> Optional[str]:
    if not rrs:
        return None
    if _patient_is_adult(patient_age):
        adult = next(
            (r for r in rrs if r.get("population") and "adult" in str(r["population"]).lower()),
            None,
        )
        if adult:
            return _format_reference_range(adult)
    return _format_reference_range(rrs[0])

def _coerce_to_page_metadata(json_obj: Dict[str, Any], page_index: int) -> Dict[str, Any]:
    """
    Map the model's full-report JSON schema → our internal PageMetadata shape.

    LLM schema                   Internal schema
    ──────────────────────────   ─────────────────────────────
    report_metadata.patient.name → patient_details.full_name
    report_metadata.patient.gender → patient_details.sex
    report_metadata.report_type  → report_metadata.report_type
    report_metadata.lab_id       → report_metadata.accession_number
    report_metadata.hospital_name → report_metadata.hospital_name + lab_name
    report_metadata.hospital_address → report_metadata.lab_address
    report_metadata.doctor_name  → report_metadata.referring_doctor
    report_metadata.dates.*      → report_metadata.report_date / sample_collection_date
    pages[i].tests[j].panel      → tests[j].panel
    pages[i].tests[j].observed_value → tests[j].value
    pages[i].tests[j].reference_ranges → tests[j].reference_ranges (all entries)
    pages[i].tests[j].specimen_type → tests[j].specimen_type
    pages[i].comments_and_flags  → flags
    """
    # Find the right page object — model was told "page N" so it should output that number.
    # We accept both the exact match and fall back to the first page if only one exists.
    pages = json_obj.get("pages") or []
    page_number = page_index + 1
    page_obj: Dict[str, Any] = {}
    if isinstance(pages, list):
        for p in pages:
            if isinstance(p, dict) and p.get("page_number") == page_number:
                page_obj = p
                break
        if not page_obj:
            # Fallback: first page (handles single-page responses where model
            # echoes page_number=1 regardless of what we told it)
            if pages and isinstance(pages[0], dict):
                page_obj = pages[0]

    rm_in      = json_obj.get("report_metadata") or {}
    patient_in = (rm_in.get("patient") if isinstance(rm_in, dict) else {}) or {}
    dates      = (rm_in.get("dates")   if isinstance(rm_in, dict) else {}) or {}

    patient_details = {
        "full_name": patient_in.get("name"),
        "age":       patient_in.get("age"),
        "sex":       patient_in.get("gender"),   # LLM uses "gender", model uses "sex"
    }

    hospital = rm_in.get("hospital_name")
    report_metadata = {
        "report_type":            rm_in.get("report_type"),
        "report_date":            dates.get("report_date")      or rm_in.get("report_date"),
        "sample_collection_date": dates.get("collection_date")  or rm_in.get("sample_collection_date"),
        "referring_doctor":       rm_in.get("doctor_name")      or rm_in.get("referring_doctor"),
        "lab_name":               hospital                       or rm_in.get("lab_name"),
        "lab_address":            rm_in.get("hospital_address") or rm_in.get("lab_address"),
        "hospital_name":          hospital,
        "accession_number":       rm_in.get("lab_id")           or rm_in.get("accession_number"),
    }

    tests_out: List[Dict[str, Any]] = []
    for t in (page_obj.get("tests") or []):
        if not isinstance(t, dict):
            continue
        # Preserve ALL reference range entries
        rrs_out: List[Dict[str, Any]] = []
        for rr_item in (t.get("reference_ranges") or []):
            if isinstance(rr_item, dict):
                rrs_out.append({
                    "population": rr_item.get("population"),
                    "range":      rr_item.get("range"),
                })
            elif isinstance(rr_item, str):
                rrs_out.append({"population": None, "range": rr_item})

        rr_primary = _choose_primary_reference_range(rrs_out, patient_details.get("age"))

        tests_out.append({
            "panel":            t.get("panel"),
            "test_name":        t.get("test_name"),
            "method":           t.get("method"),
            "value":            t.get("observed_value"),   # LLM key → internal key
            "unit":             t.get("unit"),
            "reference_range":  rr_primary,
            "reference_ranges": rrs_out or None,           # ALL entries
            "flag":             t.get("flag"),
            "specimen_type":    t.get("specimen_type"),
            "remarks":          t.get("remarks"),
            "page_index":       page_index,
        })

    flags = page_obj.get("comments_and_flags") or []
    if not isinstance(flags, list):
        flags = [str(flags)]

    return {
        "page_index":      page_index,
        "patient_details": patient_details,
        "report_metadata": report_metadata,
        "tests":           tests_out,
        "flags":           flags,
    }


# ── Computed flags (H/L position within range) ────────────────────────────────

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

def _parse_num(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = _NUM_RE.search(s.replace(",", ""))
    try:
        return float(m.group(0)) if m else None
    except Exception:
        return None

def _parse_range(rr: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not rr:
        return None, None
    rr_norm = str(rr).replace(",", "").replace("≤", "<=").replace("≥", ">=")
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:[-\u2013\u2014]|to)\s*([-+]?\d+(?:\.\d+)?)", rr_norm, flags=re.IGNORECASE)
    if m:
        try:
            a, b = float(m.group(1)), float(m.group(2))
            return (a, b) if a <= b else (b, a)
        except Exception:
            pass
    m = re.search(r"(?:<=|<)\s*([-+]?\d+(?:\.\d+)?)", rr_norm)
    if m:
        try:
            return None, float(m.group(1))
        except Exception:
            pass
    m = re.search(r"(?:>=|>)\s*([-+]?\d+(?:\.\d+)?)", rr_norm)
    if m:
        try:
            return float(m.group(1)), None
        except Exception:
            pass
    return None, None

def enrich_tests(page_meta: PageMetadata) -> PageMetadata:
    """Compute HIGH/LOW/NORMAL flags and position ratios from extracted values."""
    for t in page_meta.tests:
        v        = _parse_num(t.value)
        lo, hi   = _parse_range(t.reference_range)
        t.value_num    = v
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
        if   v < lo: t.computed_flag = "LOW"
        elif v > hi: t.computed_flag = "HIGH"
        else:        t.computed_flag = "NORMAL"
        t.position_ratio = max(0.0, min(1.0, (v - lo) / (hi - lo)))
        if   t.position_ratio < 0.15:  t.position_label = "VERY_LOW"
        elif t.position_ratio < 0.40:  t.position_label = "LOW"
        elif t.position_ratio <= 0.60: t.position_label = "MID"
        elif t.position_ratio <= 0.85: t.position_label = "HIGH"
        else:                          t.position_label = "VERY_HIGH"
    return page_meta



# ── LLM prompt ────────────────────────────────────────────────────────────────

# The JSON schema the model must output
_SCHEMA = """{
  "report_metadata": {
    "patient": {"name": "string", "age": "string", "gender": "string"},
    "report_type": "string or null",
    "lab_id": "string or null",
    "hospital_name": "string or null",
    "hospital_address": "string or null",
    "doctor_name": "string or null",
    "referral_source": "string or null",
    "dates": {
      "registration_date": "string or null",
      "collection_date": "string or null",
      "received_date": "string or null",
      "report_date": "string or null"
    },
    "report_status": "string or null"
  },
  "pages": [
    {
      "page_number": 1,
      "comments_and_flags": ["string"],
      "tests": [
        {
          "panel": "string or null",
          "test_name": "string",
          "method": "string or null",
          "observed_value": "string or null",
          "unit": "string or null",
          "flag": "string or null",
          "reference_ranges": [
            {"population": "string or null", "range": "string"}
          ],
          "specimen_type": "string or null"
        }
      ]
    }
  ]
}"""

_SYSTEM_PROMPT = (
    "You are a medical lab report data extraction specialist.\n"
    "You will be shown one or more page images of a lab report.\n"
    "Extract ALL data and return a single JSON object.\n\n"
    "FIELD RULES:\n"
    "1. test_name: clean title-case name only — do NOT include method text.\n"
    "2. method: the analytical method printed beneath or beside the test name "
    "   (e.g. 'Biuret', 'UV WITH P5P', 'Diazonium Ion, Blanked'). null if absent.\n"
    "3. reference_ranges: capture EVERY row of the reference interval table for "
    "   each test as a separate object.  If Total Protein has 8 population rows, "
    "   output all 8.  Use the population label as-is (e.g. 'Premature', "
    "   '7 months-1 year', 'Adults').\n"
    "4. If a patient is an adult and adult reference ranges are printed, make sure "
    "   the Adults row is included exactly; the server will use it as the primary range.\n"
    "5. flag: the H / L / HH / LL / HIGH / LOW / CRITICAL marker printed beside "
    "   the observed value, or null.\n"
    "6. observed_value: numeric result only (strip any unit suffix).\n"
    "7. comments_and_flags: ALL footnotes, interpretive comments, advisory text, "
    "   and verification details on each page.\n"
    "8. Extract ONLY what is visible — do NOT invent or infer values.\n"
    "9. If a field is absent use null.  If no reference ranges exist use [].\n"
    "10. Return ONLY valid JSON — no markdown fences, no prose.\n\n"
    "SCHEMA:\n" + _SCHEMA
)


# ── Prompt builders ──────────────────────────────────────────────────────────

def build_vision_messages_single(
    page_b64:     str,
    page_number:  int,         # 1-based, for the model
    total_pages:  int,
    report_context: Optional[Dict[str, Any]] = None,   # metadata from page 1
) -> List[Dict[str, Any]]:
    """
    Build messages for a SINGLE page call.

    report_context is injected as a text prefix so pages 2-N know the patient
    name, lab, dates etc. even though they don't contain the header.

    The model is asked to output a single-page JSON:
      { "report_metadata": {...}, "pages": [{"page_number": N, "tests": [...], ...}] }
    """
    context_text = ""
    if report_context:
        context_text = (
            "\nREPORT CONTEXT (from page 1 — do NOT re-extract, just use for reference):\n"
            + json.dumps(report_context, indent=2)
            + "\n"
        )

    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"This is page {page_number} of {total_pages} of a medical lab report."
                f"{context_text}\n"
                "Extract all test data visible on THIS PAGE ONLY and return JSON per the schema.\n"
                "For report_metadata: copy from context if provided, else extract from this page.\n"
                "Return ONLY valid JSON, no prose."
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url":    f"data:image/png;base64,{page_b64}",
                "detail": OPENAI_IMAGE_DETAIL,
            },
        },
    ]
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]


def build_vision_messages_batch(page_b64_list: List[str]) -> List[Dict[str, Any]]:
    """
    Build messages for a multi-page batch call (fallback / small documents).
    All pages in one user message for full cross-page context.
    """
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"The following {len(page_b64_list)} image(s) are consecutive pages "
                "of one medical lab report.  Extract all data per the instructions."
            ),
        }
    ]
    for i, b64 in enumerate(page_b64_list):
        content.append({"type": "text", "text": f"--- Page {i + 1} of {len(page_b64_list)} ---"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": OPENAI_IMAGE_DETAIL},
        })
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]

def _vision_model_plan(primary_retries: int) -> List[Tuple[str, int]]:
    plan = [(OPENAI_MODEL, max(1, primary_retries))]
    fallback = (OPENAI_FALLBACK_MODEL or "").strip()
    if OPENAI_USE_FALLBACK and fallback and fallback != OPENAI_MODEL:
        plan.append((fallback, max(1, OPENAI_FALLBACK_RETRIES)))
    return plan

def _looks_suspiciously_empty(page_meta: PageMetadata, page_number: int) -> bool:
    if page_number != 1 or page_meta.tests:
        return False
    pd = page_meta.patient_details
    rm = page_meta.report_metadata
    return not any([
        pd.full_name,
        pd.patient_id,
        pd.age,
        rm.accession_number,
        rm.report_date,
        rm.sample_collection_date,
    ])

def _strip_empty(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {k: _strip_empty(v) for k, v in obj.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(obj, list):
        return [v for v in (_strip_empty(x) for x in obj) if v not in (None, "", [], {})]
    return obj


# ── Single-page extraction (used for parallel calls) ─────────────────────────

async def _extract_single_page(
    page_b64:       str,
    page_number:    int,
    orig_index:     int,
    total_pages:    int,
    report_context: Optional[Dict[str, Any]],
    max_retries:    int = OPENAI_PRIMARY_RETRIES,
    semaphore:      Optional[asyncio.Semaphore] = None,
) -> PageMetadata:
    """Call OpenAI Vision for one page and return a PageMetadata.
    This function NEVER raises — it always returns a PageMetadata,
    even if all retries are exhausted or an unexpected error occurs.
    """
    try:
        timeout = LLM_TIMEOUT

        for model_name, attempts_for_model in _vision_model_plan(max_retries):
            messages = build_vision_messages_single(
                page_b64, page_number, total_pages, report_context
            )
            for attempt in range(1, attempts_for_model + 1):
                try:
                    logger.info(
                        f"Vision: page {page_number}/{total_pages}, model {model_name}, attempt {attempt}"
                    )
                    async with (semaphore if semaphore else _null_semaphore()):
                        raw = await _call_openai_vision(messages, timeout=timeout, model=model_name)
                    raw      = _clean_output(raw)
                    json_obj = _extract_json(raw)

                    coerced   = _coerce_to_page_metadata(json_obj, page_number - 1)
                    page_meta = PageMetadata.model_validate(coerced)
                    page_meta.page_index = orig_index
                    for t in page_meta.tests:
                        if t.page_index is None:
                            t.page_index = orig_index
                    page_meta = enrich_tests(page_meta)
                    if _looks_suspiciously_empty(page_meta, page_number):
                        raise ValueError("suspiciously empty first-page extraction")
                    if model_name != OPENAI_MODEL:
                        page_meta.flags = sorted(set((page_meta.flags or []) + [f"used_fallback_model:{model_name}"]))
                    logger.info(
                        f"Page {page_number}: extracted {len(page_meta.tests)} test(s) with {model_name}."
                    )
                    return page_meta

                except RateLimitError as e:
                    wait = e.retry_after + attempt * 2
                    logger.warning(
                        f"Page {page_number} model {model_name} attempt {attempt} rate-limited — "
                        f"waiting {wait:.1f}s (Retry-After={e.retry_after}s)"
                    )
                    if attempt < attempts_for_model:
                        await asyncio.sleep(wait)
                        continue

                except Exception as e:
                    logger.warning(f"Page {page_number} model {model_name} attempt {attempt} failed: {e}")
                    if attempt < attempts_for_model:
                        backoff = min(2 ** attempt, 60)
                        await asyncio.sleep(backoff)
                        messages = build_vision_messages_single(
                            page_b64, page_number, total_pages, report_context
                        )
                        messages = messages + [{
                            "role":    "user",
                            "content": "Previous response was invalid JSON. Return ONLY a valid JSON object, no prose or markdown.",
                        }]

        logger.error(f"Page {page_number} failed across configured model cascade.")

    except Exception as e:
        # Catch-all: something went wrong outside the retry loop (e.g. building
        # messages, unexpected coroutine error).  Log and fall through to the
        # safe empty result below.
        logger.exception(f"Page {page_number} unexpected error: {e}")

    return PageMetadata(
        page_index=orig_index,
        flags=[f"vision_extraction_failed_page_{page_number}"],
    )


from contextlib import asynccontextmanager as _acm

@_acm
async def _null_semaphore():
    """No-op context manager used when no semaphore is passed."""
    yield


# ── Main extraction pipeline (parallel) ──────────────────────────────────────

async def extract_from_images(
    page_b64_list: List[str],
    source_pages:  List[int],    # original 0-based page indices
) -> List[PageMetadata]:
    """
    Parallel extraction strategy:

      Step 1 — Page 1 extracted first (synchronously).
               Its report_metadata (patient name, lab, dates) is captured as
               report_context and injected into every subsequent page call so
               they don't need the header to be visible to fill metadata.

      Step 2 — Pages 2-N are fired concurrently with asyncio.gather.
               Each call is a single-image request (~4-8s) rather than an
               11-image request (~40-70s).  Wall time = slowest single page,
               not the sum.

    For small documents (≤ 2 pages) we skip the overhead and call both pages
    concurrently from the start.
    """
    total = len(page_b64_list)
    if total == 0:
        return []

    logger.info(f"Parallel vision extraction: {total} page(s) total.")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_VISION)

    # ── Step 1: extract page 1 to get report context ─────────────────────────
    page1_meta = await _extract_single_page(
        page_b64    = page_b64_list[0],
        page_number = 1,
        orig_index  = source_pages[0],
        total_pages = total,
        report_context = None,
        semaphore   = semaphore,
    )

    if total == 1:
        return [page1_meta]

    # Build report_context from page 1's metadata so subsequent pages can fill
    # patient/lab fields even when those aren't printed on their page.
    rm = page1_meta.report_metadata
    pd = page1_meta.patient_details
    report_context: Dict[str, Any] = {
        "patient": {
            "name":   pd.full_name,
            "age":    pd.age,
            "gender": pd.sex,
        },
        "lab_id":           rm.accession_number,
        "hospital_name":    rm.hospital_name,
        "hospital_address": rm.lab_address,
        "doctor_name":      rm.referring_doctor,
        "dates": {
            "collection_date": rm.sample_collection_date,
            "report_date":     rm.report_date,
        },
        "report_status": None,
    }
    # Strip empty nested values to keep the context compact. If page 1 failed
    # completely, this becomes {}, and pages 2-N still extract page-local data.
    report_context = _strip_empty(report_context)

    # ── Step 2: pages 2-N in parallel ────────────────────────────────────────
    tasks = [
        _extract_single_page(
            page_b64       = page_b64_list[i],
            page_number     = i + 1,
            orig_index      = source_pages[i],
            total_pages     = total,
            report_context  = report_context,
            semaphore       = semaphore,
        )
        for i in range(1, total)
    ]

    remaining = await asyncio.gather(*tasks, return_exceptions=True)

    # _extract_single_page never raises, but if somehow an exception slipped
    # through (e.g. CancelledError during shutdown), substitute an empty page.
    safe_remaining: List[PageMetadata] = []
    for i, result in enumerate(remaining):
        if isinstance(result, BaseException):
            page_number = i + 2   # pages 2-N
            logger.error(f"Page {page_number} gather exception (unexpected): {result}")
            safe_remaining.append(PageMetadata(
                page_index=source_pages[i + 1],
                flags=[f"vision_extraction_failed_page_{page_number}"],
            ))
        else:
            safe_remaining.append(result)

    pages_meta = [page1_meta] + safe_remaining
    total_tests = sum(len(p.tests) for p in pages_meta)
    logger.info(
        f"Parallel extraction complete: {total_tests} tests across {total} page(s)."
    )
    return pages_meta


# ── FastAPI app ───────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set.")
    else:
        logger.info(f"Vision mode ready. Model: {OPENAI_MODEL}")
    yield

app = FastAPI(
    title="Medical Report Extractor — Vision Edition",
    version="1.0",
    lifespan=lifespan,
)

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Report Extractor</title>
    <style>
        :root {
            --tone-quartz: #7f8c8d;
            --owner-rev: "amruth-s-v1";
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; color: white; margin-bottom: 30px; animation: fadeIn 0.6s ease-in; }
        .header h1 { font-size: 2.4em; font-weight: 700; margin-bottom: 8px; text-shadow: 2px 2px 6px rgba(0,0,0,0.4); }
        .header p { font-size: 1.05em; opacity: 0.85; }
        .vision-badge {
            display: inline-block; background: linear-gradient(135deg, #f093fb, #f5576c);
            color: white; padding: 4px 14px; border-radius: 20px; font-size: 0.85em;
            font-weight: 700; margin-top: 8px; letter-spacing: 0.05em;
        }
        .card {
            background: white; border-radius: 16px; padding: 30px;
            margin-bottom: 24px; box-shadow: 0 10px 40px rgba(0,0,0,0.25);
            animation: slideUp 0.5s ease-out;
        }
        .upload-area {
            border: 3px dashed #2c5364; border-radius: 12px; padding: 60px 20px;
            text-align: center; cursor: pointer; transition: all 0.3s ease;
            background: linear-gradient(135deg, #f5f7fa 0%, #c8d6df 100%);
            position: relative; overflow: hidden;
        }
        .upload-area:hover { border-color: #f5576c; transform: translateY(-2px); box-shadow: 0 8px 20px rgba(245,87,108,0.25); }
        .upload-area.dragover { background: linear-gradient(135deg, #203a43 0%, #2c5364 100%); border-color: white; }
        .upload-area.dragover * { color: white !important; }
        .upload-icon { font-size: 4em; margin-bottom: 16px; color: #2c5364; animation: bounce 2s infinite; }
        .upload-text { font-size: 1.3em; color: #2d3748; margin-bottom: 8px; font-weight: 600; }
        .upload-hint { font-size: 0.95em; color: #718096; }
        .file-input { display: none; }
        .selected-file {
            margin-top: 16px; padding: 12px 20px;
            background: #f0fff4; border: 1px solid #9ae6b4;
            border-radius: 8px; color: #22543d; font-weight: 500; animation: fadeIn 0.3s ease-in;
        }
        .btn {
            background: linear-gradient(135deg, #203a43 0%, #2c5364 100%);
            color: white; border: none; padding: 16px 40px; font-size: 1.1em;
            font-weight: 600; border-radius: 12px; cursor: pointer; transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(44,83,100,0.4); display: inline-block; margin-top: 20px;
        }
        .btn:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(44,83,100,0.6); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .btn-secondary {
            background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
            box-shadow: 0 4px 15px rgba(229,62,62,0.4); margin-left: 12px;
        }
        .btn-reset {
            background: linear-gradient(135deg, #718096 0%, #4a5568 100%);
            box-shadow: 0 4px 15px rgba(113,128,150,0.4); animation: float 3s ease-in-out infinite;
        }
        .button-group { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-top: 20px; }
        .status {
            margin-top: 20px; padding: 16px; border-radius: 8px;
            font-weight: 500; display: none; animation: fadeIn 0.3s ease-in;
        }
        .status.info    { background: #ebf8ff; color: #2c5282; border: 1px solid #90cdf4; }
        .status.success { background: #f0fff4; color: #22543d; border: 1px solid #9ae6b4; }
        .status.error   { background: #fff5f5; color: #742a2a; border: 1px solid #fc8181; }
        .progress-bar {
            width: 100%; height: 8px; background: #e2e8f0;
            border-radius: 4px; overflow: hidden; margin-top: 12px; display: none;
        }
        .progress-fill {
            height: 100%; background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
            width: 0%; transition: width 0.4s ease; animation: pulse 1.5s ease-in-out infinite;
        }
        .output-section { display: none; animation: fadeIn 0.6s ease-in; }
        .output-label {
            font-size: 1.2em; font-weight: 600; color: #2d3748;
            margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
        }
        .output-box {
            background: #1a202c; color: #e2e8f0; padding: 24px; border-radius: 12px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; font-size: 0.9em;
            line-height: 1.6; max-height: 650px; overflow-y: auto;
            white-space: pre-wrap; word-wrap: break-word;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.3);
        }
        .output-box::-webkit-scrollbar { width: 12px; }
        .output-box::-webkit-scrollbar-track { background: #2d3748; border-radius: 6px; }
        .output-box::-webkit-scrollbar-thumb { background: #f5576c; border-radius: 6px; }
        .json-key     { color: #81e6d9; }
        .json-string  { color: #fbd38d; }
        .json-number  { color: #b794f4; }
        .json-boolean { color: #fc8181; }
        .json-null    { color: #a0aec0; font-style: italic; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; margin-left: 8px; }
        .badge-success { background: #c6f6d5; color: #22543d; }
        .badge-vision  { background: linear-gradient(135deg, #f093fb, #f5576c); color: white; }
        .spinner {
            display: inline-block; width: 20px; height: 20px;
            border: 3px solid rgba(255,255,255,.3); border-radius: 50%;
            border-top-color: #fff; animation: spin 0.8s linear infinite;
        }
        @keyframes fadeIn  { from { opacity: 0; }  to { opacity: 1; } }
        @keyframes slideUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes bounce  { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
        @keyframes spin    { to { transform: rotate(360deg); } }
        @keyframes pulse   { 0%,100% { opacity: 1; } 50% { opacity: 0.7; } }
        @keyframes float   { 0%,100% { transform: translateY(0px); } 50% { transform: translateY(-5px); } }
    </style>
</head>
<body data-track="as7314" data-ui="medocr">
    <div class="container" data-owner="amruth-s" data-rev="v1">
        <div class="header">
            <h1> Medical Report Extractor</h1>
            <p>Upload a lab report </p>
        </div>

        <div class="card">
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📄</div>
                <div class="upload-text">Click to select or drag & drop your report</div>
                <div class="upload-hint">Supports PDF, PNG, JPG, JPEG, TIFF (Max 50 MB)</div>
                <input type="file" id="fileInput" class="file-input"
                       accept=".pdf,.png,.jpg,.jpeg,.tiff,.tif" multiple>
            </div>
            <div id="selectedFile" class="selected-file" style="display:none;"></div>
            <div class="button-group">
                <button id="processBtn" class="btn"           style="display:none;">Extract</button>
                <button id="cancelBtn"  class="btn btn-secondary" style="display:none;">Cancel</button>
                <button id="resetBtn"   class="btn btn-reset"     style="display:none;">Start Over</button>
            </div>
            <div id="status" class="status"></div>
            <div id="progressBar" class="progress-bar">
                <div id="progressFill" class="progress-fill"></div>
            </div>
        </div>

        <div id="jsonSection" class="card output-section">
            <div class="output-label">
                 Structured JSON Data
                <span class="badge badge-vision">Vision Extracted</span>
                <button id="downloadBtn" class="btn"
                        style="margin-top:0;padding:8px 20px;font-size:0.9em;margin-left:auto;display:none;">
                    Download JSON
                </button>
            </div>
            <div id="jsonOutput" class="output-box"></div>
        </div>
    </div>

    <script>
        const uploadArea    = document.getElementById('uploadArea');
        const fileInput     = document.getElementById('fileInput');
        const selectedFileDiv = document.getElementById('selectedFile');
        const processBtn    = document.getElementById('processBtn');
        const cancelBtn     = document.getElementById('cancelBtn');
        const resetBtn      = document.getElementById('resetBtn');
        const downloadBtn   = document.getElementById('downloadBtn');
        const statusDiv     = document.getElementById('status');
        const progressBar   = document.getElementById('progressBar');
        const progressFill  = document.getElementById('progressFill');
        const jsonSection   = document.getElementById('jsonSection');
        const jsonOutput    = document.getElementById('jsonOutput');

        let selectedFiles = null, abortController = null, isProcessing = false;

        function resetAll() {
            selectedFiles = null; abortController = null; isProcessing = false;
            fileInput.value = '';
            selectedFileDiv.style.display = 'none'; selectedFileDiv.textContent = '';
            processBtn.style.display = 'none'; cancelBtn.style.display = 'none';
            resetBtn.style.display = 'none'; downloadBtn.style.display = 'none';
            processBtn.disabled = false;
            statusDiv.style.display = 'none'; statusDiv.textContent = '';
            progressBar.style.display = 'none'; progressFill.style.width = '0%';
            jsonSection.style.display = 'none'; jsonOutput.innerHTML = '';
            uploadArea.classList.remove('dragover');
        }

        uploadArea.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', (e) => {
            selectedFiles = e.target.files;
            if (selectedFiles.length > 0) {
                const oversized = Array.from(selectedFiles).filter(f => f.size > 50 * 1024 * 1024);
                if (oversized.length > 0) {
                    showStatus('error', 'File too large (max 50 MB): ' + oversized.map(f => f.name).join(', '));
                    fileInput.value = ''; selectedFiles = null;
                    selectedFileDiv.style.display = 'none'; processBtn.style.display = 'none';
                    return;
                }
                selectedFileDiv.textContent = '✅ Selected: ' + Array.from(selectedFiles).map(f => f.name).join(', ');
                selectedFileDiv.style.display = 'block';
                processBtn.style.display = 'inline-block';
            }
        });

        uploadArea.addEventListener('dragover',  (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
        uploadArea.addEventListener('dragleave', ()  => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault(); uploadArea.classList.remove('dragover');
            selectedFiles = e.dataTransfer.files;
            if (selectedFiles.length > 0) {
                selectedFileDiv.textContent = '✅ Selected: ' + Array.from(selectedFiles).map(f => f.name).join(', ');
                selectedFileDiv.style.display = 'block';
                processBtn.style.display = 'inline-block';
            }
        });

        processBtn.addEventListener('click', async () => {
            if (!selectedFiles || selectedFiles.length === 0) {
                showStatus('error', 'Please select a file first'); return;
            }
            isProcessing = true; processBtn.disabled = true;
            cancelBtn.style.display = 'inline-block'; resetBtn.style.display = 'none';
            jsonSection.style.display = 'none';
            progressBar.style.display = 'block'; progressFill.style.width = '15%';
            showStatusHTML('info', '<div class="spinner"></div> Sending pages to OpenAI Vision...');

            const formData = new FormData();
            for (let file of selectedFiles) formData.append('files', file);
            abortController = new AbortController();

            try {
                progressFill.style.width = '40%';
                const response = await fetch('/api/process-report', {
                    method: 'POST', body: formData, signal: abortController.signal
                });
                if (!response.ok) throw new Error(await response.text());

                progressFill.style.width = '85%';
                showStatusHTML('info', '<div class="spinner"></div> Parsing structured data...');
                const data = await response.json();
                progressFill.style.width = '100%';

                const formattedJson = JSON.stringify(data.pages, null, 2);
                jsonOutput.innerHTML = syntaxHighlight(formattedJson);
                jsonSection.style.display = 'block';

                downloadBtn.style.display = 'inline-block';
                downloadBtn.onclick = () => {
                    const blob = new Blob([formattedJson], {type: 'application/json'});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url; a.download = 'medical_report_vision.json'; a.click();
                    URL.revokeObjectURL(url);
                };

                const testCount = data.pages.reduce((s, p) => s + (p.tests?.length || 0), 0);
                showStatus('success', `✅ Done! Extracted ${data.pages.length} page(s) with ${testCount} test results.`);
                progressBar.style.display = 'none';
                cancelBtn.style.display = 'none'; resetBtn.style.display = 'inline-block';

            } catch (error) {
                if (error.name === 'AbortError') {
                    showStatus('error', '⛔ Processing cancelled.');
                } else {
                    // Use textContent (not innerHTML) to avoid XSS from server-provided strings
                    statusDiv.className = 'status error';
                    statusDiv.textContent = '❌ Error: ' + error.message;
                    statusDiv.style.display = 'block';
                }
                progressBar.style.display = 'none';
                cancelBtn.style.display = 'none'; resetBtn.style.display = 'inline-block';
            } finally {
                isProcessing = false; processBtn.disabled = false; abortController = null;
                cancelBtn.disabled = false;
            }
        });

        cancelBtn.addEventListener('click', () => {
            if (abortController && isProcessing) {
                abortController.abort();
                showStatus('info', 'Cancelling...');
                cancelBtn.disabled = true;
            }
        });

        resetBtn.addEventListener('click', () => {
            resetAll();
            showStatus('info', '🔄 Ready for a new report');
            setTimeout(() => statusDiv.style.display = 'none', 2000);
        });

        function showStatus(type, message) {
            statusDiv.className = 'status ' + type;
            statusDiv.textContent = message;
            statusDiv.style.display = 'block';
        }

        function showStatusHTML(type, html) {
            // Only call this with trusted, hardcoded HTML strings (e.g. spinner)
            statusDiv.className = 'status ' + type;
            statusDiv.innerHTML = html;
            statusDiv.style.display = 'block';
        }

        function syntaxHighlight(json) {
            json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function(match) {
                let cls = 'json-number';
                if (/^"/.test(match)) cls = /:$/.test(match) ? 'json-key' : 'json-string';
                else if (/true|false/.test(match)) cls = 'json-boolean';
                else if (/null/.test(match)) cls = 'json-null';
                return '<span class="' + cls + '">' + match + '</span>';
            });
        }
    </script>
</body>
</html>"""


@app.post("/api/process-report", response_class=JSONResponse)
async def api_process_report(files: List[UploadFile] = File(...)):
    """Convert uploaded files to page images and extract structured data."""

    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Max {MAX_FILES_PER_REQUEST} files per request.")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured.")

    temp_dir = tempfile.mkdtemp(prefix="medical_vision_as7314_")
    try:
        # ── 1. Stream uploads to disk (avoids loading 50MB files into RAM) ──────
        paths: List[Path] = []
        for f in files:
            suffix = Path(f.filename or "").suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
            out = Path(temp_dir) / f"upload_{len(paths)}{suffix}"
            size = 0
            with out.open("wb") as dst:
                while True:
                    chunk = await f.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_UPLOAD_SIZE_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File '{f.filename}' exceeds {MAX_UPLOAD_SIZE_MB} MB."
                        )
                    dst.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail=f"File '{f.filename}' is empty.")
            paths.append(out)

        # ── 2. Convert to page images ─────────────────────────────────────────
        loop = asyncio.get_running_loop()
        all_pages: List[Image.Image] = []
        for path in paths:
            pages = await loop.run_in_executor(None, lambda p=path: file_to_page_images(p))
            all_pages.extend(pages)

        if not all_pages:
            raise HTTPException(status_code=422, detail="No pages could be extracted from uploaded file(s).")

        logger.info(f"Total pages to process: {len(all_pages)}")

        # ── 3. Encode images ──────────────────────────────────────────────────
        page_b64_list = await loop.run_in_executor(None, lambda: pages_to_base64(all_pages))

        # ── 4. Vision extraction ──────────────────────────────────────────────
        source_indices = list(range(len(all_pages)))
        pages_meta = await extract_from_images(page_b64_list, source_indices)

        return JSONResponse(content={
            "pages":      [p.model_dump(exclude_none=True) for p in pages_meta],
            "page_count": len(pages_meta),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"API error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    print("Open your browser to: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
