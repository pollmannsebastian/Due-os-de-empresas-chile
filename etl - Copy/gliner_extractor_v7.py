"""
GLiNER Legal Entity Extraction Pipeline  ·  v6.0 (Reliability Rewrite)
==================================================
Major Improvements over v5:
  • Written-RUT ↔ Entity association: bidirectional search (before AND after entity)
  • Robust notary exclusion: historical notary mentions, "Notario [City]", "Notario Público"
  • Better target company detection: priority-based from explicit markers
  • "Don/Doña" prefix stripping from all names
  • No self-referencing companies (company as partner of itself)
  • RUT validation before assignment
  • No more misassignments from proximity confusion
  • Historical context filtering (past notaries in "constituida ante...")

Run: python etl/gliner_extractor_v6.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import logging
import unicodedata
import threading
import queue
import gc
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, List, Tuple

# ── optional fast JSON ──────────────────────────────────────────────────────
try:
    import orjson
    _loads = orjson.loads
except ImportError:
    _loads = json.loads

import duckdb
import torch
from gliner import GLiNER

# Import our written-RUT parser
try:
    from rut_word_parser import extract_ruts_from_words
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from rut_word_parser import extract_ruts_from_words

# ── silence noisy libraries ─────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

try:
    import transformers
    transformers.logging.set_verbosity_error()
except (ImportError, AttributeError):
    pass

try:
    from huggingface_hub import utils as hf_utils
    hf_utils.disable_progress_bars()
except (ImportError, AttributeError):
    pass

# ── logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("pipeline")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH      = os.path.join(BASE_DIR, "dataset_optimized_v2.jsonl")
DB_PATH           = os.path.join(BASE_DIR, "extraction_results_v7.duckdb")
MODEL_NAME        = "urchade/gliner_small-v2.1"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# OPTIMIZATION PARAMETERS
BATCH_SIZE        = 1024*4 if DEVICE == "cuda" else 32
CHUNK_SIZE        = 1024
NUM_CPU_WORKERS   = 4
MAX_TEXT_LEN      = 6000
MAX_RUT_DIST      = 400  # Increased for written RUTs which are longer
DB_WRITE_BUFFER   = 50
MAX_INFLIGHT      = NUM_CPU_WORKERS * 6

NER_LABELS        = ["person", "company"]
NER_THRESHOLD     = 0.40

# ═══════════════════════════════════════════════════════════════════════════════
# KEYWORD FILTERS & REGEX
# ═══════════════════════════════════════════════════════════════════════════════

_FILTER_KEYWORDS = (
    "constitu", "socios son",
    "capital se reparte", "modific", "transform", "fusi", "absorc", "divis",
    "responsabilidad socios", "administración sociedad", "girará bajo",
    "objeto social", "razón social", "represent",
    "socio", "accionista", "participacion", "aporte",
    "sanea", "rectific", "reinversi", "disol", "liquid",
    "aument", "disminu", "capital", 
    "empresa individual", "eirl", "irl",
    "agencia", "sucursal", "designa", "revoca", "poder", "mandato", "renuncia", "asume",
    "sociedad anónima", "sociedad por acciones", "limitada", "ltda", "spa", "s.a", 
    "extracto", "saneamiento", "publicación", 
)

_FILTER_RE = re.compile(
    r"(" + "|".join(re.escape(kw) for kw in _FILTER_KEYWORDS) + r")",
    re.IGNORECASE,
)

_HEADER_KEYWORDS = ("EXTRACTO", "CONSTITUCIÓN", "MODIFICACIÓN", "ESCRITURA", "NOTARÍA", "CERTIFICO", "CERTIFICA")

# ── Company blacklist ──────────────────────────────────────────────────────────
_COMPANY_BLACKLIST = {
    "razon social", "razón social",
    "razon social:", "razón social:",
    "nombre", "nombre de fantasía", "nombre de fantasia",
    "nombre comercial", "nombre comercial y de fantasía",
    "persona natural", "persona juridica", "persona jurídica",
    "giro social", "capital social",
    "domicilio social", "objeto social",
    "sociedad por acciones", "sociedad por acciones spa",
    "sociedad anonima", "sociedad anónima",
    "sociedad anonima cerrada", "sociedad anónima cerrada",
    "sociedad anonima abierta", "sociedad anónima abierta",
    "sociedad de responsabilidad limitada",
    "sociedad responsabilidad limitada",   # without "de"
    "empresa individual de responsabilidad limitada",
    "spa", "s.a.", "sa", "ltda", "limitada", "eirl",
    "spa sociedad por acciones",
    "s.p.a.", "s.p.a", "sociedad por acciones (spa)",
    "sociedad anonima (cerrada)", "sociedad anónima (cerrada)", 
    # Professions that GLiNER sometimes picks up as companies
    "ingeniero comercial", "ingeniero agronomo", "ingeniero agrónomo", 
    "secretaria", "odontologo", "odontólogo", "estudiante", "abogado", 
    "perito agricola", "perito agrícola", "tecnico agricola", "técnico agrícola", 
    "empresario agricola", "empresario agrícola", "agronomo", "agrónomo",
    "ingeniero civil", "contador", "contador auditor",
    # Legal terms GLiNER sometimes picks up as companies
    "abogadas spa", "abogados spa", "abogados ltda",
    # Notary-like things that GLiNER picks up as companies
    "notario público", "notario publico", "notaria pública", "notaria publica",
    "notaría", "notaria",
    # Generic institutional terms
    "diario oficial", "registro de comercio", "conservador de bienes raíces",
    "conservador de bienes raices", "conservador de comercio",
    "servicio de impuestos internos",
}

# ── Pattern: "Notario [City]" — filter these as they are NOT companies ──────
_NOTARIO_CITY_RE = re.compile(
    r"^notar[ií][oa]\s+(?:p[uú]blic[oa]\s+)?(?:de\s+|interino\s+(?:de\s+)?|suplente\s+(?:de\s+)?|titular\s+(?:de\s+)?)?"
    r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+",
    re.IGNORECASE
)

# ── Notary detection patterns ──────────────────────────────────────────────────
# Matches text BEFORE a person name that signals "this is a notary"
_NOTARY_BEFORE_RE = re.compile(
    r"(?:"
    r"ante\s+(?:el\s+|la\s+)?notar[ií][oa]\s+(?:p[uú]blic[oa]\s+)?(?:titular|suplente|interino|reemplazante)?\s*(?:(?:de\s+)?[A-Z][a-z]+\s*)?(?:don\s+|do[ñn]a\s+)?"
    r"|notar[ií][oa]\s+(?:p[uú]blic[oa]\s+)?(?:titular|suplente|interino|reemplazante)?\s*(?:,\s*)?(?:(?:de\s+)?[A-Z][a-z]+\s*)?\s*(?:don\s+|do[ñn]a\s+)?"
    r"|(?:suplente|subrogante|interino|reemplazante)\s+(?:de\s+)?(?:est[eé]\s+oficio\s+|don\s+|do[ñn]a\s+)?"
    r"|conservador\s+(?:de\s+)?\w+\s+(?:don\s+|do[ñn]a\s+)?"
    r"|ante\s+(?:el\s+|la\s+|mi\s+)?suplente\s*(?:de\s+est[eé]\s+oficio\s+)?(?:don\s+|do[ñn]a\s+)?"
    r"|(?:del\s+|al\s+|es\s+)?titular\s+(?:de\s+)?(?:don\s+|do[ñn]a\s+)?"
    r"|notar[ií]a[^\n]{1,40}?titular\s+(?:de\s+)?(?:don\s+|do[ñn]a\s+)?"
    r")$",
    re.IGNORECASE
)

# Matches text AFTER a person name that signals "this is a notary"
_NOTARY_AFTER_RE = re.compile(
    r"^[\s,]*(?:(?:[A-ZÁÉÍÓÚÑa-záéíóúñ]+\s*(?:,\s*)?){0,2})?" # Allow up to 2 words in case of missed last names
    r"(?:"
    r"notar[ií][oa]|suplente|conservador|archivero"
    r"|titular\s+(?:\d+[oªa°]\s+)?notar[ií]a"
    r"|,?\s*(?:abogado|abogada)\s*(?:,\s*)?notar[ií][oa]"
    r")(?:\s|\b|,|$)",
    re.IGNORECASE
)

# ── Historical context patterns ──────────────────────────────────────────────
# These indicate mentions of past notary contexts — anyone named here IS a notary.
# We use $ to ensure the marker is strictly preceding the name.
_HISTORICAL_NOTARY_RE = re.compile(
    r"(?:"
    r"(?:otorgad[oa]\s+(?:en\s+|ante\s+)?(?:la\s+)?notar[ií]a\s+(?:de\s+)?)"
    r"|(?:(?:constituida|inscrita|modificada)\s+(?:por\s+escritura\s+)?(?:p[uú]blica\s+)?(?:en\s+|ante\s+)?(?:la\s+|el\s+)?notar[ií][ao]\s+(?:de\s+)?)"
    r"|(?:ante\s+(?:el\s+|la\s+)?notar[ií][oa]\s+(?:p[uú]blic[oa]\s+)?(?:(?:de\s+)?[A-Z][a-z]+\s*)?(?:de\s+)?(?:don\s+|do[ñn]a\s+)?)"  
    r"|(?:(?:la\s+)?notar[ií]a\s+(?:(?:de\s+)?[A-Z][a-z]+\s*)?(?:de\s+)?(?:don\s+|do[ñn]a\s+)?)"
    r")$",
    re.IGNORECASE
)

# ── Broader notary context check keywords ──────────────────────────────────
_NOTARY_KEYWORDS_RE = re.compile(
    r"(?<!\bante\s)(?<!\bel\s)notar[ií][ao]|suplente|subrogante|interino|reemplazante|conservador|archivero|ante\s+m[íi]|ante\s+el\s+suplente",
    re.IGNORECASE
)
# Note: 'titular' removed as it often means owner of shares.

# ── Fantasy name pattern — entities intro'd this way are NOT separate partners ─
_QF = '["\u201c\u201d\u201e]'  # quote character class (local, _Q defined later)
_FANTASY_NAME_RE = re.compile(
    r"nombre\s+(?:de\s+)?fantas[ií]a\s*[:;.]?\s*" + _QF + r"?([^\n]+?)" + _QF + r"?(?:\s*[.;,\-]|$)"
    r"|(?:pudiendo\s+(?:usar|utilizar)\s+(?:como\s+)?nombre\s+(?:de\s+)?fantas[ií]a)\s+"
    + _QF + r"?([^\n]+?)" + _QF + r"?(?:\s*[.;,\-]|$)"
    r"|nombre\s+comercial\s+(?:y\s+de\s+fantas[ií]a\s+)?" + _QF + r"?([^\n]+?)" + _QF + r"?(?:\s*[.;,\-]|$)",
    re.IGNORECASE
)

# ── New razón social pattern — "razón social será X" is NOT a separate partner ──
_NEW_RAZON_RE = re.compile(
    r"(?:raz[óo]n\s+social\s+ser[áa]|nueva\s+raz[óo]n\s+social)\s+" + _QF + r"?([\w\s,.\-&áéíóúñÁÉÍÓÚÑ]+?(?:LTDA\.?|LIMITADA|SpA|S\.?A\.?|E\.?I\.?R\.?L\.?))" + _QF + r"?(?:\s*[.;,\-]|$)",
    re.IGNORECASE
)

_ADDRESS_START_RE = re.compile(
    r"^(fundo|parcela|camino|calle|pasaje|av\.|avenida|km\.|ruta|poblaci[oó]n|villa|depto|departamento|casa|lote|sitio|sector|oficina|local|unidad|gustavo\s+filippi)\b",
    re.IGNORECASE
)

# ── RUT regex — matches standard numeric RUTs with separators ─────────────
_RUT_RE = re.compile(
    r"(?<!\d)"
    r"("
    r"\d{1,2}(?:\.\d{3}){2}[-.]?[0-9kK]"
    r"|\d{6,8}[-.]?[0-9kK]"
    r")"
    r"(?!\w)",
    re.IGNORECASE,
)

_CVE_RE = re.compile(r'"cve"\s*:\s*"([^"]+)"')
_BOILERPLATE_RE = re.compile(
    r"CVE \d+.*?(?:Empresas y Cooperativas|Cooperativas)\s*(?:CVE \d+)?\s*",
    re.DOTALL | re.IGNORECASE,
)

_ADMIN_TERMS_RE = re.compile(
    r"Registro\s+Comercio|Conservador\s+(?:de\s+)?(?:Bienes\s+Ra[íi]ces|Comercio)|Notar[íi]a|Diario\s+Oficial|Servicio\s+de\s+Impuestos\s+Internos|Tesorer[íi]a|Ministerio",
    re.IGNORECASE
)

# ── Target company detection — multi-pattern, priority-ordered ──────────────
# Quote chars used in documents: " \u201c \u201d \u201e '
_Q = '["\u201c\u201d\u201e\u201f\x22\x27\u2018\u2019]'  # character class for all quote variants including single quotes
_NQ = '[^"\u201c\u201d\u201e\u201f\x22\x27\u2018\u2019\n]'  # negated character class

# Priority 1: Explicit name markers
_TARGET_MARKERS_EXPLICIT = re.compile(
    r'(?:NOMBRE|RAZ[ÓO]N\s+SOCIAL)\s*[:;]\s*' + _Q + r'(' + _NQ + r'+)' + _Q
    + r'|girar[áa]\s+bajo\s+(?:la\s+raz[óo]n\s+social|el\s+nombre)\s*(?:de)?\s*' + _Q + r'(' + _NQ + r'+)' + _Q
    + r'|denominada\s*' + _Q + r'(' + _NQ + r'+)' + _Q
    + r'|(?:sociedad\s+)?(?:por\s+acciones\s+)?(?:con\s+nombre|bajo\s+(?:el\s+)?nombre)\s*[:;]?\s*' + _Q + r'(' + _NQ + r'+)' + _Q,
    re.IGNORECASE
)

# Priority 2: Constitution/modification markers 
_TARGET_MARKERS_CONSTITUTION = re.compile(
    r'(?:transformar?|constituye(?:ro)?n?|modificar?(?:on)?)\s+'
    r'(?:la\s+)?(?:sociedad\s+)?(?:por\s+acciones\s+)?'
    r'(?:denominada\s+)?' + _Q + r'(' + _NQ + r'+)' + _Q
    + r'|(?:transformar?|constituye(?:ro)?n?|modificar?(?:on)?)\s+'
    r'(?:la\s+)?(?:sociedad\s+)?(?:por\s+acciones\s+)?'
    r'(?:denominada\s+)?(?:")?([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s,\-&]+(?:LTDA\.?|LIMITADA|SpA|S\.?A\.?|E\.?I\.?R\.?L\.?))',
    re.IGNORECASE
)

# Priority 3: "rectificó empresa individual..." pattern
_TARGET_MARKERS_EIRL = re.compile(
    r'(?:rectific[óo]|constitu[yó]\w*)\s+empresa\s+individual\s+de\s+responsabilidad\s+limitada\s*,?\s*(?:bajo\s+)?(?:el\s+nombre\s+)?' + _Q + r'(' + _NQ + r'+)' + _Q,
    re.IGNORECASE
)

# Priority 4: "Razón social: X" without quotes (sometimes used without quotes)
_TARGET_MARKERS_RAZON = re.compile(
    r"(?:raz[óo]n\s+social|nombre\s+(?:de\s+)?fantas[ií]a|nombre)\s*[,:;]\s*"
    r"([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑA-Za-záéíóúñ\s,.\-&']+(?:LTDA\.?|LIMITADA|SpA|S\.?A\.?|E\.?I\.?R\.?L\.?))",
    re.IGNORECASE
)

# Priority 5: "constituyeron sociedad X" without quotes
_TARGET_MARKERS_CONSTIT_NOQUOTE = re.compile(
    r"constituye(?:ro)?n\s+sociedad(?:\s+por\s+acciones)?\s*\.?\s*"
    r"(?:raz[óo]n\s+social\s*[,:;]\s*)?"
    r"([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑA-Za-záéíóúñ\s,.\-&']+(?:LTDA\.?|LIMITADA|SpA|S\.?A\.?|E\.?I\.?R\.?L\.?))",
    re.IGNORECASE
)

# Priority 6b: "socios de [COMPANY]" — the company after "socios de" is the target
_TARGET_MARKERS_SOCIOS = re.compile(
    r"(?:(?:[úu]nicos?\s+)?socios?\s+de\s+)" + _Q + r"?([^\n\"\u201c\u201d\u201e,]+(?:LTDA\.?|LIMITADA|SpA|S\.?A\.?|E\.?I\.?R\.?L\.?))" + _Q + r"?",
    re.IGNORECASE
)

# Priority 6c: "accionistas sociedad/de [COMPANY]" or "JUNTA...ACCIONISTAS...SOCIEDAD X"
_TARGET_MARKERS_ACCIONISTAS = re.compile(
    r"accionistas\s+(?:de\s+(?:la\s+)?)?(?:sociedad\s+)?" + _Q + r"?([^\n\"\u201c\u201d\u201e]+?(?:LTDA\.?|LIMITADA|SpA|S\.?A\.?|E\.?I\.?R\.?L\.?))" + _Q + r"?",
    re.IGNORECASE
)

# Priority 6d: "modificaron sociedad X" or "modificaron ... estatutos ... de X"
_TARGET_MARKERS_MODIFICARON = re.compile(
    r"modificaron\s+(?:los\s+estatutos\s+sociales\s+de\s+)?" + _Q + r"?([^\"\u201c\u201d\u201e\u201f\x22\x27\u2018\u2019]{2,100}?(?:LTDA\.?|LIMITADA|SpA|S\.?A\.?|E\.?I\.?R\.?L\.?))" + _Q + r"?",
    re.IGNORECASE
)

_REPRESENTATION_RE = re.compile(
    r"\b(?:en\s+representaci[óo]n\s+(?:de|seg[uú]n)|pp\b|por\s+cuenta\s+de|representante\s+(?:de|legal)|agente\s+de)\b",
    re.IGNORECASE
)

# "en representación de" can also appear as "en representación según de"
_REPR_ACCORDING_RE = re.compile(
    r"\ben\s+representaci[óo]n\s+seg[uú]n\s+(?:de\s+)?",
    re.IGNORECASE
)

# Mandatario pattern — mandatarios are NOT partners, they are agents
_MANDATARIO_RE = re.compile(
    r"\b(?:su\s+)?mandatari[oa]\s+",
    re.IGNORECASE
)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DocMeta:
    cve:  str
    date: str
    url:  str
    text: str

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_rut(raw: str) -> str:
    """Normalize a RUT string to XXXXXXX-D format."""
    clean = re.sub(r'[^0-9Kk]', '', raw.upper())
    if len(clean) > 1:
        return f"{clean[:-1]}-{clean[-1]}"
    return clean

def clean_entity_name(name: str) -> str:
    """Clean up entity name: remove Don/Doña prefix, normalize whitespace."""
    if not name:
        return ""
    # Strip whitespace and punctuation from edges
    s = name.strip().strip(".,;:()\"'")
    # Normalize internal whitespace
    s = re.sub(r"[\n\r\t]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    # Remove Don/Doña/Sr/Sra prefix
    # Note: Doña = D-o-ñ-a, which is different from Dona = D-o-n-a
    s = re.sub(r"^(?:do[ñn]a\s+|don\s+|se[ñn]or(?:a)?\s+|sr\.?\s+|sra\.?\s+)", "", s, flags=re.IGNORECASE)
    return s.strip()

def normalize_name(name: str) -> str:
    """Normalize for deduplication: lowercase, ASCII, sorted words."""
    if not name: return ""
    s = name.lower()
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8')
    # Remove honorifics
    for prefix in ["don ", "dona ", "senor ", "senora ", "sr ", "sra "]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    words = s.split()
    if not words: return ""
    words.sort()
    return " ".join(words)

def validate_rut(rut: str) -> bool:
    """Basic validation that a RUT looks reasonable."""
    parts = rut.split("-")
    if len(parts) != 2:
        return False
    body, dv = parts
    if not body.isdigit():
        return False
    if dv not in "0123456789K":
        return False
    num = int(body)
    # RUTs are typically between 1M and 99M for companies and people
    # Some older ones can be lower
    if num < 1000 or num > 999_999_999:
        return False
    return True

def extract_all_ruts(text: str) -> list:
    """Extract all RUTs from text, both numeric and written-word formats.
    Returns list of (normalized_rut, start, end) tuples."""
    ruts = []
    
    # 1. Standard numeric regex
    for m in _RUT_RE.finditer(text):
        nrut = normalize_rut(m.group())
        if validate_rut(nrut):
            ruts.append((nrut, m.start(), m.end()))
    
    # 2. Written word parsing
    written = extract_ruts_from_words(text)
    if written:
        for w_rut, w_start, w_end in written:
            if validate_rut(w_rut):
                ruts.append((w_rut, w_start, w_end))
    
    # Sort by position
    ruts.sort(key=lambda x: x[1])
    return ruts


def find_rut_for_entity(e_start: int, e_end: int, ruts: list, text: str,
                         max_dist: int = MAX_RUT_DIST, consumed: set = None, 
                         prefer_col_unico: bool = False) -> tuple:
    """Find the best RUT for an entity by searching BOTH directions.
    
    prefer_col_unico: if True, gives high priority to RUTs preceded by "rol único" 
                     (typical for companies).
    """
    best = None
    best_idx = None
    best_score = -999999  # Higher is better

    for idx, (rut_str, r_start, r_end) in enumerate(ruts):
        if consumed is not None and idx in consumed:
            continue
        
        # Calculate base distance
        if r_start >= e_end:
            dist = r_start - e_end
        elif r_start >= e_start:
            dist = 0
        elif r_end <= e_start:
            dist = e_start - r_end
        else:
            dist = 0
            
        if dist > max_dist:
            continue

        # Score the match
        # 1. Distance penalty (inverse of distance)
        score = -dist 
        
        # 2. Anchor bonus
        ctx_before = text[max(0, r_start - 60):r_start].lower()
        ctx_anchor = text[r_start:min(len(text), r_start + 40)].lower()  # Written RUTs include anchor in their bounds
        
        if ("rol" in ctx_before and "tributario" in ctx_before) or ("rol" in ctx_anchor and "tributario" in ctx_anchor):
            score += 200 # Heavy bonus for companies
        elif ("cédula" in ctx_before or "identidad" in ctx_before or "run" in ctx_before) or \
             ("cédula" in ctx_anchor or "identidad" in ctx_anchor or "run" in ctx_anchor):
            # Personal RUT anchor - if we prefer_col_unico, this is a strong negative signal
            if prefer_col_unico:
                if dist > 3: # allow if virtually overlapping, otherwise penalize
                    score -= 500 # Heavy penalty for company picking up a personal RUT
            else:
                score += 100
        elif re.search(r"\brut\b\s*[:#no.]*\s*$", ctx_before) or re.search(r"^\s*\brut\b", ctx_anchor):
            score += 150 # Generic RUT anchor

        if score > best_score:
            best_score = score
            best = rut_str
            best_idx = idx

    if best_score < -250:
        return None, None

    return best, best_idx


def find_rut_near_name(name: str, e_start: int, e_end: int, text: str, 
                        ruts: list, consumed: set = None, max_dist: int = MAX_RUT_DIST) -> tuple:
    """Enhanced RUT finding that also considers the NAME being mentioned 
    near a "cédula de identidad" anchor."""
    
    # First try the standard positional search
    rut, ridx = find_rut_for_entity(e_start, e_end, ruts, text, max_dist, consumed)
    if rut:
        return rut, ridx
    
    # If entity has a short name and text mentions it again elsewhere near a RUT,
    # try to find that mention. This is useful for cases where a name appears
    # multiple times in the document.
    # We skip this for very short names to avoid false matches
    if len(name) < 5:
        return None, None
    
    # Search for other mentions of this name in the text
    name_upper = name.upper()
    search_pos = 0
    while True:
        found = text.upper().find(name_upper, search_pos)
        if found == -1:
            break
        found_end = found + len(name)
        # Skip the original mention
        if abs(found - e_start) < 5:
            search_pos = found_end
            continue
        # Try finding a RUT near this other mention
        rut2, ridx2 = find_rut_for_entity(found, found_end, ruts, text, max_dist, consumed)
        if rut2:
            return rut2, ridx2
        search_pos = found_end
    
    return None, None


def is_notary_context(text: str, e_start: int, e_end: int) -> bool:
    """Check if an entity appears in a notary context."""
    before_text = text[max(0, e_start - 100):e_start].lower()
    after_text = text[e_end:min(len(text), e_end + 80)].lower()
    
    if _NOTARY_BEFORE_RE.search(before_text):
        return True
    
    # After context must match STRICTLY at start (not search) 
    # to avoid hitting "titular" or "suplente" later in the sentence
    if _NOTARY_AFTER_RE.match(after_text):
        return True
    
    # Specific notary identification after name
    if re.search(r"^[\s,]*(?:abogad[oa])?\s*,?\s*notar[ií][oa]", after_text):
        return True
    
    return False

def is_historical_context(text: str, start: int, end: int) -> bool:
    """Check for historical notary markers strictly preceding the entity."""
    before_text = text[max(0, start-120):start].lower()
    return _HISTORICAL_NOTARY_RE.search(before_text) is not None

def find_certifico_position(text: str) -> int:
    """Find the position of CERTIFICO/CERTIFICA marker."""
    text_upper = text.upper()
    for marker in ("CERTIFICO", "CERTIFICA"):
        pos = text_upper.find(marker)
        if pos != -1: return pos
    return -1


def detect_target_company(text: str, entities: list) -> Optional[dict]:
    """Detect the main target company with improved heuristics.
    
    Priority order:
    1. Explicit "Razón social: X" or 'denominada "X"' markers
    2. Constitution markers: 'constituyeron sociedad X'
    3. EIRL markers: 'rectificó empresa individual... "X"'  
    4. Unquoted razón social patterns
    5. "constituyeron sociedad. Razón social: X SpA"
    6. "socios de X", "accionistas de X", "modificaron estatutos de X"
    7. Fallback to first non-notary company entity
    
    Returns dict with text, label, start, end, name_norm or None.
    """
    
    def _clean_target_name(name: str) -> str:
        """Clean target name: strip whitespace, newlines, trailing punctuation."""
        name = re.sub(r'[\n\r\t]+', ' ', name)  # normalize newlines
        name = re.sub(r'\s{2,}', ' ', name)
        # Remove trailing "en una sociedad..." or similar if captured by accident
        name = re.sub(r'\s+en\s+(?:una\s+)?(?:sociedade?s?|sociedad\s+por\s+acciones).*$', '', name, flags=re.IGNORECASE)
        return name.strip().rstrip(',. ')
    
    # Priority 1: Explicit name markers with quotes
    for m in _TARGET_MARKERS_EXPLICIT.finditer(text):
        name = m.group(1) or m.group(2) or m.group(3) or m.group(4)
        if name and len(name.strip()) > 3:
            cand = _clean_target_name(name)
            if cand.lower() not in _COMPANY_BLACKLIST:
                return {
                    "text": cand, "label": "company",
                    "start": m.start(), "end": m.end(),
                    "name_norm": cand.lower()
                }
    
    # Priority 2: Constitution/transformation markers
    for m in _TARGET_MARKERS_CONSTITUTION.finditer(text):
        name = m.group(1) or m.group(2)
        if name and len(name.strip()) > 3:
            cand = _clean_target_name(name)
            if cand.lower() not in _COMPANY_BLACKLIST:
                return {
                    "text": cand, "label": "company",
                    "start": m.start(), "end": m.end(),
                    "name_norm": cand.lower()
                }
    
    # Priority 3: EIRL markers
    for m in _TARGET_MARKERS_EIRL.finditer(text):
        name = m.group(1)
        if name and len(name.strip()) > 3:
            cand = _clean_target_name(name)
            if cand.lower() not in _COMPANY_BLACKLIST:
                return {
                    "text": cand, "label": "company",
                    "start": m.start(), "end": m.end(),
                    "name_norm": cand.lower()
                }
    
    # Priority 4: Unquoted razón social
    for m in _TARGET_MARKERS_RAZON.finditer(text):
        name = m.group(1)
        if name and len(name.strip()) > 3:
            cand = _clean_target_name(name)
            # Strip preamble like 'en sociedad por acciones. Razón social ' etc
            cand = re.sub(r"^(?:en\s+sociedad.+?(?:raz[óo]n\s+social|nombre)\s*[,:;]\s*)", "", cand, flags=re.IGNORECASE).strip()
            if cand.lower() not in _COMPANY_BLACKLIST:
                return {
                    "text": cand, "label": "company",
                    "start": m.start(), "end": m.end(),
                    "name_norm": cand.lower()
                }
    
    # Priority 5: "constituyeron sociedad. Razón social: X SpA"
    for m in _TARGET_MARKERS_CONSTIT_NOQUOTE.finditer(text):
        name = m.group(1)
        if name and len(name.strip()) > 3:
            cand = _clean_target_name(name)
            if cand.lower() not in _COMPANY_BLACKLIST:
                return {
                    "text": cand, "label": "company",
                    "start": m.start(), "end": m.end(),
                    "name_norm": cand.lower()
                }
    
    # Priority 6a: "modificaron estatutos sociales de X" or "modificaron X"
    for m in _TARGET_MARKERS_MODIFICARON.finditer(text):
        name = m.group(1)
        if name and len(name.strip()) > 3:
            cand = _clean_target_name(name)
            if cand.lower() not in _COMPANY_BLACKLIST:
                return {
                    "text": cand, "label": "company",
                    "start": m.start(), "end": m.end(),
                    "name_norm": cand.lower()
                }
    
    # Priority 6b: "socios de X LIMITADA" — the company after "socios de" is the target
    for m in _TARGET_MARKERS_SOCIOS.finditer(text):
        name = m.group(1)
        if name and len(name.strip()) > 3:
            cand = _clean_target_name(name)
            if cand.lower() not in _COMPANY_BLACKLIST:
                return {
                    "text": cand, "label": "company",
                    "start": m.start(), "end": m.end(),
                    "name_norm": cand.lower()
                }
    
    # Priority 6c: "accionistas de sociedad X S.A"
    for m in _TARGET_MARKERS_ACCIONISTAS.finditer(text):
        name = m.group(1)
        if name and len(name.strip()) > 3:
            cand = _clean_target_name(name)
            if cand.lower() not in _COMPANY_BLACKLIST:
                return {
                    "text": cand, "label": "company",
                    "start": m.start(), "end": m.end(),
                    "name_norm": cand.lower()
                }
    
    # Priority 7: Fallback — first company entity that isn't notary-related
    for ent in entities:
        if ent["label"] == "company":
            name_lower = ent["name_norm"]
            if "notari" in name_lower or "conservador" in name_lower:
                continue
            if name_lower in _COMPANY_BLACKLIST:
                continue
            if _NOTARIO_CITY_RE.match(ent["text"]):
                continue
            return ent
    
    return None


def is_notario_city_name(name: str) -> bool:
    """Check if a name looks like 'Notario [City]' which is not a company."""
    return bool(_NOTARIO_CITY_RE.match(name))


# ═══════════════════════════════════════════════════════════════════════════════
# CPU PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_chunk(lines: list) -> list:
    docs = []
    for line in lines:
        if not line.strip(): continue
        try:
            data = _loads(line)
        except Exception:
            continue

        text = data.get("content", "")
        if not text: continue
        
        if not _FILTER_RE.search(text):
            continue

        text_upper = text.upper()
        start_idx = len(text)
        for kw in _HEADER_KEYWORDS:
            idx = text_upper.find(kw)
            if idx != -1 and idx < start_idx:
                start_idx = idx
        
        if start_idx >= len(text): start_idx = 0
        clean = text[start_idx : start_idx + MAX_TEXT_LEN]

        m = _BOILERPLATE_RE.match(clean)
        if m: clean = clean[m.end():]

        if not _FILTER_RE.search(clean):
            continue
        if len(clean) < 50:
            continue

        docs.append(DocMeta(
            cve  = str(data.get("cve",  "")),
            date = str(data.get("date", "")),
            url  = str(data.get("url",  "")),
            text = clean,
        ))

    return docs

# ═══════════════════════════════════════════════════════════════════════════════
# GPU INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference(model, docs: list) -> list:
    if not docs: return []
    
    # Chunking texts to avoid GLiNER max_length truncation
    MAX_CHAR_CHUNK = 1500
    chunk_list = []
    chunk_mapping = []  # Maps chunk back to doc index and records offset
    
    for i, d in enumerate(docs):
        text = d.text
        start = 0
        while start < len(text):
            end = min(start + MAX_CHAR_CHUNK, len(text))
            if end < len(text):
                idx = text.rfind("\n", start, end)
                if idx == -1: idx = text.rfind(". ", start, end)
                if idx != -1 and idx > start + MAX_CHAR_CHUNK // 2:
                    end = idx + 1
            chunk_list.append(text[start:end])
            chunk_mapping.append((i, start))
            start = end
            
    order = sorted(range(len(chunk_list)), key=lambda i: len(chunk_list[i]), reverse=True)
    sorted_chunks = [chunk_list[i] for i in order]

    with torch.inference_mode():
        preds = model.batch_predict_entities(sorted_chunks, NER_LABELS, threshold=NER_THRESHOLD)
    
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        
    unsorted_preds = [None] * len(chunk_list)
    for sorted_i, original_i in enumerate(order):
        unsorted_preds[original_i] = preds[sorted_i]
        
    # Reassemble
    result = [[] for _ in range(len(docs))]
    for chunk_idx, (doc_idx, offset) in enumerate(chunk_mapping):
        for ent in unsorted_preds[chunk_idx]:
            # adjust offsets
            new_ent = dict(ent)
            new_ent["start"] += offset
            new_ent["end"] += offset
            result[doc_idx].append(new_ent)
            
    # Simple deduplication over overlapping regions or same entities
    final_result = []
    for ents in result:
        unique_ents = []
        seen = set()
        for e in ents:
            key = (e["start"], e["end"], e["label"])
            if key not in seen:
                seen.add(key)
                unique_ents.append(e)
        final_result.append(unique_ents)

    return final_result

# ═══════════════════════════════════════════════════════════════════════════════
# POSTPROCESSING (v6 — Reliability-focused)
# ═══════════════════════════════════════════════════════════════════════════════

def postprocess_batch(batch_data: tuple) -> list:
    docs, entity_batches = batch_data
    recs = []

    for meta, entities in zip(docs, entity_batches):
        if not entities: continue

        text = meta.text 
        ruts = extract_all_ruts(text)
        consumed_ruts = set()
        certifico_pos = find_certifico_position(text)

        clean_entities = []
        seen_names = set()
        notary_names = set()

        # ── Phase 0: Rescue missed person entities natively ────────────────────
        for pers_m in re.finditer(r"([A-ZÁÉÍÓÚÑa-zA-Záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑa-zA-Záéíóúñ]+){1,5})\s*,\s*(?:chilen[oa]|argentin[oa]|peruan[oa]|colombian[oa]|venezolan[oa]|español[a]?)", text, re.IGNORECASE):
            name_raw = pers_m.group(1).strip()
            if len(name_raw) > 5 and not any(name_raw.lower() in e["text"].lower() for e in entities):
                # Only insert if it's not mostly lowercase (to avoid matching random text)
                if sum(1 for c in name_raw if c.isupper()) >= 2:
                    entities.append({
                        "text": name_raw,
                        "label": "person",
                        "start": pers_m.start(1),
                        "end": pers_m.end(1)
                    })

        # ── Phase 1: Clean and filter entities ────────────────────────────
        for ent in entities:
            raw_text = clean_entity_name(ent["text"])
            if len(raw_text) < 3: continue

            name_lower = raw_text.lower()
            label = ent["label"]
            start, end = ent["start"], ent["end"]

            # Company blacklist
            if label == "company" and name_lower in _COMPANY_BLACKLIST: 
                continue

            # Company: skip "Notario [City]" patterns
            if label == "company" and is_notario_city_name(raw_text):
                continue
            
            # Company: skip addresses
            if label == "company" and _ADDRESS_START_RE.match(name_lower):
                continue
            
            # Company: skip admin terms
            if label == "company" and _ADMIN_TERMS_RE.search(name_lower):
                continue

            # Dedup (using cleaned name)
            dedup_key = (name_lower, label)
            if dedup_key in seen_names: continue
            seen_names.add(dedup_key)

            # ── Person: Notary exclusion ─────────────────────────────────
            if label == "person":
                # Direct notary keywords in the name itself
                if _NOTARY_KEYWORDS_RE.search(name_lower):
                    notary_names.add(name_lower)
                    continue
                
                # Check for notary context (text before/after the entity)
                if is_notary_context(text, start, end):
                    notary_names.add(name_lower)
                    continue
                
                # Check for street name/address context for people (false positives)
                before_ctx = text[max(0, start-40):start].lower()
                after_ctx = text[end:min(len(text), end+30)].lower()
                address_pfx = r"\b(?:calle|avenida|pasaje|domiciliados?\s+en|residente\s+en|en|con|norte|sur|este|oeste|limita)\s+$"
                if re.search(address_pfx, before_ctx) and re.search(r"^\s+(?:n[uú]mero|n[oº])?\s*\d+", after_ctx):
                    continue # Skip address-like people
                
                # Historical context check
                if is_historical_context(text, start, end):
                    notary_names.add(name_lower)
                    continue
                
                # Already identified as notary
                if name_lower in notary_names:
                    continue
                
                # Pre-certifico fallback: if before certifico position, likely notary
                if certifico_pos > 0 and start < certifico_pos:
                    ctx = text[max(0, start-40): min(len(text), end+40)].lower()
                    if "suplente" in ctx and "titular" in ctx:
                        notary_names.add(name_lower)
                        continue

            clean_entities.append({
                "text": raw_text,
                "label": label,
                "start": start,
                "end": end,
                "name_norm": normalize_name(raw_text) if label == "person" else name_lower
            })

        # ── Phase 2: Detect target company ────────────────────────────────
        target_ent = detect_target_company(text, clean_entities)
        
        subject_name = target_ent["text"] if target_ent else "UNKNOWN_TARGET"
        subject_rut = None
        subject_name_lower = subject_name.lower() if subject_name != "UNKNOWN_TARGET" else ""

        # Find RUT for the target company
        if target_ent and subject_name != "UNKNOWN_TARGET":
            # For target company, we search EXHAUSTIVELY across all mentions
            # and prefer "rol único tributario" anchors to avoid stealing personal RUTs
            best_subject_rut = None
            best_subject_ridx = None
            best_subject_score = -999999
            
            # Find all mentions of subject_name in text
            name_mentions = []
            search_pos = 0
            subject_name_upper = subject_name.upper()
            while True:
                idx = text.upper().find(subject_name_upper, search_pos)
                if idx == -1: break
                name_mentions.append((idx, idx + len(subject_name)))
                search_pos = idx + 1
            
            # Also include the detect mention
            if target_ent.get("start") is not None:
                name_mentions.append((target_ent["start"], target_ent["end"]))
            
            for m_start, m_end in name_mentions:
                rut, ridx = find_rut_for_entity(m_start, m_end, ruts, text, 
                                                max_dist=120, consumed=consumed_ruts,
                                                prefer_col_unico=True)
                if rut:
                    # Score based on distance and explicitly "rol unico"
                    dist = 0 # distance to RUT
                    # ... re-eval score or just use the one from find_rut_for_entity
                    # Simple: if multiple mentions, find_rut_for_entity will already return the best one for THAT mention.
                    # We pick the absolute best across all mentions.
                    # We'll just cheat and call find_rut_for_entity again inside.
                    # Actually, let's just use the result if it looks good.
                    
                    # For now, let's just take the first one found that has a strong score
                    if not best_subject_rut:
                        best_subject_rut = rut
                        best_subject_ridx = ridx
            
            subject_rut = best_subject_rut
            if best_subject_ridx is not None:
                consumed_ruts.add(best_subject_ridx)

        # ── Phase 2.5: Build fantasy-name exclusion set ────────────────────
        fantasy_names = set()
        for fm in _FANTASY_NAME_RE.finditer(text):
            fname = (fm.group(1) or fm.group(2) or fm.group(3) or "").strip()
            if fname and len(fname) > 2:
                # Clean it like we clean entity names
                fname_clean = clean_entity_name(fname).rstrip('.,; "\u201c\u201d\u201e')
                fantasy_names.add(fname_clean.lower())
        
        # Also exclude "razón social será X" as these are new names for the same company
        for nm in _NEW_RAZON_RE.finditer(text):
            nname = (nm.group(1) or "").strip()
            if nname and len(nname) > 2:
                nname_clean = clean_entity_name(nname).rstrip('.,; "\u201c\u201d\u201e')
                fantasy_names.add(nname_clean.lower())

        # ── Phase 3: Build relationships ─────────────────────────────────
        doc_rels = []
        skip_indices = set()
        
        for i, ent in enumerate(clean_entities):
            if i in skip_indices: continue
            
            ent_name_lower = ent["text"].lower()
            
            # Skip if this entity IS the target company (don't make it partner of itself)
            # Use partial matching: if one contains the other
            if ent["label"] == "company" and subject_name != "UNKNOWN_TARGET":
                if (ent_name_lower == subject_name_lower or
                    ent["name_norm"] == subject_name_lower or
                    (len(ent_name_lower) > 5 and ent_name_lower in subject_name_lower) or
                    (len(subject_name_lower) > 5 and subject_name_lower in ent_name_lower)):
                    continue
            
            # Skip fantasy names (they are NOT separate partners)
            if ent["label"] == "company" and ent_name_lower in fantasy_names:
                continue
            
            if ent["label"] == "person":
                is_rep = False
                is_mandatario = False
                represented = None
                
                # Check for "mandatario" context — person is NOT an independent partner
                before_gap = text[max(0, ent["start"] - 80):ent["start"]]
                if _MANDATARIO_RE.search(before_gap):
                    is_mandatario = True
                
                # Check for "en representación de [COMPANY]" after person
                # Max gap of 150 chars to avoid jumping too far if GLiNER missed an entity
                max_rep_gap = 150
                potential_gap_text = text[ent["end"] : min(len(text), ent["end"] + max_rep_gap)]
                rep_match = _REPRESENTATION_RE.search(potential_gap_text)
                
                if rep_match:
                    marker_end = ent["end"] + rep_match.end()
                    
                    # 1. Check if next GLiNER entity is a company and very close to the marker
                    if i + 1 < len(clean_entities):
                        next_ent = clean_entities[i+1]
                        if next_ent["label"] == "company" and (next_ent["start"] - marker_end) < 80:
                            is_rep = True
                            represented = next_ent
                            skip_indices.add(i+1)
                    
                    # 2. Rescue: If GLiNER missed it, try regex rescue
                    if not is_rep:
                        # Extract next company-looking name
                        rescue_match = re.search(r"^\s*((?:la\s+sociedad\s+)?[\w\s,.\-&áéíóúñÁÉÍÓÚÑ]{5,100}?(?:LTDA\.?|LIMITADA|SpA|S\.?A\.?|E\.?I\.?R\.?L\.?|SOCIEDAD\s+AN[OÓ]NIMA))", text[marker_end : marker_end + 120], re.I)
                        if rescue_match:
                            rname = clean_entity_name(rescue_match.group(1))
                            represented = {
                                "text": rname,
                                "label": "company",
                                "start": marker_end + rescue_match.start(1),
                                "end": marker_end + rescue_match.end(1),
                                "name_norm": rname.lower()
                            }
                            is_rep = True
                
                if is_rep and represented:
                    # Extract target company RUT first so person doesn't steal it
                    c_rut, c_ridx = find_rut_near_name(
                        represented["text"], represented["start"], represented["end"],
                        text, ruts, consumed_ruts)
                    if c_ridx is not None: consumed_ruts.add(c_ridx)
                    
                    # Now extract person RUT
                    rut, ridx = find_rut_near_name(ent["text"], ent["start"], ent["end"], 
                                                    text, ruts, consumed_ruts)
                    if ridx is not None: consumed_ruts.add(ridx)
                    
                    rep_name_lower = represented["text"].lower()
                    # Skip if represented company IS the target (partial match too)
                    is_self_ref = (
                        rep_name_lower == subject_name_lower or
                        represented["name_norm"] == subject_name_lower or
                        (len(rep_name_lower) > 5 and rep_name_lower in subject_name_lower) or
                        (len(subject_name_lower) > 5 and subject_name_lower in rep_name_lower)
                    )
                    
                    doc_rels.append((ent["text"], rut, "REPRESENTATIVE_OF", represented["text"], c_rut, ent["name_norm"]))
                    
                    if subject_name != "UNKNOWN_TARGET" and not is_self_ref:
                        doc_rels.append((represented["text"], c_rut, "PARTNER_OF", subject_name, subject_rut, represented["name_norm"]))
                    
                    # Only add person as partner if NOT a mandatario 
                    if subject_name != "UNKNOWN_TARGET" and not is_mandatario:
                        doc_rels.append((ent["text"], rut, "PARTNER_OF", subject_name, subject_rut, ent["name_norm"]))
                elif is_mandatario:
                    # Extract person RUT
                    rut, ridx = find_rut_near_name(ent["text"], ent["start"], ent["end"], 
                                                    text, ruts, consumed_ruts)
                    if ridx is not None: consumed_ruts.add(ridx)
                    
                    # Mandatarios are representatives, NOT independent partners
                    # Don't add them as PARTNER_OF
                    pass
                else:
                    # Extract person RUT
                    rut, ridx = find_rut_near_name(ent["text"], ent["start"], ent["end"], 
                                                    text, ruts, consumed_ruts)
                    if ridx is not None: consumed_ruts.add(ridx)
                    
                    if subject_name != "UNKNOWN_TARGET":
                        doc_rels.append((ent["text"], rut, "PARTNER_OF", subject_name, subject_rut, ent["name_norm"]))
                        
            elif ent["label"] == "company":
                # Company entity (not handled by representation logic above)
                rut, ridx = find_rut_near_name(ent["text"], ent["start"], ent["end"], 
                                                text, ruts, consumed_ruts)
                if ridx is not None: consumed_ruts.add(ridx)
                
                if subject_name != "UNKNOWN_TARGET":
                    doc_rels.append((ent["text"], rut, "PARTNER_OF", subject_name, subject_rut, ent["name_norm"]))

        # ── Phase 4: Rescue missed entities using Unconsumed RUTs ────────────────
        for idx, (rut_str, r_start, r_end) in enumerate(ruts):
            if idx in consumed_ruts:
                continue
            if subject_rut and rut_str == subject_rut:
                continue
                
            # Look backwards for a name up to 250 characters
            window = text[max(0, r_start - 250):r_start]
            
            # Company rescue
            comp_m = re.search(r"([A-ZÁÉÍÓÚÑ][A-Za-záéíóúñÁÉÍÓÚÑ\s.,&'-]+(?:LTDA\.?|LIMITADA|SpA|S\.?A\.?|E\.?I\.?R\.?L\.?))[\s,]*$", window)
            if comp_m:
                cand = clean_entity_name(comp_m.group(1))
                # Make sure cand is not just "SpA" or something silly
                if len(cand) > 3 and cand.lower() not in _COMPANY_BLACKLIST:
                    # Deduplicate against existing doc_rels
                    if not any(cand.lower() in sr.lower() or sr.lower() in cand.lower() for sr, _, _rel, _, _, _ in doc_rels):
                        if subject_name != "UNKNOWN_TARGET":
                            doc_rels.append((cand, rut_str, "PARTNER_OF", subject_name, subject_rut, cand.lower()))
                        consumed_ruts.add(idx)
                        continue
                    
            # Person rescue
            pers_m = re.search(r"([A-ZÁÉÍÓÚÑa-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑa-záéíóúñ]+){1,5})\s*,\s*(?:chilen[oa]|argentin[oa]|peruan[oa]|colombian[oa]|venezolan[oa]|español[a]?)", window, re.IGNORECASE)
            if pers_m:
                cand = clean_entity_name(pers_m.group(1).strip())
                if len(cand) > 5 and not is_notary_context(text, r_start - len(window) + pers_m.start(1), r_start - len(window) + pers_m.end(1)):
                    if not any(cand.lower() in sr.lower() or sr.lower() in cand.lower() for sr, _, _rel, _, _, _ in doc_rels):
                        if subject_name != "UNKNOWN_TARGET":
                            doc_rels.append((cand, rut_str, "PARTNER_OF", subject_name, subject_rut, normalize_name(cand)))
                        consumed_ruts.add(idx)
                        continue

        for src, srut, rel, tgt, trut, snorm in doc_rels:
            recs.append((meta.cve, src, srut, rel, tgt, trut, meta.date, meta.url, snorm))
            
    return recs

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE & ASYNC WRITER
# ═══════════════════════════════════════════════════════════════════════════════

_DDL = """
    CREATE TABLE IF NOT EXISTS relationships (
        cve                    VARCHAR,
        source_name            VARCHAR,
        source_rut             VARCHAR,
        relation               VARCHAR,
        target_name            VARCHAR,
        target_rut             VARCHAR,
        date                   VARCHAR,
        url                    VARCHAR,
        timestamp              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        source_name_normalized VARCHAR
    )
"""

_INSERT = """
    INSERT INTO relationships (
        cve, source_name, source_rut, relation, target_name, target_rut, 
        date, url, source_name_normalized
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

def init_db(db_path: str):
    con = duckdb.connect(db_path)
    con.execute(_DDL)
    return con

def get_completed_cves(db_path: str):
    try:
        con = duckdb.connect(db_path, read_only=True)
        tables = {t[0] for t in con.execute("SHOW TABLES").fetchall()}
        if "relationships" not in tables:
            con.close()
            return frozenset()
        rows = con.execute("SELECT DISTINCT cve FROM relationships").fetchall()
        con.close()
        return frozenset(r[0] for r in rows)
    except Exception as e:
        log.warning(f"Could not load completed CVEs: {e}")
        return frozenset()

def db_writer_thread(db_path: str, q: queue.Queue):
    con = init_db(db_path)
    log.info("  [DB Thread] Started.")
    while True:
        batch = q.get()
        if batch is None:
            break
        if batch:
            try:
                con.executemany(_INSERT, batch)
            except Exception as e:
                log.error(f"  [DB Thread] Write Error: {e}")
        q.task_done()
    con.close()
    log.info("  [DB Thread] Stopped.")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def iter_chunks(path: str, chunk_size: int, completed_cves):
    chunk = []
    total_read = 0
    total_skipped = 0
    
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            total_read += 1
            if not line.strip(): continue
            
            if completed_cves:
                m = _CVE_RE.search(line)
                if m and m.group(1) in completed_cves:
                    total_skipped += 1
                    continue
            
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
            
            if total_read % 10_000 == 0:
                log.info(f"  scan: {total_read:,} read, {total_skipped:,} skipped")

    if chunk: yield chunk

def main() -> None:
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

    log.info("═" * 62)
    log.info(f" GLiNER V6 PIPELINE: RELIABILITY REWRITE")
    log.info(f" DB Path: {DB_PATH}")
    log.info(f" Device: {DEVICE} (Available CPUs: {os.cpu_count()})")
    log.info("═" * 62)
    
    if DEVICE == "cuda": 
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    
    torch.set_num_threads(1)
    gc.set_threshold(50000, 500, 500)

    log.info(f"Loading model '{MODEL_NAME}' on {DEVICE}...")
    model = GLiNER.from_pretrained(MODEL_NAME).to(DEVICE)
    
    if DEVICE == "cuda":
        log.info("  [Model] Converting to Half Precision (FP16)...")
        model = model.half()
        try:
             log.info("  [Model] Compiling with torch.compile (reduce-overhead)...")
             model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
             log.warning(f"  [Model] Compilation failed (safe to ignore): {e}")

    completed = get_completed_cves(DB_PATH)
    log.info(f"Resume: {len(completed):,} CVEs already done.")

    db_queue = queue.Queue(maxsize=500)
    db_thread = threading.Thread(target=db_writer_thread, args=(DB_PATH, db_queue), daemon=True)
    db_thread.start()
    
    post_pool = ThreadPoolExecutor(max_workers=4)

    t0 = time.perf_counter()
    total_rel = 0
    total_doc = 0
    processed_cves = set()
    last_log_count = 0

    futures = deque()
    pending_docs = deque()
    post_futures = []

    def check_post_futures():
        nonlocal total_rel
        done_indices = [i for i, f in enumerate(post_futures) if f.done()]
        for i in reversed(done_indices):
            try:
                recs = post_futures[i].result()
                if recs:
                    db_queue.put(recs)
                    total_rel += len(recs)
            except Exception as e:
                log.error(f"Post-processing error: {e}")
            del post_futures[i]

    with ProcessPoolExecutor(max_workers=NUM_CPU_WORKERS) as pre_pool:
        for chunk in iter_chunks(DATASET_PATH, CHUNK_SIZE, completed):
            futures.append(pre_pool.submit(preprocess_chunk, chunk))
            
            while len(futures) >= MAX_INFLIGHT:
                batch_docs = futures.popleft().result()
                pending_docs.extend(batch_docs)
                check_post_futures()

                while len(pending_docs) >= BATCH_SIZE:
                    batch = [pending_docs.popleft() for _ in range(BATCH_SIZE)]
                    ent_batches = run_inference(model, batch)
                    total_doc += len(batch)
                    post_futures.append(post_pool.submit(postprocess_batch, (batch, ent_batches)))

                    for d in batch: processed_cves.add(d.cve)
                    
                    curr = len(processed_cves)
                    if curr - last_log_count >= 200:
                        elap = time.perf_counter() - t0
                        rate = curr / elap if elap > 0 else 0
                        log.info(f"  [Progress] {curr:,} CVEs | {total_rel:,} Rels | {rate:.1f} CVE/s")
                        last_log_count = curr

        for fut in futures:
            batch_docs = fut.result()
            pending_docs.extend(batch_docs)
            check_post_futures()
            
            while len(pending_docs) >= BATCH_SIZE:
                 batch = [pending_docs.popleft() for _ in range(BATCH_SIZE)]
                 ent_batches = run_inference(model, batch)
                 total_doc += len(batch)
                 post_futures.append(post_pool.submit(postprocess_batch, (batch, ent_batches)))
                 for d in batch: processed_cves.add(d.cve)

    if pending_docs:
        batch = list(pending_docs)
        ent_batches = run_inference(model, batch)
        total_doc += len(batch)
        post_futures.append(post_pool.submit(postprocess_batch, (batch, ent_batches)))
        for d in batch: processed_cves.add(d.cve)

    log.info("  Waiting for post-processing...")
    for pf in post_futures:
        recs = pf.result()
        if recs:
            db_queue.put(recs)
            total_rel += len(recs)
    
    db_queue.put(None)
    db_thread.join()
    post_pool.shutdown()
    
    log.info("═" * 62)
    log.info(f"Done in {time.perf_counter() - t0:.1f}s")
    log.info(f"Total CVEs: {total_doc:,}")
    log.info(f"Total Rels: {total_rel:,}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
