"""
Spanish Written-RUT Parser
===========================
Extracts RUTs written as Spanish number-words from legal documents.

Example:
  "cédula de identidad número seis millones quinientos sesenta y dos mil
   seiscientos noventa guion K"  →  "6562690-K"

Usage:
  from rut_word_parser import extract_ruts_from_words
  results = extract_ruts_from_words(text)
  # → [("6562690-K", 34, 128), ...]
"""
from __future__ import annotations
import re
from typing import List, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# SPANISH NUMBER WORD → DIGIT MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

_UNITS = {
    "cero": 0, "uno": 1, "una": 1, "un": 1, "dos": 2, "tres": 3,
    "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
    "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14,
    "quince": 15, "dieciséis": 16, "dieciseis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19,
    "veinte": 20, "veintiuno": 21, "veintiún": 21, "veintidós": 22,
    "veintidos": 22, "veintitrés": 23, "veintitres": 23,
    "veinticuatro": 24, "veinticinco": 25, "veintiséis": 26,
    "veintiseis": 26, "veintisiete": 27, "veintiocho": 28,
    "veintinueve": 29,
}

_TENS = {
    "treinta": 30, "cuarenta": 40, "cincuenta": 50,
    "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90,
}

_HUNDREDS = {
    "cien": 100, "ciento": 100, "doscientos": 200, "doscientas": 200,
    "trescientos": 300, "trescientas": 300, "cuatrocientos": 400,
    "cuatrocientas": 400, "quinientos": 500, "quinientas": 500,
    "seiscientos": 600, "seiscientas": 600, "setecientos": 700,
    "setecientas": 700, "ochocientos": 800, "ochocientas": 800,
    "novecientos": 900, "novecientas": 900,
}

_MULTIPLIERS = {
    "mil": 1_000,
    "millón": 1_000_000, "millon": 1_000_000,
    "millones": 1_000_000,
}

# DV words → character
_DV_MAP = {
    "cero": "0", "uno": "1", "una": "1", "un": "1", "dos": "2",
    "tres": "3", "cuatro": "4", "cinco": "5", "seis": "6",
    "siete": "7", "ocho": "8", "nueve": "9", "k": "K",
}

# All known number words for tokenization
_ALL_NUMBER_WORDS = set()
_ALL_NUMBER_WORDS.update(_UNITS.keys())
_ALL_NUMBER_WORDS.update(_TENS.keys())
_ALL_NUMBER_WORDS.update(_HUNDREDS.keys())
_ALL_NUMBER_WORDS.update(_MULTIPLIERS.keys())
_ALL_NUMBER_WORDS.add("y")  # connector


def _parse_spanish_number(words: List[str]) -> int:
    """
    Parse a list of Spanish number words into an integer.
    
    Algorithm: accumulate values, handling mil/millones as multipliers.
    Example: ["seis", "millones", "quinientos", "sesenta", "y", "dos", "mil",
              "seiscientos", "noventa"] → 6_562_690
    """
    if not words:
        return 0

    total = 0
    current = 0  # accumulates the current group (before a multiplier)

    for word in words:
        w = word.lower()

        if w == "y":
            continue  # skip connector

        if w in _UNITS:
            current += _UNITS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w in _HUNDREDS:
            current += _HUNDREDS[w]
        elif w in _MULTIPLIERS:
            mul = _MULTIPLIERS[w]
            if mul == 1_000_000:
                # "seis millones" → current (6) * 1_000_000
                if current == 0:
                    current = 1
                total += current * mul
                current = 0
            elif mul == 1_000:
                # "quinientos sesenta y dos mil" → current (562) * 1000
                if current == 0:
                    current = 1
                total += current * mul
                current = 0

    total += current  # add remainder (e.g., the 690 in 6_562_690)
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT ANCHORS — markers that precede a written RUT
# ═══════════════════════════════════════════════════════════════════════════════

_RUT_CONTEXT_RE = re.compile(
    r"(?:"
    r"c[ée]dula\s+de\s+identidad\s+n[uú]mero"
    r"|rol\s+[uú]nico\s+tributario\s+n[uú]mero"
    r"|R\.?\s*U\.?\s*T\.?\s*n[uú]mero"
    r"|C\.?\s*I\.?\s*N[°º]?"
    r")\s+",
    re.IGNORECASE,
)

# Separator between number body and DV
_GUION_RE = re.compile(r"\bgu[ií][oó]n\b", re.IGNORECASE)


def _tokenize_number_region(text: str) -> Tuple[List[str], str, int]:
    """
    Starting from the beginning of `text`, consume Spanish number words
    until we hit the guion/separator.
    
    Returns:
        (number_words, dv_char, end_position)
    """
    words = []
    pos = 0
    text_len = len(text)

    while pos < text_len:
        # Skip whitespace and commas
        while pos < text_len and text[pos] in " \t\n\r,":
            pos += 1
        if pos >= text_len:
            break

        # Check for guion (separator)
        guion_match = _GUION_RE.match(text, pos)
        if guion_match:
            # Found separator — next word should be the DV
            pos = guion_match.end()
            # Skip whitespace
            while pos < text_len and text[pos] in " \t\n\r":
                pos += 1

            # Read DV: single letter K or a number word
            dv_char = None
            if pos < text_len:
                # Check for K/k
                if text[pos].upper() == "K" and (pos + 1 >= text_len or not text[pos + 1].isalpha()):
                    dv_char = "K"
                    pos += 1
                else:
                    # Try to read a DV word
                    for dv_word, dv_val in _DV_MAP.items():
                        if dv_word == "k":
                            continue  # already checked
                        wlen = len(dv_word)
                        if pos + wlen <= text_len:
                            candidate = text[pos:pos + wlen].lower()
                            if candidate == dv_word:
                                # Make sure it's a whole word
                                if pos + wlen >= text_len or not text[pos + wlen].isalpha():
                                    dv_char = dv_val
                                    pos += wlen
                                    break

            return words, dv_char, pos

        # Try to match a number word
        matched = False
        # Check longest words first to avoid partial matches
        remaining = text[pos:]
        remaining_lower = remaining.lower()

        best_word = None
        best_len = 0
        for nw in _ALL_NUMBER_WORDS:
            nw_len = len(nw)
            if nw_len > best_len and remaining_lower.startswith(nw):
                # Ensure whole word boundary
                if nw_len >= len(remaining) or not remaining[nw_len].isalpha():
                    best_word = nw
                    best_len = nw_len

        if best_word:
            words.append(best_word)
            pos += best_len
            matched = True

        if not matched:
            # Not a number word and not guion — stop consuming
            break

    return words, None, pos


def extract_ruts_from_words(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract RUTs written as Spanish number-words from text.
    
    Returns list of (normalized_rut, start_pos, end_pos) tuples,
    where start_pos is the start of the context anchor ("cédula de...").
    """
    results = []

    for anchor in _RUT_CONTEXT_RE.finditer(text):
        region_start = anchor.start()
        number_text_start = anchor.end()
        remaining = text[number_text_start:]

        number_words, dv_char, consumed = _tokenize_number_region(remaining)

        if not number_words or dv_char is None:
            continue

        body = _parse_spanish_number(number_words)
        if body < 1_000 or body > 99_999_999:
            # RUTs are typically 1M–99M range, minimum 1000
            continue

        rut_str = f"{body}-{dv_char}"
        end_pos = number_text_start + consumed
        results.append((rut_str, region_start, end_pos))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Quick self-test
    test_texts = [
        "cédula de identidad número seis millones quinientos sesenta y dos mil seiscientos noventa guion K",
        "rol único tributario número setenta y seis millones diecinueve mil trescientos veintiséis guion siete",
        "cédula de identidad número nueve millones quinientos dieciocho mil ciento setenta y ocho guión tres",
        "cédula de identidad número dieciocho millones novecientos cuarenta y dos mil ochocientos nueve guión K",
        "cédula de identidad número ocho millones seiscientos cincuenta y seis mil cuatrocientos veintidós guión K",
        "cédula de identidad número ocho millones quinientos diez mil ochocientos sesenta y cinco guión cuatro",
    ]
    expected = [
        "6562690-K",
        "76019326-7",
        "9518178-3",
        "18942809-K",
        "8656422-K",
        "8510865-4",
    ]

    print("Written-RUT Parser Self-Test")
    print("=" * 50)
    all_pass = True
    for txt, exp in zip(test_texts, expected):
        results = extract_ruts_from_words(txt)
        if results:
            got = results[0][0]
            status = "✓" if got == exp else "✗"
            if got != exp:
                all_pass = False
            print(f"  {status}  {exp:>12s}  got {got:>12s}")
        else:
            all_pass = False
            print(f"  ✗  {exp:>12s}  got NOTHING")

    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
