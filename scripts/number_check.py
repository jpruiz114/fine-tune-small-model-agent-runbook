"""Find numbers in text and normalize them so "$12.5M" and "12.5 million" compare equal."""
import re
from decimal import Decimal, InvalidOperation

SCALES = {
    "k": 10**3, "thousand": 10**3,
    "m": 10**6, "mm": 10**6, "mn": 10**6, "million": 10**6,
    "b": 10**9, "bn": 10**9, "billion": 10**9,
    "t": 10**12, "tn": 10**12, "trillion": 10**12,
}
# "one" is left out on purpose: it is too often a pronoun ("one workflow").
WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "dozen": 12,
}
SCALE_RE = r"(?:thousand|million|billion|trillion|mm|mn|bn|tn|k|m|b|t)(?![a-z])"
UNIT_RE = r"(?:%|percent(?![a-z])|per cent(?![a-z])|x(?![a-z]))"
DIGIT_RE = re.compile(
    rf"(?<![\w.])[$€£₹]?\s?(\d+(?:,\d{{3}})*(?:\.\d+)?)\s?({SCALE_RE})?\s?({UNIT_RE})?",
    re.IGNORECASE,
)
WORD_RE = re.compile(
    rf"\b({'|'.join(WORDS)})(?:-({'|'.join(WORDS)}))?\b(?:\s({SCALE_RE}))?\s?({UNIT_RE})?",
    re.IGNORECASE,
)


def _canonical(value: Decimal, scale: str | None, unit: str | None) -> str:
    if scale:
        value *= SCALES[scale.lower()]
    text = format(value.normalize(), "f")
    if unit:
        unit = unit.lower()
        text += "%" if unit.startswith("per") or unit == "%" else "x"
    return text


def extract_numbers(text: str) -> set[str]:
    """Return the set of canonical numbers in text, e.g. {"12500000", "40%", "10x"}."""
    found = set()
    for m in DIGIT_RE.finditer(text):
        try:
            value = Decimal(m.group(1).replace(",", ""))
        except InvalidOperation:
            continue
        found.add(_canonical(value, m.group(2), m.group(3)))
    for m in WORD_RE.finditer(text):
        value = Decimal(WORDS[m.group(1).lower()])
        if m.group(2):
            value += WORDS[m.group(2).lower()]
        found.add(_canonical(value, m.group(3), m.group(4)))
    return found


def unsupported_numbers(source: str, generated: str) -> set[str]:
    """Numbers in generated that do not appear anywhere in source."""
    return extract_numbers(generated) - extract_numbers(source)
