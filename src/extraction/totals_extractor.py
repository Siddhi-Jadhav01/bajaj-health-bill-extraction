import re
from src.extraction.row_classifier import find_numeric_tokens, parse_number_from_text

TOTAL_KEYWORDS = [
    "total", "sub total", "subtotal", "grand total", "net payable",
    "amount payable", "amount due", "net amount", "payable",
    "bill amount", "amount before tax"
]

def is_total_like(text: str) -> bool:
    if not text:
        return False
    t = text.lower().strip()
    return any(k in t for k in TOTAL_KEYWORDS)

def extract_total_value(text: str):
    """
    Pull numeric value from OCR total-like row.
    """
    n = parse_number_from_text(text)
    return n
