# src/extraction/row_classifier.py
import re

# Blocking keywords — rows containing these are almost never line items
BLOCKING_KEYWORDS = [
    "total", "subtotal", "grand total", "net amount", "net payable",
    "amount payable", "amount due", "tax", "discount", "round off",
    "gst", "service charge", "bill total", "invoice total", "page total",
    "balance", "paid", "cash", "card", "remarks", "signature", "authorized"
]

# Accept keywords that often indicate line items (if present + numeric)
POSITIVE_KEYWORDS = ["qty", "rate", "amount", "description", "particular", "item"]

# numeric detection (floats, commas, currency symbols, parentheses for negatives)
NUM_RE = re.compile(r"[-+]?\(?\d{1,3}(?:[,\d]{0,}\d)?(?:\.\d+)?\)?")

def _clean_text(s):
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip().lower()

def contains_blocking_keyword(text):
    t = _clean_text(text)
    tokens = t.split()

    # allow matching multi-word phrases like "grand total"
    t_spaced = " " + t + " "

    for k in BLOCKING_KEYWORDS:
        # exact token match
        if k in tokens:
            return True
        
        # phrase match but boundary-safe
        if f" {k} " in t_spaced:
            return True

    return False


def find_numeric_tokens(text):
    if not text:
        return []
    return NUM_RE.findall(text.replace(",", ""))

def parse_number_from_text(text):
    # Extract first numeric token and parse
    toks = find_numeric_tokens(text)
    if not toks:
        return None
    tok = toks[0]
    tok = tok.replace("(", "-").replace(")", "")
    tok = tok.replace(",", "")
    try:
        return float(tok)
    except:
        return None

def is_header_like(row_cells):
    # If most cells are short tokens like 's.no', 'date', 'particulars' etc → header
    header_tokens = 0
    total = max(1, len(row_cells))
    for c in row_cells:
        if not c:
            continue
        c_ = _clean_text(c)
        # if contains positive header words
        if any(h in c_ for h in ["s.no", "s.no.", "date", "particular", "particulars", "qty", "rate", "amt", "amount", "description"]):
            header_tokens += 1
    return header_tokens >= max(1, total // 2)

def is_too_short(row_cells, min_words=2):
    # row considered noise if too few alphabetic tokens across cells
    joined = " ".join([c or "" for c in row_cells])
    words = re.findall(r"[A-Za-z0-9]{2,}", joined)
    return len(words) < min_words

def is_line_item(row_cells, mapping=None, ocr_row_text=None, debug=False):
    """
    Deterministic rule-based classifier.
    - row_cells: list[str] (cells)
    - mapping: dict mapping like {"item_amount": idx, "item_name": idx, ...} (optional)
    - ocr_row_text: joined raw row text (optional)
    Returns: (bool, reason_str)
    """
    joined = " ".join([c or "" for c in row_cells]).strip()
    raw = ocr_row_text or joined
    jclean = _clean_text(joined)
    rclean = _clean_text(raw)

    # 0) quick reject: empty row
    if not jclean:
        return False, "empty"

    # 1) blocking keywords
    if contains_blocking_keyword(rclean):
        return False, "has_blocking_keyword"

    # 2) header-like rows -> reject
    if is_header_like(row_cells):
        return False, "header_like"

    # 3) too short -> reject
    if is_too_short(row_cells):
        return False, "too_short"

    # 4) if mapping provided, check amount column first
    amount_val = None
    if mapping and mapping.get("item_amount") is not None:
        idx = mapping["item_amount"]
        if idx < len(row_cells):
            amount_val = parse_number_from_text(row_cells[idx])
            if amount_val is not None:
                # positive numeric and not a blocking row -> accept
                return True, f"has_amount_in_mapped_col:{amount_val}"

    # 5) fallback: look for numeric tokens anywhere (prioritize right-most numeric)
    nums = []
    for c in reversed(row_cells):
        n = parse_number_from_text(c)
        if n is not None:
            nums.append(n)
            break
    if nums:
        return True, f"has_amount_anywhere:{nums[0]}"

    # 6) weak positive signals: contains 'qty' or 'rate' words -> candidate
    if any(pk in jclean for pk in POSITIVE_KEYWORDS):
        return True, "positive_keyword"

    # 7) default: reject
    return False, "no_numeric_or_keyword"
