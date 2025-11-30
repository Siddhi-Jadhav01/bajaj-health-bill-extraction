# src/extraction/table_extractor.py
import re
import numpy as np
from src.extraction.row_classifier import is_line_item


NUMERIC_CHARS_RE = re.compile(r"[-\d\.,()]+")
CURRENCY_CLEAN_RE = re.compile(r"[^\d\.\-\(\)]")  # allow digits, dot, minus, parentheses

# ---------------------------
# Utilities
# ---------------------------
def is_number_like(s: str) -> bool:
    if not s or not isinstance(s, str):
        return False
    s = s.strip()
    # quick heuristic: contains digit after removing common noise
    cleaned = CURRENCY_CLEAN_RE.sub("", s)
    return bool(re.search(r"\d", cleaned))


def parse_number(s: str):
    """Robustly parse numeric tokens to float; return None if not parseable."""
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    # remove common OCR junk characters around numbers
    s = s.replace("\u201c", "").replace("“", "").replace("’", "").replace("`", "").replace("'", "")
    s = s.replace("O", "0")  # common OCR O->0
    # Normalize spaces around x multiply sign
    s = s.replace("×", "x")
    # If we see multiple tokens like "1180.00 x 1.00", we won't blindly parse whole string here.
    # We'll try to extract explicit numeric tokens below.
    # Replace thousands separators: remove commas if dot also present OR comma used as thousands.
    # If comma present & no dot, treat comma as decimal separator (e.g., 240,00)
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    # Remove currency symbols and stray characters except - . ()
    s = CURRENCY_CLEAN_RE.sub("", s)
    if s == "":
        return None
    # parenthesis maybe negative amounts
    try:
        if "(" in s and ")" in s:
            s = "-" + s.replace("(", "").replace(")", "")
        return float(s)
    except Exception:
        # last resort: extract first float-looking substring
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
        return None


def extract_all_numbers(text: str):
    """Return list of parsed numbers (floats) found in text, in order."""
    if not text or not isinstance(text, str):
        return []
    # Replace comma thousands like "1,500.00" handled by parse_number, so keep tokens with commas/dots
    # Find sequences containing digits, dots, commas, parentheses and optional leading minus
    tokens = re.findall(r"[-]?\d[\d,\.()]*", text.replace("\u2018", "").replace("\u2019", ""))
    nums = []
    for t in tokens:
        v = parse_number(t)
        if v is not None:
            nums.append(v)
    return nums


def extract_last_number(text: str):
    """Return last numeric value in text or None."""
    nums = extract_all_numbers(text)
    return nums[-1] if nums else None


# ---------------------------
# Filtering & region helpers
# ---------------------------
def filter_words_by_conf(words, min_conf=50):
    """Keep only words with conf >= min_conf or conf is None (be permissive)."""
    filtered = []
    for w in words:
        conf = w.get("conf", None)
        if conf is None or conf >= min_conf:
            filtered.append(w)
    return filtered


def shrink_region(region, shrink_ratio=0.02):
    """Shrink bounding box slightly to avoid catching headings/totals on edges."""
    w = max(1, region["x2"] - region["x1"])
    h = max(1, region["y2"] - region["y1"])
    dx = int(w * shrink_ratio)
    dy = int(h * shrink_ratio)
    return {
        "x1": region["x1"] + dx,
        "y1": region["y1"] + dy,
        "x2": region["x2"] - dx,
        "y2": region["y2"] - dy,
        "type": region.get("type", "table"),
    }


def word_in_region(word_box, region):
    wx1, wy1, wx2, wy2 = word_box
    cx = (wx1 + wx2) // 2
    cy = (wy1 + wy2) // 2
    return region["x1"] <= cx <= region["x2"] and region["y1"] <= cy <= region["y2"]


# ---------------------------
# Row clustering (adaptive)
# ---------------------------
def cluster_words_into_rows(words, y_tol=None):
    """
    Cluster words into rows. If y_tol not provided, compute median height * 0.6
    to adapt to different resolutions.
    """
    if not words:
        return []

    words_sorted = sorted(words, key=lambda w: (w["box"][1], w["box"][0]))

    # estimate typical word height
    heights = [(w["box"][3] - w["box"][1]) for w in words_sorted if (w["box"][3] - w["box"][1]) > 0]
    median_h = int(np.median(heights)) if heights else 12
    if y_tol is None:
        y_tol = max(8, int(median_h * 0.6))

    rows = []
    current_row = []
    last_y = None

    for w in words_sorted:
        mid_y = (w["box"][1] + w["box"][3]) / 2
        if last_y is None or abs(mid_y - last_y) <= y_tol:
            current_row.append(w)
        else:
            rows.append(current_row)
            current_row = [w]
        last_y = mid_y

    if current_row:
        rows.append(current_row)
    return rows


def sort_row_by_x(row):
    return sorted(row, key=lambda w: w["box"][0])


# ---------------------------
# Column inference
# ---------------------------
def infer_columns(rows, x_threshold=50):
    """
    Use k-means-lite like clustering on mid-x positions to form column centers.
    """
    mids = []
    for r in rows:
        for w in r:
            x1, _, x2, _ = w["box"]
            mids.append((x1 + x2) // 2)
    if not mids:
        return []

    mids = sorted(mids)
    clusters = []
    current = [mids[0]]
    for x in mids[1:]:
        if abs(x - current[-1]) <= x_threshold:
            current.append(x)
        else:
            clusters.append(int(sum(current) / len(current)))
            current = [x]
    clusters.append(int(sum(current) / len(current)))
    return clusters


def split_row_into_cells(row, columns):
    """Assign words to nearest column center; always return list length=len(columns)."""
    if not columns:
        # fallback: treat entire row as single cell
        return [" ".join([w["text"] for w in sort_row_by_x(row)])]
    cells = [""] * len(columns)
    for w in row:
        x1, _, x2, _ = w["box"]
        mid = (x1 + x2) // 2
        # choose nearest column
        idx = min(range(len(columns)), key=lambda i: abs(mid - columns[i]))
        cells[idx] = (cells[idx] + " " + w["text"]).strip()
    return [c.strip() for c in cells]


# ---------------------------
# Column mapping heuristics
# ---------------------------
def pick_amount_column(rows_cells):
    """Choose the rightmost column that is numeric for most rows."""
    if not rows_cells:
        return None
    num_cols = max(len(r) for r in rows_cells)
    scores = [0] * num_cols
    for r in rows_cells:
        for i in range(num_cols):
            val = r[i] if i < len(r) else ""
            # numeric if contains number-like OR contains pattern like 'xxx x yyy' (rate x qty)
            if is_number_like(val) or re.search(r"\d+\s*[x×]\s*\d", val):
                scores[i] += 1
    # pick rightmost among top scoring columns
    best_idx = None
    best_score = -1
    for i, s in enumerate(scores):
        if s > best_score or (s == best_score and (best_idx is None or i > best_idx)):
            best_idx = i
            best_score = s
    if best_score <= 0:
        return None
    return best_idx


def pick_qty_rate_columns(rows_cells, amount_idx):
    """
    Heuristic:
    - Look left of amount column for columns that often contain 'x' or small numbers.
    - Return (qty_idx, rate_idx)
    """
    if amount_idx is None:
        return None, None
    num_cols = max(len(r) for r in rows_cells) if rows_cells else 0
    qty_idx = None
    rate_idx = None

    # Search immediate left columns for qty pattern first (has 'x' or many small ints)
    for i in range(amount_idx - 1, -1, -1):
        col_vals = [r[i] for r in rows_cells if i < len(r)]
        if not col_vals:
            continue
        contains_x_frac = sum(1 for v in col_vals if re.search(r"\d+\s*[x×]\s*\d+", v)) / max(1, len(col_vals))
        numeric_frac = sum(1 for v in col_vals if is_number_like(v)) / max(1, len(col_vals))
        # If many values contain 'x' or numeric and shorter width, mark as qty
        if contains_x_frac > 0.05 or (numeric_frac > 0.3 and all(len(v) < 20 for v in col_vals)):
            qty_idx = i
            break

    # rate likely left of qty; find the nearest left column with numeric fraction
    if qty_idx is not None:
        for j in range(qty_idx - 1, -1, -1):
            col_vals = [r[j] for r in rows_cells if j < len(r)]
            numeric_frac = sum(1 for v in col_vals if is_number_like(v)) / max(1, len(col_vals))
            if numeric_frac > 0.2:
                rate_idx = j
                break
    else:
        # fallback: look for any column left of amount that looks numeric (candidate rate/qty)
        for j in range(amount_idx - 1, -1, -1):
            col_vals = [r[j] for r in rows_cells if j < len(r)]
            numeric_frac = sum(1 for v in col_vals if is_number_like(v)) / max(1, len(col_vals))
            if numeric_frac > 0.3:
                rate_idx = j
                break

    return qty_idx, rate_idx


def detect_header_row(rows_cells):
    """Find header row by looking for keywords (particulars, qty, rate, amt)."""
    keywords = ["partic", "item", "description", "qty", "rate", "amt", "amount"]
    for idx, row in enumerate(rows_cells[:3]):  # check first up to 3 rows
        joined = " ".join([c.lower() for c in row if c]).lower()
        if any(k in joined for k in keywords):
            return idx
    # fallback: if the first row contains non-numeric mostly (likely header), pick 0
    return 0


# ---------------------------
# Row cleaning and numeric parsing
# ---------------------------
def is_total_row(text):
    if not text:
        return False
    t = text.lower()
    for kw in ["total", "subtotal", "grand total", "net amount", "amount payable", "total of", "grand", "subtotal:"]:
        if kw in t:
            return True
    return False


def clean_cell_text(s):
    if s is None:
        return ""
    return " ".join(s.strip().split())


def is_header_like_row(rc):
    """Return True if row looks like a header (S.No, Date, Particulars, Qty, Rate)"""
    joined = " ".join([c.lower() for c in rc if c])
    header_keywords = ["s.no", "s.no.", "s.no", "particular", "particulars", "qty", "rate", "amt", "amount"]
    matches = sum(1 for k in header_keywords if k in joined)
    return matches >= 2

def row_to_structured(row, mapping):
    """Convert a raw row (list of cells) to structured line item."""
    def safe_get(idx):
        if idx is None: 
            return None
        if idx < len(row):
            return row[idx]
        return None

    name = safe_get(mapping.get("item_name"))
    qty = parse_number(safe_get(mapping.get("item_quantity")))
    rate = parse_number(safe_get(mapping.get("item_rate")))
    amt = parse_number(safe_get(mapping.get("item_amount")))

    # compute amount if missing
    if amt is None and qty is not None and rate is not None:
        try:
            amt = qty * rate
        except:
            pass

    return {
        "item_name": name.strip() if name else None,
        "item_quantity": qty,
        "item_rate": rate,
        "item_amount": amt
    }


# ---------------------------
# Main extraction: region -> items
# ---------------------------
# ---------------------------
# Additional helpers & classifier wiring
# ---------------------------
def extract_last_number(text):
    """Return last numeric-like token parsed to float, or None."""
    if not text:
        return None
    # find tokens that look like numbers
    tokens = re.findall(r"[-\d\.,()]+", text)
    if not tokens:
        return None
    # try tokens from right to left
    for t in reversed(tokens):
        val = parse_number(t)
        if val is not None:
            return val
    return None

def is_header_like_row(rc):
    """Heuristic: a row is header-like if it contains header keywords or mostly non-numeric short tokens."""
    joined = " ".join([c or "" for c in rc]).lower()
    header_kw = ["s.no", "s.no.", "s.no", "particular", "particulars", "qty", "rate", "amt", "amount", "date", "code", "sac", "bill"]
    if any(k in joined for k in header_kw):
        return True
    # if row is very short and contains words like 'total' treat as header/section
    tokens = [t for t in joined.split() if t]
    if len(tokens) <= 2 and any(not is_number_like(t) for t in tokens):
        return True
    return False

def map_columns_to_fields(rows_cells):
    """
    Try to map inferred column indices to logical fields using header row heuristics.
    Returns mapping dict: {"item_name": idx or None, "item_quantity": idx or None, "item_rate": idx or None, "item_amount": idx or None}
    """
    mapping = {"item_name": None, "item_quantity": None, "item_rate": None, "item_amount": None}
    if not rows_cells:
        return mapping
    header_idx = detect_header_row(rows_cells)
    # ensure header_idx in range
    header_idx = header_idx if header_idx < len(rows_cells) else 0
    header_row = rows_cells[header_idx]
    for idx, col in enumerate(header_row):
        if not col:
            continue
        cl = col.lower()
        if any(k in cl for k in ["partic", "item", "description", "particulars", "name"]):
            if mapping["item_name"] is None:
                mapping["item_name"] = idx
        if any(k in cl for k in ["qty", "quantity", "unit"]):
            if mapping["item_quantity"] is None:
                mapping["item_quantity"] = idx
        if any(k in cl for k in ["rate", "price", "unit rate", "rate(rs", "rate (rs"]):
            if mapping["item_rate"] is None:
                mapping["item_rate"] = idx
        if any(k in cl for k in ["amt", "amount", "net amount", "value"]):
            if mapping["item_amount"] is None:
                mapping["item_amount"] = idx
    # fallback heuristics: if amount not found, pick rightmost numeric column later
    return mapping

# Try to import external row classifier (user's module). If not found, use internal simple classifier.
try:
    from src.extraction.row_classifier import is_line_item
except Exception:
    # internal fallback classifier
    def is_line_item(rc, mapping=None, ocr_row_text=None):
        """
        rc: list of cell strings for the row
        mapping: optional mapping dict
        ocr_row_text: raw joined text
        returns (bool, reason)
        """
        joined = (ocr_row_text or " ".join([c or "" for c in rc])).lower()
        # filter totals/footers
        if is_total_row(joined):
            return False, "total-like"
        # header-like
        if is_header_like_row(rc):
            return False, "header-like"
        # if any rightmost numeric-looking token present -> likely item
        # quick numeric detection
        if any(is_number_like(c) for c in rc):
            # but avoid page-level fields
            if any(k in joined for k in ["patient", "bill no", "mobile", "address", "ward", "reg no", "final bill", "hospital"]):
                return False, "page-metadata"
            return True, "has-number"
        # otherwise reject
        return False, "no-numeric"

# ---------------------------
# Updated extract_table_from_region (replace your old one with this)
# ---------------------------
def extract_table_from_region(ocr_words, region, min_conf=50):
    # 1) shrink region a little to avoid borders/headers captured
    region = shrink_region(region, shrink_ratio=0.02)

    # 2) filter words inside region and by confidence
    inside = [w for w in ocr_words if word_in_region(w["box"], region)]
    inside = filter_words_by_conf(inside, min_conf=min_conf)

    if not inside or len(inside) < 4:
        return []

    # 3) cluster into rows
    rows = cluster_words_into_rows(inside)
    rows = [sort_row_by_x(r) for r in rows]

    if not rows:
        return []

    # 4) infer columns and create rows_cells
    columns = infer_columns(rows, x_threshold=60)
    rows_cells = [split_row_into_cells(r, columns) for r in rows]

    # 5) remove empty rows and obvious total/header lines
    filtered_rows = []
    for rc in rows_cells:
        joined = " ".join([c for c in rc if c])
        if not joined.strip():
            continue
        if is_total_row(joined):
            continue
        filtered_rows.append(rc)
    if not filtered_rows:
        return []

    # 6) detect header row index & header row
    header_idx = detect_header_row(filtered_rows)
    if header_idx >= len(filtered_rows):
        header_idx = 0
    header_row = filtered_rows[header_idx]

    # 7) pick amount, qty, rate columns using data rows (exclude header)
    data_rows = filtered_rows[header_idx + 1 :] if header_idx + 1 < len(filtered_rows) else filtered_rows
    amount_idx = pick_amount_column(data_rows) if data_rows else pick_amount_column(filtered_rows)
    qty_idx, rate_idx = pick_qty_rate_columns(data_rows if data_rows else filtered_rows, amount_idx)

    # also build mapping from header heuristics (useful if you want explicit mapping)
    header_mapping = map_columns_to_fields(filtered_rows)

    # 8) iterate data rows (skip header row)
    items = []
    for i, rc in enumerate(filtered_rows):
        if i <= header_idx:
            continue
        # skip rows that look header-like
        if is_header_like_row(rc):
            continue

        # helper to safely get cell by index
        def get_val(idx):
            if idx is None:
                return None
            if idx < len(rc):
                return rc[idx]
            return None

        # attempt to detect rate × qty patterns anywhere in the row
        detected_qty = None
        detected_rate = None
        detected_amount = None

        for j, cell in enumerate(rc):
            if not cell:
                continue
            m = re.search(r"(-?[\d\.,]+\(?\d*\)?)\s*[x×]\s*(-?[\d\.,]+\(?\d*\)?)", cell)
            if m:
                left = parse_number(m.group(1))
                right = parse_number(m.group(2))
                if left is not None and right is not None:
                    # assume left=rate, right=qty (common)
                    detected_rate = left
                    detected_qty = right
                    # try to find an explicit amount in amount column
                    if amount_idx is not None:
                        detected_amount = extract_last_number(get_val(amount_idx) or "")
                    break

        # fallback: read amount/qty/rate from respective columns
        if detected_amount is None and amount_idx is not None:
            detected_amount = extract_last_number(get_val(amount_idx) or "")

        if detected_qty is None and qty_idx is not None:
            detected_qty = extract_last_number(get_val(qty_idx) or "")

        if detected_rate is None and rate_idx is not None:
            detected_rate = extract_last_number(get_val(rate_idx) or "")

        # if nothing in dedicated columns, try to parse near-right cells (rightmost numeric tokens)
        if detected_amount is None:
            detected_amount = extract_last_number(" ".join([get_val(k) or "" for k in range(len(rc)-2, len(rc))]))

        # if still no qty but amount contains "a x b"
        if detected_qty is None and detected_rate is None and amount_idx is not None:
            candidate = get_val(amount_idx)
            if candidate and "x" in candidate:
                parts = re.split(r"[x×]", candidate)
                if len(parts) >= 2:
                    left_num = extract_last_number(parts[0])
                    right_num = extract_last_number(parts[1])
                    if left_num is not None and right_num is not None:
                        detected_rate = left_num
                        detected_qty = right_num
                        detected_amount = extract_last_number(candidate)

        # compute amount if missing but rate & qty present
        if detected_amount is None and (detected_rate is not None and detected_qty is not None):
            try:
                detected_amount = float(detected_rate) * float(detected_qty)
            except Exception:
                detected_amount = None

        # Compose item_name: left-most non-numeric cells excluding qty/rate/amount columns
        name_parts = []
        for j, c in enumerate(rc):
            if j in (amount_idx, qty_idx, rate_idx):
                continue
            if c and not is_number_like(c):
                name_parts.append(clean_cell_text(c))
        item_name = " ".join(name_parts).strip() if name_parts else clean_cell_text(get_val(0) or "")

        # Remove leading serial numbers/dates from item_name
        item_name = re.sub(r"^\d{1,2}[\./-]\d{1,2}[\./-]\d{2,4}\s*", "", item_name)  # drop leading dates
        item_name = re.sub(r"^\d+\.\s*", "", item_name)  # drop "1. " style serials

        # Final numeric values
        qty_val = float(detected_qty) if detected_qty is not None else None
        rate_val = float(detected_rate) if detected_rate is not None else None
        amount_val = float(detected_amount) if detected_amount is not None else None

        # sanity guards
        if qty_val is not None and abs(qty_val) > 1e6:
            qty_val = None
        if rate_val is not None and abs(rate_val) > 1e9:
            rate_val = None
        if amount_val is not None and abs(amount_val) > 1e12:
            amount_val = None

        # final reject: nothing useful
        if (not item_name or item_name.strip() == "") and amount_val is None and rate_val is None and qty_val is None:
            continue

        # Avoid page metadata rows
        lower_name = (item_name or "").lower()
        if any(k in lower_name for k in ["patient", "bill no", "mobile", "address", "ward", "reg no", "final bill", "hospital"]):
            continue

        # Run classifier (external or fallback) to decide if this row is a line item
        joined_row_text = " ".join([c or "" for c in rc])
        try:
            keep, reason = is_line_item(rc, mapping=header_mapping, ocr_row_text=joined_row_text)
        except Exception:
            keep, reason = True, "classifier-failed-default-keep"

        if not keep:
            # skip non-line items
            continue

        items.append({
            "item_name": item_name if item_name else None,
            "item_quantity": float(qty_val) if qty_val is not None else None,
            "item_rate": float(rate_val) if rate_val is not None else None,
            "item_amount": float(amount_val) if amount_val is not None else None
        })

    return items


# ---------------------------
# Page-level extraction
# ---------------------------
def extract_tables_from_page(ocr_json, regions, min_conf=50):
    """
    ocr_json: {"page_no":..., "words":[{"text","box","conf"}...], ...}
    regions: list of {"x1","y1","x2","y2","type":"table"}
    """
    words = ocr_json.get("words", [])
    items = []
    for region in regions:
        # skip tiny regions
        w = region["x2"] - region["x1"]
        h = region["y2"] - region["y1"]
        if w < 50 or h < 20:
            continue
        extracted = extract_table_from_region(words, region, min_conf=min_conf)
        if extracted:
            items.extend(extracted)
    return items

