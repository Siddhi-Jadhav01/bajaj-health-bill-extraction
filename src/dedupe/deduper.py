# src/dedupe/deduper.py
import hashlib
import re
from typing import List, Dict, Tuple
try:
    from Levenshtein import ratio as levenshtein_ratio
    LEVENSHTEIN_AVAILABLE = True
except Exception:
    from difflib import SequenceMatcher
    LEVENSHTEIN_AVAILABLE = False

# -------------------------
# Helpers
# -------------------------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    # remove multiple spaces and non-printable characters
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\x20-\x7E]", "", s)
    return s

def normalize_number(n) -> float:
    if n is None:
        return None
    try:
        return round(float(n), 2)
    except:
        return None

def row_fingerprint(row: Dict) -> str:
    # canonical string: itemname|qty|rate|amount
    parts = [
        normalize_text(row.get("item_name")),
        str(normalize_number(row.get("item_quantity"))),
        str(normalize_number(row.get("item_rate"))),
        str(normalize_number(row.get("item_amount"))),
    ]
    canonical = "|".join(parts)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()

def iou(boxA, boxB) -> float:
    # boxes are [x1,y1,x2,y2]
    if not boxA or not boxB:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    return (interArea / union) if union > 0 else 0.0

def text_similarity(a: str, b: str) -> float:
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if LEVENSHTEIN_AVAILABLE:
        try:
            return levenshtein_ratio(a_n, b_n)  # returns 0..1
        except:
            pass
    # fallback
    return SequenceMatcher(None, a_n, b_n).ratio()

def is_header_footer(text: str) -> bool:
    if not text:
        return False
    t = normalize_text(text)
    # heuristics: common header/footer words
    header_keywords = ["invoice", "tax", "page", "total", "discount", "phone", "gst", "invoice no", "bill no", "bill"]
    # if line is very short and is like 'page 1 of 2' or contains 'page'
    if re.search(r"\bpage\b", t) or re.search(r"\bof\b", t) and len(t) < 30:
        return True
    # if text equals common header words (clinic/hospital names vary, so don't be too strict)
    for k in header_keywords:
        if k in t:
            return True
    return False

# -------------------------
# Main dedupe function
# -------------------------
def dedupe_rows(rows: List[Dict],
                page_heights: Dict = None,
                iou_thresh: float = 0.7,
                fuzzy_text_thresh: float = 0.9,
                fuzzy_amount_tolerance: float = 0.5,
                header_footer_pct: float = 0.06
                ) -> Tuple[List[Dict], List[Dict]]:
    """
    rows: list of dicts with keys:
      - item_name (str)
      - item_amount (float)
      - item_rate (float)
      - item_quantity (float)
      - box (optional) -> [x1,y1,x2,y2]
      - page_no (optional) -> int or str

    page_heights: optional dict mapping page_no -> page_height_in_pixels
                  used for header/footer detection.

    Returns: (unique_rows, audit_log)
    audit_log: list of dicts with dropped row info and reason
    """
    if page_heights is None:
        page_heights = {}

    kept = []
    fingerprint_seen = {}  # fingerprint -> index in kept
    audit = []

    # Stage 0: prepare normalized copy and ensure each row has page_no and box
    rows_prepped = []
    for idx, r in enumerate(rows):
        r2 = dict(r)  # shallow copy
        r2["_orig_index"] = idx
        r2["_norm_name"] = normalize_text(r.get("item_name"))
        r2["_norm_qty"] = normalize_number(r.get("item_quantity"))
        r2["_norm_rate"] = normalize_number(r.get("item_rate"))
        r2["_norm_amount"] = normalize_number(r.get("item_amount"))
        # ensure box exists (if not, set to None)
        r2.setdefault("box", None)
        # ensure page_no exists
        r2.setdefault("page_no", r.get("page_no", 1))
        rows_prepped.append(r2)

    # Stage 1: Exact text hash (per page)
    for r in rows_prepped:
        page = str(r["page_no"])
        fingerprint = row_fingerprint(r)
        key = (page, fingerprint)
        if key in fingerprint_seen:
            kept_idx = fingerprint_seen[key]
            audit.append({
                "reason": "exact_hash_duplicate",
                "kept_index": kept_idx,
                "dropped_index": r["_orig_index"],
                "dropped_row": r
            })
            continue
        # else keep for now
        fingerprint_seen[key] = len(kept)
        kept.append(r)

    # Stage 2: Bounding-box overlap + text similarity
    final = []
    for i, r in enumerate(kept):
        dropped_flag = False
        for j, k in enumerate(final):
            # only compare rows on same page
            if str(r["page_no"]) != str(k["page_no"]):
                continue
            # require both boxes to exist
            if r.get("box") and k.get("box"):
                score_iou = iou(r["box"], k["box"])
                sim = text_similarity(r["_norm_name"], k["_norm_name"])
                # if boxes overlap heavily and text similar -> drop r
                if score_iou >= iou_thresh and sim >= 0.85:
                    audit.append({
                        "reason": "bbox_overlap_and_text_similar",
                        "kept_index": final.index(k),
                        "dropped_index": r["_orig_index"],
                        "dropped_row": r,
                        "iou": score_iou,
                        "text_sim": sim
                    })
                    dropped_flag = True
                    break
        if not dropped_flag:
            final.append(r)

    # Stage 3: Fuzzy text match + amount equal (or within tolerance)
    unique = []
    for r in final:
        is_dup = False
        for u in unique:
            if str(r["page_no"]) != str(u["page_no"]):
                continue
            sim = text_similarity(r["_norm_name"], u["_norm_name"])
            amt_r = r["_norm_amount"]
            amt_u = u["_norm_amount"]
            # amount equal within tolerance or both None
            amt_close = False
            if amt_r is None and amt_u is None:
                amt_close = True
            elif amt_r is None or amt_u is None:
                amt_close = False
            else:
                amt_close = abs(amt_r - amt_u) <= fuzzy_amount_tolerance
            if sim >= fuzzy_text_thresh and amt_close:
                audit.append({
                    "reason": "fuzzy_text_and_amount_match",
                    "kept_index": unique.index(u),
                    "dropped_index": r["_orig_index"],
                    "dropped_row": r,
                    "text_sim": sim,
                    "amount_r": amt_r,
                    "amount_u": amt_u
                })
                is_dup = True
                break
        if not is_dup:
            unique.append(r)

    # Stage 4: Header/Footer removal
    final_kept = []
    for r in unique:
        box = r.get("box")
        page = str(r["page_no"])
        page_h = page_heights.get(page, None)
        removed = False
        # try header/footer heuristics only if box and page height available
        if box and page_h:
            top_pct = box[1] / float(page_h)
            bottom_pct = (page_h - box[3]) / float(page_h)
            # if within top or bottom N% AND text looks like header/footer => drop
            if (top_pct <= header_footer_pct or bottom_pct <= header_footer_pct) and is_header_footer(r["_norm_name"]):
                audit.append({
                    "reason": "header_footer_removed",
                    "dropped_index": r["_orig_index"],
                    "dropped_row": r,
                    "top_pct": top_pct,
                    "bottom_pct": bottom_pct
                })
                removed = True
        if not removed:
            final_kept.append(r)

    # Prepare return: strip internal keys
    unique_rows = []
    for r in final_kept:
        # remove internal keys
        out = dict(r)
        for k in ["_orig_index", "_norm_name", "_norm_qty", "_norm_rate", "_norm_amount"]:
            out.pop(k, None)
        unique_rows.append(out)

    return unique_rows, audit


# -------------------------
# Quick unit test when run as script
# -------------------------
if __name__ == "__main__":
    # small sanity test
    sample = [
        {"item_name": "Paracetamol 500mg", "item_amount": 30.0, "item_rate": 15.0, "item_quantity": 2, "box":[10,10,300,40], "page_no":1},
        {"item_name": "Paracetamol 500mg", "item_amount": 30.0, "item_rate": 15.0, "item_quantity": 2, "box":[12,12,302,42], "page_no":1},
        {"item_name": "Cephalexin 250mg", "item_amount": 80.0, "item_rate": 40.0, "item_quantity": 2, "box":[10,60,300,90], "page_no":1}
    ]
    uniq, audit = dedupe_rows(sample, page_heights={"1":200})
    print("Kept:", uniq)
    print("Audit:", audit)
