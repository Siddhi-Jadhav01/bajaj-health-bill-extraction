# src/pipeline/run_full_extraction.py
import os
import json
import re
from pathlib import Path

from src.ocr.ocr_layout import ocr_and_layout
from src.extraction.table_extractor import extract_table_from_region
from src.dedupe.deduper import dedupe_rows

# ---------------------------------------------
# REGION CLEANUP HELPERS
# ---------------------------------------------
def shrink_region(region, shrink=20):
    return {
        "x1": region["x1"] + shrink,
        "y1": region["y1"] + shrink,
        "x2": region["x2"] - shrink,
        "y2": region["y2"] - shrink,
        "type": region.get("type", "table")
    }


def is_valid_region(region):
    width = region["x2"] - region["x1"]
    height = region["y2"] - region["y1"]
    return width > 200 and height > 80


# ---------------------------------------------
# ITEM FILTERING
# ---------------------------------------------
def filter_items(items):
    clean = []
    for it in items:
        name = it.get("item_name")

        if not name and it.get("item_amount") is None:
            continue

        useless = ["address", "patient", "mobile", "room", "ward", "bill no", "husband"]
        if name and any(u in name.lower() for u in useless):
            continue

        clean.append(it)

    return clean


# ---------------------------------------------
# TOTAL / SUBTOTAL EXTRACTION
# ---------------------------------------------
TOTAL_KEYWORDS = [
    "total", "sub total", "subtotal", "grand total", "net amount",
    "amount payable", "net payable", "final amount", "final total"
]

NUM_RE = re.compile(r"[-+]?\(?\d[\d,]*\.?\d*\)?")


def parse_number(text):
    if not text:
        return None
    m = NUM_RE.findall(text.replace(" ", ""))
    if not m:
        return None
    tok = m[-1]
    tok = tok.replace("(", "-").replace(")", "")
    tok = tok.replace(",", "")
    try:
        return float(tok)
    except:
        return None


def extract_totals_from_page(ocr_json):
    results = []
    for w in ocr_json.get("words", []):
        t = w["text"].lower()
        if any(k in t for k in TOTAL_KEYWORDS):
            amount = parse_number(t)
            if amount is not None:
                results.append({
                    "label": t,
                    "amount": amount
                })
    return results


# ---------------------------------------------
# PHASE 6 — RECONCILIATION
# ---------------------------------------------
def reconcile_totals(unique_rows, total_rows, tolerance=1.0):

    computed_total = 0.0
    for r in unique_rows:
        amt = r.get("item_amount")
        if isinstance(amt, (int, float)):
            computed_total += amt

    invoice_total = None
    if total_rows:
        invoice_total = max(t["amount"] for t in total_rows)

    if invoice_total is None:
        return {
            "computed_total": computed_total,
            "invoice_total": None,
            "difference": None,
            "suggestions": ["No total row detected in OCR"]
        }

    diff = invoice_total - computed_total

    suggestions = []
    if abs(diff) > tolerance:
        suggestions.append(f"Mismatch detected (diff={diff}). Possible missing rows.")
        suggestions.append("Review rows marked as non-line items.")
        suggestions.append("Check subtotal rows for page boundaries.")

    return {
        "computed_total": computed_total,
        "invoice_total": invoice_total,
        "difference": diff,
        "suggestions": suggestions
    }


# ---------------------------------------------
# PHASE 7 — PAGE CLASSIFICATION
# ---------------------------------------------
MEDICINE_KEYWORDS = ["mg", "tablet", "capsule", "syrup", "inj", "injection"]
def classify_page(items, ocr_words):
    """
    Returns: "Final Bill", "Pharmacy", "Bill Detail", or "Unknown"
    """

    # Normalize OCR input (tests may pass list instead of dict)
    if isinstance(ocr_words, dict):
        word_list = ocr_words.get("words", [])
    else:
        word_list = ocr_words

    # 1) FINAL BILL DETECTION
    text_blob = " ".join([w["text"].lower() for w in word_list])
    if any(k in text_blob for k in TOTAL_KEYWORDS):
        return "Final Bill"

    # 2) PHARMACY DETECTION
    med_hits = 0
    for it in items:
        name = (it.get("item_name") or "").lower()
        if any(k in name for k in MEDICINE_KEYWORDS):
            med_hits += 1

    if med_hits >= 2:
        return "Pharmacy"

    # 3) UNKNOWN PAGE → only when EMPTY page
    if not items:
        return "Unknown"

    # 4) DEFAULT
    return "Bill Detail"


# Backward compatibility
classify_page_type = classify_page

# ---------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------
def process_invoice(preprocessed_folder):
    folder = Path(preprocessed_folder)

    images = sorted(folder.glob("*.png"))
    if not images:
        raise ValueError("No preprocessed images found:", preprocessed_folder)

    print(f"Found {len(images)} pages. Extracting...\n")

    final_output = {
        "pagewise_line_items": [],
        "total_item_count": 0,
        "totals": [],
        "reconciliation": {}
    }

    all_unique_items = []
    page_heights = {}

    page_no = 1

    for img_path in images:

        print(f"➡ Processing page {page_no}: {img_path.name}")

        # OCR + LAYOUT
        ocr_json, regions = ocr_and_layout(str(img_path))
        print(f"   OCR words: {len(ocr_json['words'])}")
        print(f"   Table regions: {regions}")
        page_heights[str(page_no)] = ocr_json.get("height")

        # shrink + filter
        regions = [shrink_region(r) for r in regions if is_valid_region(r)]

        # extract rows
        extracted = []
        for r in regions:
            items = extract_table_from_region(ocr_json["words"], r)
            for it in items:
                it["page_no"] = page_no
                extracted.append(it)

        extracted = filter_items(extracted)

        # dedupe
        unique_rows, audit = dedupe_rows(extracted, page_heights=page_heights)
        print(f"   → After dedupe: kept {len(unique_rows)}, removed {len(audit)} duplicates")

        # ---- PAGE CLASSIFICATION (NEW) ----
        page_type = classify_page(unique_rows, ocr_json)

        final_output["pagewise_line_items"].append({
            "page_no": page_no,
            "page_type": page_type,
            "bill_items": unique_rows
        })

        all_unique_items.extend(unique_rows)

        # extract totals
        total_vals = extract_totals_from_page(ocr_json)
        if total_vals:
            final_output["totals"].extend(total_vals)

        page_no += 1

    final_output["total_item_count"] = len(all_unique_items)

    # reconciliation
    final_output["reconciliation"] = reconcile_totals(all_unique_items, final_output["totals"])

    return final_output


# ---------------------------------------------
# CLI
# ---------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.run_full_extraction <preprocessed_folder>")
        exit()

    folder = sys.argv[1]
    result = process_invoice(folder)

    print("\n\n✅ Extraction completed!")
    print(json.dumps(result, indent=2))
