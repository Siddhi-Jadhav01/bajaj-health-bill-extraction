# tests/test_dedupe.py

import json
from src.dedupe.deduper import dedupe_rows

def test_basic_duplicate():
    rows = [
        {"item_name": "Paracetamol 500mg", "item_quantity": 2, "item_rate": 15.0, "item_amount": 30.0, "box": [10,10,200,40], "page_no":1},
        {"item_name": "Paracetamol 500mg", "item_quantity": 2, "item_rate": 15.0, "item_amount": 30.0, "box": [12,12,202,42], "page_no":1},  # duplicate
    ]

    unique, audit = dedupe_rows(rows, page_heights={"1": 200})
    print("\n--- BASIC DUPLICATE TEST ---")
    print("Unique:", json.dumps(unique, indent=2))
    print("Audit:", json.dumps(audit, indent=2))


def test_bbox_overlap():
    rows = [
        {"item_name": "X-ray Chest", "item_quantity": 1, "item_rate": 180.0, "item_amount": 180.0, "box": [100,100,300,150], "page_no":1},
        {"item_name": "X-ray Chest", "item_quantity": 1, "item_rate": 180.0, "item_amount": 180.0, "box": [105,105,295,145], "page_no":1},
    ]

    unique, audit = dedupe_rows(rows, page_heights={"1": 1000})
    print("\n--- BBOX OVERLAP TEST ---")
    print("Unique:", json.dumps(unique, indent=2))
    print("Audit:", json.dumps(audit, indent=2))


def test_fuzzy_text_match():
    rows = [
        {"item_name": "Paracetamol 500 mg", "item_quantity": 1, "item_rate": 10.0, "item_amount": 10.0, "box": [10,60,200,90], "page_no":1},
        {"item_name": "Paracetmol 500mg",  "item_quantity": 1, "item_rate": 10.0, "item_amount": 10.0, "box": [15,65,205,95], "page_no":1},  # slightly misspelled
    ]

    unique, audit = dedupe_rows(rows, page_heights={"1": 1200})
    print("\n--- FUZZY TEXT MATCH TEST ---")
    print("Unique:", json.dumps(unique, indent=2))
    print("Audit:", json.dumps(audit, indent=2))


def test_header_footer_removal():
    rows = [
        {"item_name": "Invoice No: 12345", "item_amount": None, "item_rate": None, "item_quantity": None, "box": [10, 5, 300, 25], "page_no": 1},
        {"item_name": "Paracetamol 500mg", "item_amount": 30, "item_rate": 15, "item_quantity": 2, "box": [10, 200, 300, 240], "page_no": 1},
    ]
    # page height = 1000 → box at y1=5 is top 0.5% → header
    unique, audit = dedupe_rows(rows, page_heights={"1": 1000}, header_footer_pct=0.02)

    print("\n--- HEADER/FOOTER REMOVAL TEST ---")
    print("Unique:", json.dumps(unique, indent=2))
    print("Audit:", json.dumps(audit, indent=2))


if __name__ == "__main__":
    test_basic_duplicate()
    test_bbox_overlap()
    test_fuzzy_text_match()
    test_header_footer_removal()
