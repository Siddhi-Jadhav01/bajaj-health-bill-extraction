# tests/test_reconciliation.py

from src.pipeline.run_full_extraction import reconcile_totals

def test_case_basic():
    print("\n--- BASIC RECONCILIATION TEST ---")

    unique_rows = [
        {"item_name": "CBC", "item_amount": 240.0},
        {"item_name": "LFT", "item_amount": 400.0},
        {"item_name": "X-Ray", "item_amount": 180.0},
    ]

    non_line_rows = [
        {"text": "Sub Total", "amount": 820.0},
        {"text": "Total Amount Payable", "amount": 820.0}
    ]

    result = reconcile_totals(unique_rows, non_line_rows)

    print("Computed Total:", result["computed_total"])
    print("Invoice Total :", result["invoice_total"])
    print("Difference    :", result["difference"])
    print("Suggestions   :", result["suggestions"])


def test_case_mismatch():
    print("\n--- MISMATCH DETECTION TEST ---")

    unique_rows = [
        {"item_name": "CBC", "item_amount": 240.0},
        {"item_name": "LFT", "item_amount": 400.0},
        {"item_name": "X-Ray", "item_amount": 180.0},
    ]

    non_line_rows = [
        {"text": "Grand Total", "amount": 900.0}   # mismatch (820 vs 900)
    ]

    result = reconcile_totals(unique_rows, non_line_rows)

    print("Computed Total:", result["computed_total"])
    print("Invoice Total :", result["invoice_total"])
    print("Difference    :", result["difference"])
    print("Suggestions   :", result["suggestions"])


if __name__ == "__main__":
    test_case_basic()
    test_case_mismatch()
