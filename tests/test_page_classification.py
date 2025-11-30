# tests/test_page_classification.py

import unittest
from src.pipeline.run_full_extraction import classify_page_type


class TestPageClassification(unittest.TestCase):

    def test_bill_detail_page(self):
        """Page containing normal line items should be Bill Detail."""
        items = [
            {"item_name": "CBC", "item_quantity": 1, "item_rate": 240, "item_amount": 240},
            {"item_name": "X-Ray Chest", "item_quantity": 1, "item_rate": 180, "item_amount": 180},
        ]
        ocr_words = [{"text": "CBC"}, {"text": "X Ray"}]

        page_type = classify_page_type(items, ocr_words)
        self.assertEqual(page_type, "Bill Detail")

    def test_final_bill_page(self):
        """If page contains total keywords → Final Bill."""
        items = [
            {"item_name": "Serum Amylase", "item_amount": 240},
            {"item_name": "Widal Test", "item_amount": 300},
        ]
        ocr_words = [
            {"text": "Grand Total"},
            {"text": "Net Amount Payable"},
        ]

        page_type = classify_page_type(items, ocr_words)
        self.assertEqual(page_type, "Final Bill")

    def test_pharmacy_page(self):
        """If mostly medicine-like items → Pharmacy."""
        items = [
            {"item_name": "Pantocid 40 mg tablet", "item_amount": 50},
            {"item_name": "Paracetamol 500mg", "item_amount": 30},
            {"item_name": "Cefixime 200mg capsule", "item_amount": 90},
        ]
        ocr_words = [{"text": "Pantocid"}, {"text": "tablet"}]

        page_type = classify_page_type(items, ocr_words)
        self.assertEqual(page_type, "Pharmacy")

    def test_unknown_page(self):
        """Fallback when no rule is triggered."""
        items = []
        ocr_words = [{"text": "Some hospital info"}, {"text": "Admission Date"}]

        page_type = classify_page_type(items, ocr_words)
        self.assertEqual(page_type, "Unknown")


if __name__ == "__main__":
    unittest.main()
