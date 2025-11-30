from src.extraction.row_classifier import is_line_item

tests = [
    ["2D echocardiography", "1.00", "1180.00"],
    ["Total of BED CHARGES", "", "6000.00"],
    ["Consultation for Inpatients", "2.00", "700.00"],
    ["S.No.", "Date", "Particulars", "Qty", "Rate", "Amt"],
]

for t in tests:
    keep, reason = is_line_item(t)
    print(t, keep, reason)
