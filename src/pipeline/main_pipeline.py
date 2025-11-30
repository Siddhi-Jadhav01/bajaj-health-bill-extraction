from src.ingestion.pipeline_phase1 import ingest_and_preprocess
from src.ocr.ocr_layout import ocr_and_layout
from src.extraction.table_extractor import extract_tables_from_page
from src.dedupe.deduper import dedupe_rows
from PIL import Image

def run_pipeline(url):
    doc_id, pages = ingest_and_preprocess(url)

    all_rows = []
    page_heights = {}

    for i, page in enumerate(pages, start=1):
        # get page height for header/footer filtering
        page_heights[str(i)] = Image.open(page).size[1]

        # OCR + tables
        ocr_json, table_regions = ocr_and_layout(page)

        # Table row extraction
        items = extract_tables_from_page(ocr_json, table_regions)

        # Make sure each row has page_no and box (if not, add)
        for row in items:
            row["page_no"] = i
            # fill row["box"] if you extracted bounding box from Phase 3

        all_rows.extend(items)

    # Phase 5 — DEDUPLICATION
    unique_rows, audit_log = dedupe_rows(all_rows, page_heights)

    print("\nUNIQUE ROWS:")
    for row in unique_rows:
        print(row)

    print("\nAUDIT LOG (duplicates removed):")
    for entry in audit_log:
        print(entry["reason"], " → ", entry["dropped_row"].get("item_name"))

    return unique_rows, audit_log
