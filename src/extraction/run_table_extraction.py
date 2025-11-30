import os
import json

from src.ocr.ocr_layout import ocr_and_layout
from src.extraction.table_extractor import extract_tables_from_page


def process_folder(preprocessed_folder):
    # Find all PNG images
    images = [f for f in os.listdir(preprocessed_folder) if f.lower().endswith(".png")]
    images = sorted(images)  # ensures page_1, page_2 order

    if not images:
        print("❌ No images found in folder:", preprocessed_folder)
        return

    print(f"\nFound {len(images)} images. Extracting tables...\n")

    all_page_items = []

    for img_name in images:
        img_path = os.path.join(preprocessed_folder, img_name)
        print(f"➡ OCR+Layout on: {img_name}")

        # Run OCR + layout
        ocr_json, table_regions = ocr_and_layout(img_path)

        # Extract tables
        items = extract_tables_from_page(ocr_json, table_regions)

        print(f"   Items found: {len(items)}")

        all_page_items.append({
            "page_no": ocr_json["page_no"],
            "items": items
        })

    return all_page_items


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_table_extraction.py <preprocessed_folder>")
        exit(1)

    folder = sys.argv[1]
    results = process_folder(folder)

    print("\n\n================ FINAL EXTRACTED ITEMS ================\n")
    print(json.dumps(results, indent=2))
