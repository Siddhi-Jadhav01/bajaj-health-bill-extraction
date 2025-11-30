import os
import json
import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import Tuple, List, Dict

# Optional layoutparser import
try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
except Exception:
    LAYOUTPARSER_AVAILABLE = False


# -------------------------------------------------------------
# OCR FUNCTIONS
# -------------------------------------------------------------
def extract_page_no_from_path(image_path: str) -> int:
    """Extract page number from filename like page_1.png"""
    base = os.path.basename(image_path).lower()
    for part in base.replace(".png", "").replace(".jpg", "").split("_"):
        if part.isdigit():
            return int(part)
    return 1


def merge_boxes(box_a, box_b):
    """Merge two bounding boxes."""
    if box_a[0] is None:
        return box_b[:]
    x1 = min(box_a[0], box_b[0])
    y1 = min(box_a[1], box_b[1])
    x2 = max(box_a[2], box_b[2])
    y2 = max(box_a[3], box_b[3])
    return [int(x1), int(y1), int(x2), int(y2)]


def cluster_words_to_blocks(words: List[Dict], y_tol: int = 10) -> List[Dict]:
    """Group OCR words into blocks by line."""
    if not words:
        return []

    words_sorted = sorted(words, key=lambda w: (w["box"][1], w["box"][0]))
    blocks = []
    current_block = {"text": "", "box": [None, None, None, None], "words": []}

    current_y = (words_sorted[0]["box"][1] + words_sorted[0]["box"][3]) // 2

    for w in words_sorted:
        mid_y = (w["box"][1] + w["box"][3]) // 2
        if abs(mid_y - current_y) <= y_tol:
            current_block["words"].append(w)
            current_block["text"] += w["text"] + " "
            current_block["box"] = merge_boxes(current_block["box"], w["box"])
        else:
            if current_block["words"]:
                current_block["text"] = current_block["text"].strip()
                blocks.append(current_block)

            current_block = {"text": w["text"] + " ", "box": w["box"][:], "words": [w]}
            current_y = mid_y

    # Final block
    if current_block["words"]:
        current_block["text"] = current_block["text"].strip()
        blocks.append(current_block)

    return blocks


def ocr_with_pytesseract(image_path: str) -> Dict:
    """Run OCR with bounding boxes."""
    img = Image.open(image_path).convert("RGB")
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    words = []
    n = len(data['text'])
    for i in range(n):
        text = data["text"][i].strip()
        if text == "":
            continue

        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        conf = float(data["conf"][i]) if data["conf"][i] != "-1" else None

        words.append({
            "text": text,
            "box": [x, y, x + w, y + h],
            "conf": conf
        })

    raw_text = pytesseract.image_to_string(img)
    blocks = cluster_words_to_blocks(words)

    return {
        "page_no": extract_page_no_from_path(image_path),
        "words": words,
        "raw_text": raw_text,
        "blocks": blocks
    }


# -------------------------------------------------------------
# TABLE DETECTION (OpenCV fallback)
# -------------------------------------------------------------
def detect_tables_with_opencv(image_path: str, debug=False):
    """Detect possible table regions using horizontal + vertical line detection."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # horizontal lines
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    hori = cv2.morphologyEx(th, cv2.MORPH_OPEN, hori_kernel, iterations=2)

    # vertical lines
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vert = cv2.morphologyEx(th, cv2.MORPH_OPEN, vert_kernel, iterations=2)

    table_mask = cv2.add(hori, vert)

    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = gray.shape
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # minimum size filter
        if w < 0.2 * w_img or h < 0.03 * h_img:
            continue

        boxes.append({"x1": x, "y1": y, "x2": x + w, "y2": y + h})

        if debug:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if debug:
        cv2.imwrite(image_path.replace(".png", "_tables_debug.png"), img)

    return boxes


# -------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------
def ocr_and_layout(image_path: str, debug=False):
    """Run OCR + table detection on a single image."""
    ocr_json = ocr_with_pytesseract(image_path)
    table_regions = detect_tables_with_opencv(image_path, debug=debug)
    return ocr_json, table_regions


# -------------------------------------------------------------
# RUN ON A FOLDER (THIS IS WHAT YOU WANT)
# -------------------------------------------------------------
def process_folder(folder_path: str):
    """Runs OCR + layout detection on ALL images in the folder."""

    if not os.path.isdir(folder_path):
        print("❌ Not a valid folder:", folder_path)
        return

    # Read images
    images = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    images = sorted(images)  # ensures page_1, page_2 order

    if not images:
        print("❌ No images found in folder:", folder_path)
        return

    print(f"Found {len(images)} images. Running OCR + layout...\n")

    results = []

    for img_path in images:
        print(f"➡ Processing: {os.path.basename(img_path)}")

        ocr_json, regions = ocr_and_layout(img_path)

        # Save OCR JSON
        out_json = img_path.replace(".png", ".ocr.json").replace(".jpg", ".ocr.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(ocr_json, f, indent=2, ensure_ascii=False)

        print(f"   OCR words: {len(ocr_json['words'])}")
        print(f"   Table regions: {regions}\n")

        results.append({
            "image": img_path,
            "ocr_json": ocr_json,
            "table_regions": regions
        })

    print("\n✅ OCR + Layout processing complete!")
    return results


# -------------------------------------------------------------
# CLI ENTRY POINT
# -------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/ocr/ocr_layout.py <folder_path>")
        sys.exit(1)

    folder = sys.argv[1]
    process_folder(folder)
