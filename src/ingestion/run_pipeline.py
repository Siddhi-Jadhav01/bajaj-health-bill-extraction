from downloader import download_document
from pdf_to_images import pdf_to_images
from preprocess import preprocess_page
import os

def process_document(url):
    print("Downloading...")
    doc_id, doc_path = download_document(url)

    print("Downloaded:", doc_path)

    # Determine extension
    ext = doc_path.split(".")[-1].lower()

    print("Converting PDF to images..." if ext == "pdf" else "Image detected...")

    # Step 2 — Convert or use as single page
    if ext == "pdf":
        pages = pdf_to_images(doc_path)
    else:
        pages = [doc_path]  # single image

    print("Pages:", pages)

    print("Preprocessing pages...")
    cleaned_pages = [preprocess_page(p) for p in pages]

    print("Cleaned pages:", cleaned_pages)

    return {
        "uuid": doc_id,
        "pages": cleaned_pages
    }


if __name__ == "__main__":
    test_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    result = process_document(test_url)
    print("\nFinal Result:")
    print(result)
