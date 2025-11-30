from ingestion.downloader import download_document
from ingestion.page_converter import convert_document_to_images
from preprocessing.image_cleaner import preprocess_page

def ingest_and_preprocess(url):
    # Download doc
    doc_id, doc_path = download_document(url)

    # Convert to images
    page_images = convert_document_to_images(doc_id, doc_path)

    # Preprocess each page
    cleaned_pages = []
    for img in page_images:
        cleaned = preprocess_page(img)
        cleaned_pages.append(cleaned)

    return doc_id, cleaned_pages
