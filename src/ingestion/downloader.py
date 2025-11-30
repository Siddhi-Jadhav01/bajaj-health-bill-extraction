import os
import uuid
import requests

def download_document(url: str, base_dir=None):
    # Use cross-platform safe location
    if base_dir is None:
        base_dir = os.path.join(os.getcwd(), "tmp", "docs")
    
    os.makedirs(base_dir, exist_ok=True)

    # Generate unique ID for this document
    doc_id = str(uuid.uuid4())

    # Extract file extension
    ext = url.split("?")[0].split(".")[-1].lower()
    if ext not in ["pdf", "png", "jpg", "jpeg", "tiff"]:
        raise ValueError(f"Unsupported file type: {ext}")

    doc_path = os.path.join(base_dir, f"{doc_id}.{ext}")

    # Download file
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError("Could not download file. Invalid URL or access denied.")

    # Save file
    with open(doc_path, "wb") as f:
        f.write(resp.content)

    return doc_id, doc_path


# ------------------------------
#  Manual test runner
# ------------------------------
if __name__ == "__main__":
    test_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    doc_id, doc_path = download_document(test_url)

    print("Download successful!")
    print("Document ID:", doc_id)
    print("Saved file:", doc_path)

if __name__ == "__main__":
    try:
        download_document("https://example.com/file.txt")
    except Exception as e:
        print("Validation Test:", e)
