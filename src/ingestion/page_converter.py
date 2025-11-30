import os
from pdf2image import convert_from_path
from PIL import Image

def convert_document_to_images(doc_id, doc_path, output_dir="/tmp/docs"):
    page_dir = os.path.join(output_dir, doc_id)
    os.makedirs(page_dir, exist_ok=True)

    ext = doc_path.split(".")[-1].lower()
    page_paths = []

    if ext == "pdf":
        pages = convert_from_path(doc_path, dpi=300)
        for i, page in enumerate(pages):
            out_path = os.path.join(page_dir, f"page_{i+1}.png")
            page.save(out_path, "PNG")
            page_paths.append(out_path)

    else:  # image formats
        img = Image.open(doc_path)

        # Multi-page TIFF support
        try:
            for i in range(img.n_frames):
                img.seek(i)
                out_path = os.path.join(page_dir, f"page_{i+1}.png")
                img.save(out_path)
                page_paths.append(out_path)
        except:
            out_path = os.path.join(page_dir, "page_1.png")
            img.save(out_path)
            page_paths.append(out_path)

    return page_paths
