import os
import sys
from pdf2image import convert_from_path

def pdf_to_images(pdf_path, base_output_dir=None):
    # If no output folder passed → create next to the PDF
    if base_output_dir is None:
        # Example: for F:/Downloads/TRAINING_SAMPLES/sample.pdf
        # output → F:/Downloads/TRAINING_SAMPLES/pdf-to-images/sample/
        pdf_dir = os.path.dirname(pdf_path)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        base_output_dir = os.path.join(pdf_dir, "pdf-to-images", pdf_name)

    os.makedirs(base_output_dir, exist_ok=True)

    print(f"[INFO] Output folder: {base_output_dir}")

    # Convert PDF → images (all pages)
    pages = convert_from_path(pdf_path, dpi=300)

    image_paths = []
    for i, p in enumerate(pages, start=1):
        out_path = os.path.join(base_output_dir, f"page_{i}.png")
        p.save(out_path, "PNG")
        image_paths.append(out_path)

    return image_paths


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("USAGE: python pdf_to_images.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    pages = pdf_to_images(pdf_path)
    print("Generated Images:", pages)
