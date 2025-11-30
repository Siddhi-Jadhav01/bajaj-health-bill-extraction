import cv2
import os
import sys

def preprocess_page(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        return None

    # Resize if too small
    if img.shape[1] < 1000:
        scale = 1000 / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 10
    )

    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, h=30)

    # Output folder
    out_dir = os.path.join(os.path.dirname(image_path), "preprocessed")
    os.makedirs(out_dir, exist_ok=True)

    # Save cleaned file
    base = os.path.basename(image_path)
    out_path = os.path.join(out_dir, base.replace(".png", "_clean.png").replace(".jpg", "_clean.png").replace(".jpeg", "_clean.png"))
    cv2.imwrite(out_path, denoised)

    return out_path


def preprocess_folder(folder_path):
    # Allowed image formats
    exts = [".png", ".jpg", ".jpeg"]

    files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in exts
    ]

    if not files:
        print("❌ No images found in folder:", folder_path)
        return

    print(f"Found {len(files)} images. Preprocessing...")

    cleaned_paths = []

    for f in files:
        img_path = os.path.join(folder_path, f)
        cleaned = preprocess_page(img_path)
        if cleaned:
            cleaned_paths.append(cleaned)

    print("\n✅ Preprocessing completed!")
    print("Cleaned images:")
    for c in cleaned_paths:
        print(" -", c)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <folder_path>")
        sys.exit(1)

    folder = sys.argv[1]
    preprocess_folder(folder)
