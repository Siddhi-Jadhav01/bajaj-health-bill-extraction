import cv2
import numpy as np
import os

def preprocess_page(image_path):
    # Load image
    img = cv2.imread(image_path)

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Adaptive threshold (binarization)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # 3. Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, h=10)

    # 4. Deskew image
    coords = np.column_stack(np.where(denoised < 255))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = denoised.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    deskewed = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC)

    # 5. Resize (boost OCR quality)
    if w < 1000:
        scale = 1000 / w
        deskewed = cv2.resize(deskewed, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Save cleaned file
    cleaned_path = image_path.replace(".png", "_cleaned.png")
    cv2.imwrite(cleaned_path, deskewed)

    return cleaned_path
