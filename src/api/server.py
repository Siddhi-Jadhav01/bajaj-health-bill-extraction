# src/api/server.py
import uvicorn
import requests
import tempfile
import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline.run_full_extraction import process_invoice

app = FastAPI(title="Bill Extraction API", version="1.0")


class RequestModel(BaseModel):
    document: str   # URL to image (PNG/JPG) or PDF


def download_file(url: str, tmp_dir: str):
    """Downloads a file from URL and saves to tmp_dir."""

    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            raise Exception(f"Failed to download file. Status={response.status_code}")

        # determine extension
        content_type = response.headers.get("Content-Type", "").lower()

        if "pdf" in content_type or url.lower().endswith(".pdf"):
            ext = ".pdf"
        elif "png" in content_type:
            ext = ".png"
        elif "jpeg" in content_type or "jpg" in content_type:
            ext = ".jpg"
        else:
            # default: treat as image
            ext = ".png"

        file_path = os.path.join(tmp_dir, f"doc{ext}")
        with open(file_path, "wb") as f:
            f.write(response.content)

        return file_path

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download error: {str(e)}")


@app.post("/extract-bill-data")
async def extract_bill(data: RequestModel):
    """
    FULL PIPELINE:
    1. Download image/PDF from URL
    2. Convert PDF → images if needed
    3. Preprocess
    4. OCR + table extraction
    5. Dedupe + reconciliation
    6. Return HackRx compliant JSON
    """

    try:
        tmp_dir = tempfile.mkdtemp()

        # 1. download URL
        file_path = download_file(data.document, tmp_dir)

        # 2. run extraction (handles PDF or folder)
        result = process_invoice(tmp_dir)

        # HackRx response format
        response = {
            "is_success": True,
            "token_usage": {
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0
            },
            "data": {
                "pagewise_line_items": result.get("pagewise_line_items", []),
                "total_item_count": result.get("total_item_count", 0),
                "totals": result.get("totals", []),
                "reconciliation": result.get("reconciliation", {})
            }
        }

        return response

    except Exception as e:
        return {
            "is_success": False,
            "message": str(e)
        }


if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
