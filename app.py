"""FastAPI web service providing an invoice extraction endpoint and web viewer."""

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from invoice_extractor import cache
from invoice_extractor.pipeline import process
from invoice_extractor.schema import Result

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Invoice Extractor API", version="1.0.0")

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", summary="Serve UI")
async def index() -> FileResponse:
    """Serve the single-page application viewer."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Index file not found")
    return FileResponse(index_path)


@app.post("/extract", response_model=Result, summary="Extract invoice data")
async def extract(
    file: Annotated[UploadFile, File(description="Uploaded invoice document or image")],
    no_cache: bool = False,
    ocr_check: bool = False,
) -> Result:
    """Extract structured invoice data from an uploaded file."""
    data = await file.read()
    if no_cache:
        cache.disable()
    try:
        return process(data, ocr_check=ocr_check)
    finally:
        if no_cache:
            cache.enable()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
