import io
import logging
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

from app.inference import load_models, run_style_transfer_image, run_style_transfer_text
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hybrid NST API", version="1.0.0")

@app.on_event("startup")
async def startup():
    logger.info("Loading TFLite models...")
    load_models()
    logger.info("Models loaded.")

# Serve frontend static files
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

# Health check
@app.get("/health")
async def health():
    return {"status": "ok"}

# Image‑guided endpoint
@app.post("/style-transfer-image")
async def style_transfer_image(
    content_image: UploadFile = File(...),
    style_image: UploadFile = File(...),
):
    if not content_image.content_type.startswith("image/") or not style_image.content_type.startswith("image/"):
        raise HTTPException(400, "Both files must be images.")

    try:
        content_pil = Image.open(io.BytesIO(await content_image.read()))
        style_pil   = Image.open(io.BytesIO(await style_image.read()))
        result = run_style_transfer_image(content_pil, style_pil)

        buf = io.BytesIO()
        result.save(buf, format="JPEG", quality=90)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Image‑guided inference error: {e}")
        raise HTTPException(500, f"Inference failed: {str(e)}")

# Text‑guided endpoint 
@app.post("/style-transfer-text")
async def style_transfer_text(
    content_image: UploadFile = File(...),
    style_prompt: str = Form(...),
):
    if not content_image.content_type.startswith("image/"):
        raise HTTPException(400, "Content file must be an image.")

    try:
        content_pil = Image.open(io.BytesIO(await content_image.read()))
        result = run_style_transfer_text(content_pil, style_prompt)

        buf = io.BytesIO()
        result.save(buf, format="JPEG", quality=90)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Text‑guided inference error: {e}")
        raise HTTPException(500, f"Inference failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860)