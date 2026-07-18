from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from PIL import Image
from app.inference import run_style_transfer_image, run_style_transfer_text
import uvicorn
import os

app = FastAPI(title="Hybrid Neural Style Transfer API")

# Mount static files (CSS, JS)
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_path, 'index.html'))

@app.post("/stylize/image")
async def stylize_image(content_file: UploadFile = File(...), style_file: UploadFile = File(...)):
    content_pil = Image.open(BytesIO(await content_file.read()))
    style_pil = Image.open(BytesIO(await style_file.read()))

    result_img = run_style_transfer_image(content_pil, style_pil)

    img_io = BytesIO()
    result_img.save(img_io, 'JPEG')
    img_io.seek(0)
    return StreamingResponse(img_io, media_type="image/jpeg")

@app.post("/stylize/text")
async def stylize_text(content_file: UploadFile = File(...), prompt: str = Form(...)):
    content_pil = Image.open(BytesIO(await content_file.read()))

    result_img = run_style_transfer_text(content_pil, prompt)

    img_io = BytesIO()
    result_img.save(img_io, 'JPEG')
    img_io.seek(0)
    return StreamingResponse(img_io, media_type="image/jpeg")

@app.get("/health")
async def health_check():
    return {"status": "ready", "engine": "tensorflow_weights"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
