import sys
import os
# Ensure the root directory is in sys.path so 'custom_classes' and 'app' are discoverable
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
from app.inference import run_style_transfer_image, run_style_transfer_text
import uvicorn

app = FastAPI(title="Hybrid Neural Style Transfer API")

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

@app.get("/")
async def health_check():
    return {"status": "ready", "engine": "tensorflow_weights"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
