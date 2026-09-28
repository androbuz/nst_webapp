from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from PIL import Image
import uvicorn
import os
import shutil
import uuid

from app.inference import run_style_transfer_image, run_style_transfer_text, run_style_transfer_video

app = FastAPI(title="Hybrid Neural Style Transfer API")

# mount static files (CSS, JS)
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

# serve the main html file as homepage
@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_path, 'index.html'))

@app.head("/")
async def read_index_head():
    return FileResponse(os.path.join(static_path, 'index.html'))

# endpoint to send content image to be styled and its style image
@app.post("/style-transfer-image")
async def stylize_image(content_file: UploadFile = File(...), style_file: UploadFile = File(...)):
    # open the content and style images
    content_pil = Image.open(BytesIO(await content_file.read()))
    style_pil = Image.open(BytesIO(await style_file.read()))
    # get the styled image from the model 
    result_img = run_style_transfer_image(content_pil, style_pil)
    # write the image data to a jpeg file
    img_io = BytesIO()
    result_img.save(img_io, 'JPEG')
    img_io.seek(0)
    return StreamingResponse(img_io, media_type="image/jpeg")

# endpoint to send content image and a text prompt for style
@app.post("/style-transfer-text")
async def stylize_text(content_file: UploadFile = File(...), prompt: str = Form(...)):
    content_pil = Image.open(BytesIO(await content_file.read()))
    # get the styled image from the model 
    result_img = run_style_transfer_text(content_pil, prompt)
    # write the image data to a jpeg file
    img_io = BytesIO()
    result_img.save(img_io, 'JPEG')
    img_io.seek(0)
    return StreamingResponse(img_io, media_type="image/jpeg")

# endpoint for video-based style transfer
@app.post("/style-transfer-video")
async def stylize_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    style_file: UploadFile = None,
    prompt: str = Form(None)
):
    # Create paths for temp video files
    temp_id = str(uuid.uuid4())
    input_ext = os.path.splitext(video_file.filename)[1] or ".mp4"
    input_path = f"/tmp/input_{temp_id}{input_ext}"
    output_path = f"/tmp/output_{temp_id}.mp4"

    # saving uploaded video file locally
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(video_file.file, buffer)

    style_pil = None
    if style_file:
        style_pil = Image.open(BytesIO(await style_file.read()))

    # get the output video from the model
    run_style_transfer_video(
        input_video_path=input_path, 
        output_video_path=output_path, 
        style_pil=style_pil, 
        style_prompt=prompt
    )

    # clean up files in background after response is sent
    def cleanup_temp_files():
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

    background_tasks.add_task(cleanup_temp_files)
    return FileResponse(output_path, media_type="video/mp4", filename=f"stylized_{video_file.filename}")

# adding a health status check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ready", "engine": "tensorflow_weights"}

@app.head("/health")
async def health_check_head():
    return {"status": "ready"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
