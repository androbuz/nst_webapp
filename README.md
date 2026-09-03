---
title: Hybrid Neural Style Transfer
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Hybrid Neural Style Transfer

Get access to the web application on <a href="https://andro777-hybrid-nst-demo.hf.space/">Hugging Face</a>. Upload a content image and either a style image or a text description to generate a stylized artwork.

## How to use

1. Choose **Image Style** to upload a reference style image.
2. Choose **Text Style** to describe the desired style in words.
3. Click **Generate** and wait for the result.

## Technical Details

- **Backend**: FastAPI
- **Models**: Quantized TensorFlow Lite (TFLite)
- **Frontend**: HTML + CSS + JavaScript
- **Deployment**: Docker on Hugging Face Spaces
