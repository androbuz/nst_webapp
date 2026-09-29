---
title: Hybrid Neural Style Transfer
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Hybrid Neural Style Transfer (NST) System

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow Lite](https://img.shields.io/badge/Inference-TFLite-FF6F00.svg?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![DVC](https://img.shields.io/badge/Data_Version_Control-DVC-9CF.svg?style=flat&logo=data-version-control&logoColor=white)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

An end-to-end, edge-optimized **Hybrid Neural Style Transfer** system featuring custom Transformer architectures, cross-modal integration, and a production-grade MLOps deployment pipeline.

### Live Web Application Demo
Experience the live application hosted on Hugging Face Spaces:  
[https://andro777-hybrid-nst-demo.hf.space/](https://andro777-hybrid-nst-demo.hf.space/)

---

## Project Overview & Prototype

The system interface allows users to upload a content image and specify a style source through two distinct input methods. The first method accepts an image as a style reference, while the second processes a natural language prompt. Furthermore, the user can upload a video for temporal style transfer. These inputs are then transmitted as multipart HTTP requests to a FastAPI backend.

The server-side inference pipeline is managed by a modular service that loads three pre-quantized TFLite models at startup: an image-guided model, a text-guided model, and a CLIP text encoder. The image-guided model accepts two 256 by 256 RGB images as input. For the text-guided path, the CLIP processor tokenizes the user prompt, which is then converted into a 512-dimensional embedding by the quantized encoder. Both style transfer models utilize dynamic range quantization, reducing the model size by approximately 70% compared to the original Keras weights and facilitating deployment on resource-constrained platforms.

The backend exposes three POST endpoints, `/style-transfer-image`, `/style-transfer-text`, and `/style-transfer-video`, to handle input validation, image preprocessing, and model execution. The inference cycle, from initial processing to the generation of a JPEG output, is completed in approximately two seconds on a standard CPU. This performance is achieved through the use of TFLite and fixed input dimensions.

---

## System Architecture & Flowchart

```
                    +------------------------+
                    |  User Upload  |
                    +-----------+------------+
                                |
         +----------------------+----------------------+
         | (Content Image / Video & Style Reference)   |
         v                                             v
+-----------------------+                    +--------------------+
|  Image Style Branch   |                    | Text Prompt Branch |
| (256x256 Input) |                          | (Natural Language) |
+-----------+-----------+                    +---------+----------+
            |                                          |
            v                                          v
+-----------------------+                    +--------------------+
| Transformer Encoder   |                    | CLIP Text Encoder  |
|       & CAPE          |                    |     (512 dim)      |
+-----------+-----------+                    +---------+----------+
            |                                          |
            +-------------------+----------------------+
                                |
                                v
                     +----------------------+
                     |  Refinement Decoder  |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     |   JPEG / MP4 Output   |
                     +----------------------+
```

---

## Datasets Curation & Usage

To train, evaluate, and benchmark the Hybrid NST system, four distinct datasets were curated and leveraged, each serving a unique role in the pipeline:

* **WikiArt (Style Dataset)**: Used for extracting artistic features during training.
* **MS-COCO (Content Dataset)**: Serves as the primary source of natural content images for static training. 
* **Davis 2017 (Video Dataset)**: Used specifically for training the temporal video styling branch.
* **NPRgeneral (Benchmark Dataset)**: Comprises 20 unstylized benchmark images, utilized as an independent evaluation baseline to compare our model against established baselines (such as Magenta and AdaIN) on objective content and style reconstructions.

---

## MLOps Pipeline & Automated GitHub Export

An automated pipeline automates the synchronization between the model training environment on Kaggle and the production code repository. Upon completion of a training session, a custom export pipeline processes the high-precision model weights, converting them to a compressed FP16 format. This conversion reduces storage requirements by approximately 50% with negligible impact on stylization quality. Following compression, these weights are registered using `dvc add`. A subsequent `dvc push` command uploads the binary files to the Backblaze B2 cloud storage. Concurrently, the modified `.dvc` files are committed to the GitHub repository. This process establishes a unified reference point, clearly defining the compatibility between specific model versions and the corresponding application code.

---

## Secure Production Deployment

For deployment, Docker is used within the Hugging Face Spaces environment to establish a consistent and reproducible runtime. To enhance security, cloud credentials are not hardcoded into the system. Instead, the Dockerfile employs ARG instructions to securely integrate Hugging Face Secrets during the container build process. This mechanism enables the container to authenticate with the S3 remote storage. Subsequently, `dvc pull` is executed to download the precise model weights necessary for the web application during the build phase. This methodology ensures that the production environment is consistently configured with the correct dependencies and model weights, facilitating reliable rollbacks and uniform performance across various computing instances.
