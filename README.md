# 🎨 AI-Powered Image Generation App with Streamlit

This is a complete web-based application that uses **Artificial Intelligence** to generate and modify images using **three powerful techniques**:  
1. **Text-to-Image Generation**  
2. **Image Variation (Image-to-Image Translation)**  
3. **Image Inpainting (Editing with Mask)**  

All of this is built into a simple and interactive web interface using **Streamlit** and powered by **Stable Diffusion** models via Hugging Face's `diffusers` library.

---

## Project Purpose

This project was created to demonstrate how **Generative AI** can be integrated into a real-time web application. It serves as:
- A prototype for creative content generation tools.
- A practical demonstration of integrating open-source diffusion models in Python.
- A complete Streamlit-based AI application that can be easily deployed and scaled.

The goal is to provide **non-programmers and enthusiasts** with the ability to create stunning AI-generated images **without writing a single line of code** — just by using prompts and uploading images.

---

## What Can You Do With This App?

| Mode              | What it Does                                                                 |
|------------------|-------------------------------------------------------------------------------|
| Text-to-Image  | Type any creative sentence, and the AI will turn it into a realistic image. |
| Image Variation| Upload an image, and the AI will produce new versions with variations.       |
| Inpainting     | Upload an image and a mask to selectively edit parts of it using a prompt.   |

---

## Key Features

- 3 powerful AI image generation modes  
- Clean and user-friendly Streamlit interface  
- Built-in download buttons for saving results  
- CUDA/GPU support for fast processing  
- Automatic image resizing and formatting  
- Caching for efficient memory usage  

---

## Project Structure

```
.
├── app.py                     # Main Streamlit app
├── requirements.txt           # Required Python libraries
├── README.md                  # Documentation (this file)
└── Image_Generation.ipynb     # Whole work of my project
```

---

## How to Run This App Locally

> Prerequisite: Python 3.8+ and pip installed. GPU recommended but not required.

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/Image-Generation.git
cd Image-Generation
```

### Step 2: Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit app
```bash
streamlit run app.py
```

---

## How Each Mode Works

### 1. Text-to-Image Generation
- **What You Do**: Type a description like "a cat surfing on a watermelon in space".
- **What AI Does**: Uses Stable Diffusion to create a completely new image based on your text.
- **Behind the Scenes**: Calls `StableDiffusionPipeline` with your prompt, guidance scale, and inference steps.

---

### 2. Image Variation (Image-to-Image Translation)
- **What You Do**: Upload a base image and optionally add a prompt like "make it rainy".
- **What AI Does**: Creates a new image that’s visually similar but with variations.
- **Behind the Scenes**: Uses `StableDiffusionImg2ImgPipeline`, resizes the image to 512x512, and generates a new variation.

---

### 3. Inpainting (Image Editing with Mask)
- **What You Do**: Upload an image and a corresponding mask where white areas define the editable part, and give a prompt like "add a rocket".
- **What AI Does**: Reconstructs or fills in the masked part according to your prompt.
- **Behind the Scenes**: Uses `StableDiffusionInpaintPipeline`, resizing images to 512x512 and generating new pixels for the masked region.

---

## Technologies Used

- [Streamlit](https://streamlit.io/) – Fast Python web app framework  
- [Diffusers](https://github.com/huggingface/diffusers) – Stable Diffusion model interface  
- [PyTorch](https://pytorch.org/) – Machine learning backend  
- [Hugging Face Models](https://huggingface.co/runwayml/stable-diffusion-v1-5) – Pretrained AI models  
- [Pillow (PIL)](https://python-pillow.org/) – Image manipulation  

---

## requirements.txt Example

You can create a `requirements.txt` file like below:

```txt
streamlit
torch
transformers
accelerate
safetensors
diffusers
pillow
pyngrok
scipy
huggingface_hub
```

Install it with:
```bash
pip install -r requirements.txt
```

---

## Model Optimization

- All models use `@st.cache_resource` so they load only once.
- Runs on **GPU (CUDA)** if available, otherwise **CPU fallback**.
- Automatically resizes all images to 512×512 for Stable Diffusion compatibility.

---

## Acknowledgments

Special thanks to:
- [Hugging Face 🤗](https://huggingface.co) for making AI models accessible.
- [Stability AI](https://stability.ai/) for developing Stable Diffusion.
- The open-source community for building incredible tools.

---
