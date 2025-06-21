# 3️⃣ Save Streamlit app code as app.py
import os
from pyngrok import ngrok
import threading
import streamlit as st
from io import BytesIO
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline

@st.cache_resource(show_spinner=False)
def load_text2img_model():
    return StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def load_img2img_model():
    return StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def load_inpaint_model():
    return StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        revision="fp16" if torch.cuda.is_available() else "main",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

pipe_txt2img = load_text2img_model()
pipe_img2img = load_img2img_model()
pipe_inpaint = load_inpaint_model()

st.set_page_config(page_title="Image Generation App", layout="centered")

st.title("Image Generation - Text2Image, Variation & Inpainting")

mode = st.radio("Select Operation", ["Text-to-Image", "Image Variation", "Inpainting"])

if mode == "Text-to-Image":
    prompt = st.text_area("Enter your prompt", height=150)
    steps = st.slider("Inference Steps", 10, 50, 30)
    guidance_scale = st.slider("Guidance Scale", 5.0, 15.0, 7.5)
    if st.button("Generate Image"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Generating image..."):
                image = pipe_txt2img(prompt=prompt, guidance_scale=guidance_scale, num_inference_steps=steps).images[0]
            st.image(image, caption=prompt)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            st.download_button("Download Image", data=buffered.getvalue(), file_name="text2img.png", mime="image/png")

elif mode == "Image Variation":
    uploaded_file = st.file_uploader("Upload an image (PNG/JPEG)", type=["png", "jpg", "jpeg"])
    prompt = st.text_area("Optional prompt for variation", height=100)
    steps = st.slider("Inference Steps", 10, 50, 30)
    guidance_scale = st.slider("Guidance Scale", 5.0, 15.0, 7.5)

    if uploaded_file is not None:
        init_image = Image.open(uploaded_file).convert("RGB")
        st.image(init_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Generate Variation"):
            with st.spinner("Generating image variation..."):
                init_image = init_image.resize((512, 512))
                image = pipe_img2img(prompt=prompt if prompt.strip() else None,
                                     image=init_image,
                                     strength=0.75,
                                     guidance_scale=guidance_scale,
                                     num_inference_steps=steps).images[0]
            st.image(image, caption="Image Variation")
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            st.download_button("Download Image", data=buffered.getvalue(), file_name="variation.png", mime="image/png")

elif mode == "Inpainting":
    uploaded_file = st.file_uploader("Upload original image (PNG/JPEG)", type=["png", "jpg", "jpeg"])
    mask_file = st.file_uploader("Upload mask image (white=area to modify)", type=["png", "jpg", "jpeg"])
    prompt = st.text_area("Inpainting prompt (e.g., 'Add a spaceship')", height=100)
    steps = st.slider("Inference Steps", 10, 50, 30)
    guidance_scale = st.slider("Guidance Scale", 5.0, 15.0, 7.5)

    if uploaded_file is not None and mask_file is not None:
        original = Image.open(uploaded_file).convert("RGB").resize((512, 512))
        mask = Image.open(mask_file).convert("RGB").resize((512, 512))
        st.image(original, caption="Original Image", use_column_width=True)
        st.image(mask, caption="Mask Image", use_column_width=True)

        if st.button("Generate Inpainting"):
            if not prompt.strip():
                st.warning("Please enter an inpainting prompt.")
            else:
                with st.spinner("Generating inpainting..."):
                    image = pipe_inpaint(prompt=prompt,
                                         image=original,
                                         mask_image=mask,
                                         guidance_scale=guidance_scale,
                                         num_inference_steps=steps).images[0]
                st.image(image, caption="Inpainting Result")
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                st.download_button("Download Image", data=buffered.getvalue(), file_name="inpaint.png", mime="image/png")

