import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import streamlit as st

from src.pipeline.full_pipeline import run_pipeline
from PIL import Image

st.title("AI Image Enhancer with Evaluation")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png"])
user_text = st.text_input("Enter instruction")

if uploaded:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded.getbuffer())

    result = run_pipeline("temp.jpg", user_text)

    enhanced, suggestion, caption, psnr, ssim = result

    st.subheader("Original Image")
    st.image(Image.open("temp.jpg"))

    st.subheader("Enhanced Image")
    st.image(enhanced)

    st.write("AI Suggestion:", suggestion)
    st.write("Image Caption:", caption)

    st.subheader("Evaluation Metrics")
    st.write("PSNR:", psnr)
    st.write("SSIM:", ssim)
    from src.database.db import fetch_logs
    st.header("📦 Previous Enhancements (Database)")

if st.button("Show History"):
    logs = fetch_logs()
    for row in logs:
        st.write(f"ID: {row[0]}")
        st.write(f"Image Path: {row[1]}")
        st.write(f"Quality: {row[2]}")
        st.write(f"Caption: {row[3]}")
        st.write(f"Suggestions: {row[4]}")
        st.write(f"Enhanced Image: {row[5]}")
        st.write(f"Time: {row[6]}")
        st.write("---")