import sys
import os
import cv2

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from preprocessing.image_loader import load_image
from preprocessing.image_cleaner import denoise_image, sharpen_image
from nlp.text_cleaner import clean_text
from nlp.llm_suggestions import get_suggestions
from models.vit_caption import generate_caption
from evaluation.metrics import calculate_psnr, calculate_ssim
from evaluation.plots import plot_metrics

def run_pipeline(image_path, user_text):

    # ---------------------------
    # STEP 1: Load Original Image
    # ---------------------------
    original = load_image(image_path)

    # ---------------------------
    # STEP 2: NLP Processing
    # ---------------------------
    cleaned_text = clean_text(user_text)
    llm_output = get_suggestions(cleaned_text)

    # ---------------------------
    # STEP 3: Image Enhancement
    # ---------------------------
    enhanced = denoise_image(original)
    enhanced = sharpen_image(enhanced)

    # ---------------------------
    # STEP 4: Caption Generation
    # ---------------------------
     # STEP 4: Caption Generation
    caption = generate_caption(image_path)
    print("\n===== RESULTS =====")
    print("User Text:", cleaned_text)
    print("Caption:", caption)
    print("Suggestions:", llm_output)
    print("===================\n")
     # --------------------------
     # # STEP 5: Evaluation

    # ---------------------------
    # STEP 5: Evaluation
    # ---------------------------
    psnr_value = calculate_psnr(original, enhanced)
    ssim_value = calculate_ssim(original, enhanced)

    print("PSNR:", psnr_value)
    print("SSIM:", ssim_value)

    # Plot graph (single value demo)
    plot_metrics([psnr_value], [ssim_value])

    # ---------------------------
    # STEP 6: Save Output
    # ---------------------------
    cv2.imwrite("outputs/enhanced.jpg", enhanced)

    return enhanced, llm_output, caption, psnr_value, ssim_value
from database.db import insert_log
if __name__ == "__main__":
    run_pipeline(
        image_path="src/models/abd.jpg",
        user_text="Improve image quality"
    )