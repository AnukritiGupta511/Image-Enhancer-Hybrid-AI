# ✨ Hybrid AI Image Enhancer Pro

Welcome to the **Hybrid AI Image Enhancer Pro**! This project is an advanced, AI-powered image processing application built with Python and Streamlit. It leverages state-of-the-art machine learning pipelines and traditional computer vision algorithms to drastically improve image quality while providing rich analytical insights.

## 🌟 Key Features

- 🪄 **Image Denoising**: Utilizes Fast Non-Local Means Denoising to smooth out image grain without losing crucial details.
- 🔪 **Edge Sharpening**: Applies custom convolution kernels to enhance the clarity and crispness of your images.
- 📝 **Auto-Captioning**: Employs deep learning (Vision Transformer + GPT-2) to automatically generate descriptive captions of the uploaded images.
- 🤖 **LLM Suggestions**: NLP-driven feedback providing actionable suggestions based on your image prompts.
- 📊 **Quality Metrics**: Automatically calculates Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR) to objectively measure image improvements.
- 🕰️ **History Tracking**: Features a built-in SQLite database that archives your enhancement history, parameters, and generated outputs.

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (for a highly interactive, professional UI)
- **Computer Vision**: [OpenCV](https://opencv.org/) (Image processing & enhancement)
- **Machine Learning**: [Hugging Face Transformers](https://huggingface.co/) (`nlpconnect/vit-gpt2-image-captioning`)
- **Data Persistence**: SQLite Database

## 🚀 How to Run

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repository-url>
   cd Image-Enhancer-Hybrid-AI
   ```

2. **Install Dependencies**:
   Ensure you have the required packages installed.
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have `streamlit`, `opencv-python`, `transformers`, `torch`, `Pillow`, and other standard ML libraries installed).*

3. **Start the Application**:
   Run the Streamlit app from the root directory:
   ```bash
   streamlit run src/ui/app.py
   ```

4. **Access the App**:
   Open your browser and navigate to `http://127.0.0.1:8501`.

## 📁 Project Structure

```text
Image-Enhancer-Hybrid-AI/
├── Database/              # SQLite database storage
├── data/                  # Raw and processed data
├── models/                # Saved ML models / downloaded weights
├── outputs/               # Final enhanced output images
├── src/
│   ├── database/          # Database connection and queries
│   ├── evaluation/        # PSNR & SSIM metrics and plots
│   ├── models/            # ViT captioning logic
│   ├── nlp/               # Text cleaning and LLM suggestion logic
│   ├── pipeline/          # The core pipeline runner (full_pipeline.py)
│   ├── preprocessing/     # Denoising and sharpening algorithms
│   └── ui/                # Streamlit frontend (app.py)
└── README.md
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

---
*Powered by Hybrid AI — Enhancing pixels, one image at a time.*
