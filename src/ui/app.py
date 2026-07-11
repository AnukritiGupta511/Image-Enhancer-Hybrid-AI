import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import streamlit as st

from src.pipeline.full_pipeline import run_pipeline
from PIL import Image

# Set page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="AI Image Enhancer Pro",
    page_icon="🪄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.info(
        "Welcome to the **Hybrid AI Image Enhancer**! \n\n"
        "Upload an image, give instructions, and let our AI models process it."
    )
    st.divider()
    st.markdown("### 🧩 Features Overview")
    st.markdown(
        "- 🪄 **Denoising:** Reduces grain and noise.\n"
        "- 🔪 **Sharpening:** Enhances edge clarity.\n"
        "- 📝 **Auto-Caption:** Generates AI descriptions.\n"
        "- 📊 **Metrics:** PSNR & SSIM evaluation."
    )
    st.divider()
    st.caption("Version 2.0 Pro | Powered by Hybrid AI Pipeline")

# --- MAIN HEADER ---
st.title("🪄 AI Image Enhancer Pro")
st.markdown("##### *Transform and elevate your images with state-of-the-art Machine Learning models.*")
st.divider()

# --- TABS ---
tab1, tab2 = st.tabs(["🚀 Enhancement Studio", "🕰️ Processing History"])

with tab1:
    # Use columns to separate inputs from outputs
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("1. Setup")
        
        # Nicer upload area
        uploaded = st.file_uploader("📥 Upload an image", type=["jpg", "png", "jpeg"])
        
        # Text input with a nice label
        user_text = st.text_area(
            "✍️ AI Instructions", 
            placeholder="e.g., Please make this image sharper and reduce noise...", 
            help="Tell the AI what you want to achieve with this image."
        )
        
        process_btn = st.button(
            "✨ Start Magic Enhancement", 
            type="primary", 
            use_container_width=True,
            disabled=not (uploaded and user_text)
        )
        
        if not uploaded:
            st.info("ℹ️ Awaiting your image upload.")
        elif not user_text:
            st.warning("⚠️ Please provide some AI instructions in the text box above to enable the button.")

    with col2:
        st.subheader("2. Results Preview")
        
        if uploaded:
            if process_btn:
                # Use st.status for a very professional loading sequence
                with st.status("🚀 Processing image through AI Pipeline...", expanded=True) as status:
                    st.write("1️⃣ Loading and preprocessing image...")
                    # Save temporary file
                    with open("temp.jpg", "wb") as f:
                        f.write(uploaded.getbuffer())

                    try:
                        st.write("2️⃣ Running neural enhancement models...")
                        # Run the enhancement pipeline
                        result = run_pipeline("temp.jpg", user_text)
                        
                        st.write("3️⃣ Extracting insights and calculating metrics...")
                        # result parsing
                        enhanced = result[0]
                        suggestion = result[1] if len(result) > 1 else "No suggestion provided."
                        caption = result[2] if len(result) > 2 else "No caption provided."
                        psnr = result[3] if len(result) > 3 else "N/A"
                        ssim = result[4] if len(result) > 4 else "N/A"
                        
                        status.update(label="✅ Enhancement Complete!", state="complete", expanded=False)
                        st.balloons() # Fun professional celebration!
                        
                        # Display Images side by side
                        img_col1, img_col2 = st.columns(2)
                        with img_col1:
                            st.markdown("**Original Image**")
                            st.image(Image.open("temp.jpg"))
                        with img_col2:
                            st.markdown("**✨ Enhanced Image**")
                            st.image(enhanced)

                        st.divider()

                        # Insights Section
                        st.subheader("🤖 AI Analysis")
                        insight_col1, insight_col2 = st.columns(2)
                        with insight_col1:
                            st.info(f"**Generated Caption:**\n\n> {caption}")
                        with insight_col2:
                            st.success(f"**AI Suggestion:**\n\n> {suggestion}")

                        st.divider()

                        # Metrics Section
                        st.subheader("📊 Quality Metrics")
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        with metric_col1:
                            psnr_val = f"{psnr:.2f}" if isinstance(psnr, (int, float)) else psnr
                            st.metric(label="Signal-to-Noise Ratio (PSNR)", value=f"{psnr_val} dB")
                        with metric_col2:
                            ssim_val = f"{ssim:.4f}" if isinstance(ssim, (int, float)) else ssim
                            st.metric(label="Structural Similarity (SSIM)", value=ssim_val)
                        with metric_col3:
                            quality = "Excellent" if isinstance(ssim, (int, float)) and ssim > 0.8 else "Good" if isinstance(ssim, (int, float)) and ssim > 0.6 else "Needs Work"
                            if not isinstance(ssim, (int, float)):
                                quality = "N/A"
                            st.metric(label="Overall Assessment", value=quality)

                    except Exception as e:
                        status.update(label="❌ Error during processing", state="error")
                        st.error(f"An error occurred: {str(e)}")
            else:
                # Placeholder while waiting for the button press
                st.image(Image.open(uploaded))
                st.caption("👆 Ready for processing. Click the **Start Magic Enhancement** button.")
        else:
            # Empty state
            st.markdown(
                """
                <div style="border: 2px dashed #ccc; padding: 50px; text-align: center; border-radius: 10px; color: #888;">
                    <h4>No Image Uploaded</h4>
                    <p>Your enhanced results will appear here.</p>
                </div>
                """, unsafe_allow_html=True
            )

with tab2:
    col_hist_1, col_hist_2 = st.columns([3, 1])
    with col_hist_1:
        st.subheader("📂 Enhancement Archive")
        st.write("Browse through the database of your previously enhanced images.")
    with col_hist_2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            pass # Forces a rerun
    
    st.divider()
    
    try:
        from src.database.db import fetch_logs
        logs = fetch_logs()
        if logs and len(logs) > 0:
            for row in logs:
                # Use a nice expander for each log entry
                with st.expander(f"✨ Job #{row[0]} — {row[6]} (Quality: {row[2]})"):
                    st.write(f"**Original Image:** `{row[1]}`")
                    st.write(f"**Enhanced Output:** `{row[5]}`")
                    st.write(f"**AI Caption:** {row[3]}")
                    st.write(f"**Suggestions:** {row[4]}")
        else:
            st.info("No enhancement history found. Try enhancing an image first!")
    except Exception as e:
        st.warning(f"Could not load history: {e}. Ensure the SQLite database is properly initialized.")