import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw, ImageFont
import io

# ----------------------------
# CONFIGURATION
# ----------------------------

HF_TOKEN = st.secrets["HF_TOKEN"]

client = InferenceClient(
    model="stabilityai/stable-diffusion-3-medium",
    token=HF_TOKEN,
    provider="fal-ai"
)

# ----------------------------
# UTILITY FUNCTION TO ADD TEXT
# ----------------------------

def overlay_text(img, title, subtext):
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("arial.ttf", 42)
        font_sub = ImageFont.truetype("arial.ttf", 28)
    except:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    W, _ = img.size
    draw.text(((W - draw.textlength(title, font=font_title)) / 2, 30), title, font=font_title, fill="white")
    draw.text(((W - draw.textlength(subtext, font=font_sub)) / 2, 90), subtext, font=font_sub, fill="white")
    return img

# ----------------------------
# STREAMLIT UI
# ----------------------------

st.set_page_config(page_title="Featured Image Generator", layout="centered")
st.title("🏠 Magicbricks Featured Image Generator")

with st.form("input_form"):
    prompt = st.text_area("🖼️ Image Prompt", value="modern Indian apartment in Bangalore, garden view, bright daylight, listing style", height=100)
    title = st.text_input("📝 Main Text (overlay)", value="2BHK Flat in Bangalore")
    subtext = st.text_input("📍 Subtext (price/location)", value="₹85 Lakh | Ready to Move")

    submitted = st.form_submit_button("🚀 Generate Featured Image")

if submitted:
    with st.spinner("Generating image using Hugging Face..."):
        try:
            image = client.text_to_image(prompt)
            final_img = overlay_text(image, title, subtext)

            st.image(final_img, caption="✅ Featured Image Generated")

            # Download option
            buffer = io.BytesIO()
            final_img.save(buffer, format="PNG")
            byte_data = buffer.getvalue()

            st.download_button(
                label="⬇️ Download Image",
                data=byte_data,
                file_name="featured_image.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"❌ Failed to generate image: {e}")
