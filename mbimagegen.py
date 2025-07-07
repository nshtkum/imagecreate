import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw, ImageFont
import io

# ----------------------------
# CONFIGURATION
# ----------------------------

HF_TOKEN = st.secrets["HF_TOKEN"]  # stored securely in secrets.toml

client = InferenceClient(
    model="stabilityai/stable-diffusion-3-medium",
    token=HF_TOKEN,
    provider="fal-ai"
)

# ----------------------------
# TEXT OVERLAY FUNCTION (IMPROVED)
# ----------------------------

def overlay_text(img, title, subtext):
    # Resize to 1200x630
    img = img.resize((1200, 630))
    draw = ImageDraw.Draw(img)

    # Try to load font
    try:
        font_title = ImageFont.truetype("arial.ttf", 60)
        font_sub = ImageFont.truetype("arial.ttf", 40)
    except:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    # Add a semi-transparent black bar at the top
    bar_height = 160
    overlay = Image.new("RGBA", (1200, bar_height), (0, 0, 0, 180))  # black with opacity
    img.paste(overlay, (0, 0), overlay)

    # Draw title and subtext on top of the bar
    draw.text((40, 40), title, font=font_title, fill="white")
    draw.text((40, 100), subtext, font=font_sub, fill="white")

    return img

# ----------------------------
# STREAMLIT UI
# ----------------------------

st.set_page_config(page_title="Featured Image Generator", layout="centered")
st.title("🏠 Magicbricks Featured Image Generator")

with st.form("input_form"):
    prompt = st.text_area("🖼️ Image Prompt", value="modern Indian apartment in Bangalore, garden view, bright daylight", height=100)
    title = st.text_input("📝 Main Text (overlay)", value="2BHK Flat in Bangalore")
    subtext = st.text_input("📍 Subtext (price/location)", value="₹85 Lakh | Ready to Move")

    submitted = st.form_submit_button("🚀 Generate Featured Image")

if submitted:
    with st.spinner("Generating image using Hugging Face..."):
        try:
            image = client.text_to_image(prompt)
            final_img = overlay_text(image, title, subtext)

            st.image(final_img, caption="✅ Featured Image Preview")

            # Download option
            buf = io.BytesIO()
            final_img.save(buf, format="PNG")
            byte_data = buf.getvalue()

            st.download_button(
                label="⬇️ Download Featured Image",
                data=byte_data,
                file_name="featured_image.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"❌ Failed to generate image: {e}")
