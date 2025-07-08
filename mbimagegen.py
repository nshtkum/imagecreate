import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import io
import requests
from typing import Optional

# ----------------------------
# CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="Property Image Generator", 
    layout="centered",
    page_icon="🏠",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# SCALE TO ZERO FUNCTION
# ----------------------------
def scale_endpoint_to_zero(model_id="stabilityai/stable-diffusion-3-medium"):
    """Scale Hugging Face endpoint to zero to save cost"""
    try:
        hf_token = st.secrets.get("HF_TOKEN")
        if not hf_token:
            st.warning("Cannot scale endpoint: HF_TOKEN not found in secrets.")
            return
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            f"https://api.endpoints.huggingface.cloud/endpoints/{model_id}/scale-to-zero",
            headers=headers
        )
        if response.status_code == 200:
            st.success("🛑 Endpoint scaled to zero after generation.")
        else:
            st.warning(f"⚠️ Failed to scale down: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Exception while scaling down: {str(e)}")

# ----------------------------
# HUGGING FACE CLIENT INIT
# ----------------------------
@st.cache_resource
def init_hf_client():
    try:
        HF_TOKEN = st.secrets.get("HF_TOKEN")
        if not HF_TOKEN:
            st.warning("⚠️ HF_TOKEN not found in secrets.")
            return None
        return InferenceClient(
            model="stabilityai/stable-diffusion-3-medium",
            token=HF_TOKEN
        )
    except Exception as e:
        st.error(f"Failed to initialize HF client: {e}")
        return None

client = init_hf_client()

# ----------------------------
# IMAGE GENERATION FUNCTION
# ----------------------------
def generate_ai_image(prompt):
    if not client:
        raise Exception("Hugging Face client not initialized")

    enhanced_prompt = f"{prompt}, high quality, 4k, professional, no text"
    image = client.text_to_image(enhanced_prompt)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

# ----------------------------
# OVERLAY FUNCTION
# ----------------------------
def create_property_overlay(img, primary_text, secondary_text):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((1200, 675), Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    gradient_start = img.height - 200
    for y in range(200):
        a = int(15 + (200 * (y / 200)**3))
        draw.rectangle([0, gradient_start + y, img.width, gradient_start + y + 1], fill=(231, 76, 60, a))

    def get_font(size):
        for font_path in [
            "arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica-Bold.ttf"
        ]:
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
        return ImageFont.load_default()

    font_primary = get_font(36)
    font_secondary = get_font(24)

    def draw_text(text, pos, font):
        x, y = pos
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 160))
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    if primary_text:
        w, h = draw.textsize(primary_text, font=font_primary)
        draw_text(primary_text, ((img.width - w) // 2, img.height - 160), font_primary)

    if secondary_text:
        w, h2 = draw.textsize(secondary_text, font=font_secondary)
        draw_text(secondary_text, ((img.width - w) // 2, img.height - 110), font_secondary)

    result = Image.alpha_composite(img.convert("RGBA"), overlay)
    enhancer = ImageEnhance.Contrast(result)
    return enhancer.enhance(1.05).convert("RGB")

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.markdown('<h1 style="text-align: center; color: white; background: #e74c3c; padding: 1rem; border-radius: 10px;">🏠 Property Image Generator</h1>', unsafe_allow_html=True)

st.sidebar.header("🔧 Settings")
output_format = st.sidebar.selectbox("Download Format", ["PNG", "JPEG"])

image_source = st.radio("Image Source", ["🤖 Generate with AI", "📁 Upload Your Own Image"])
if image_source == "🤖 Generate with AI":
    prompt = st.text_area("Describe your property", value="Diwali Festival home with lights")
else:
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

primary_text = st.text_input("Primary Text", value="Diwali Festival")
secondary_text = st.text_input("Secondary Text", value="Celebrate with Light")

if st.button("🎨 Generate Property Image", use_container_width=True):
    try:
        if image_source == "🤖 Generate with AI":
            if not prompt.strip():
                st.error("Please describe the property")
                st.stop()
            base_image = generate_ai_image(prompt)
        else:
            if not uploaded_file:
                st.error("Please upload an image file")
                st.stop()
            base_image = Image.open(uploaded_file)

        final_image = create_property_overlay(base_image, primary_text, secondary_text)
        st.success("✅ Image generated successfully!")
        st.image(final_image, use_container_width=True)

        buffer = io.BytesIO()
        final_image.save(buffer, format=output_format)
        st.download_button("⬇️ Download", buffer.getvalue(), file_name=f"property.{output_format.lower()}", mime=f"image/{output_format.lower()}")

        # 🔁 SCALE TO ZERO TO SAVE COST
        scale_endpoint_to_zero()

    except Exception as e:
        st.error(f"Error: {e}")
