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
    # Resize to standard banner size
    img = img.resize((1200, 630))
    draw = ImageDraw.Draw(img)

    # Load font (fallback to default if not available)
    try:
        font_title = ImageFont.truetype("arial.ttf", 58)
        font_sub = ImageFont.truetype("arial.ttf", 38)
    except:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    # Gradient overlay at top (simulated by semi-transparent fade)
    gradient = Image.new("RGBA", (1200, 250), color=(0, 0, 0, 0))
    for y in range(250):
        opacity = int(180 * (1 - y / 250))  # fade from 180 to 0
        line = Image.new("RGBA", (1200, 1), (0, 0, 0, opacity))
        gradient.paste(line, (0, y))
    img = Image.alpha_composite(img.convert("RGBA"), gradient)

    # Add glow effect behind text
    def draw_glow_text(draw_obj, position, text, font, glow_color="black", text_color="white"):
        x, y = position
        # glow (shadow)
        for offset in [(1,1), (-1,-1), (-1,1), (1,-1)]:
            draw_obj.text((x+offset[0], y+offset[1]), text, font=font, fill=glow_color)
        # main text
        draw_obj.text(position, text, font=font, fill=text_color)

    # Text positions
    title_pos = (40, 40)
    subtext_pos = (40, 110)

    draw = ImageDraw.Draw(img)
    draw_glow_text(draw, title_pos, title, font_title)
    draw_glow_text(draw, subtext_pos, subtext, font_sub)

    return img.convert("RGB")

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
