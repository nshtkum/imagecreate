import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import io
import numpy as np
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

# Initialize HF client with error handling
@st.cache_resource
def init_hf_client():
    try:
        HF_TOKEN = st.secrets.get("HF_TOKEN")
        if not HF_TOKEN:
            st.warning("⚠️ HF_TOKEN not found in secrets. AI generation will be disabled.")
            return None
        return InferenceClient(
            model="stabilityai/stable-diffusion-3-medium",
            token=HF_TOKEN
        )
    except Exception as e:
        st.error(f"Failed to initialize HF client: {e}")
        return None

client = init_hf_client()

def create_property_overlay(img, primary_text, secondary_text):
    """Create a clean property overlay with bottom container"""
    
    # Ensure image is in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to standard social media banner size (16:9 aspect ratio)
    img = img.resize((1200, 675), Image.Resampling.LANCZOS)
    
    # Create overlay layer
    overlay = Image.new("RGBA", (1200, 675), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Color scheme
    container_red = (231, 76, 60)
    container_dark = (192, 57, 43)
    white = (255, 255, 255)
    
    # Container dimensions - covers bottom portion
    container_height = 140
    container_y = img.height - container_height
    
    # Create subtle gradient background for the container
    for y in range(container_height):
        # Create gradient from transparent to solid red
        gradient_factor = y / container_height
        alpha = int(200 + (55 * gradient_factor))  # From 200 to 255 alpha
        r = int(container_dark[0] + (container_red[0] - container_dark[0]) * gradient_factor)
        g = int(container_dark[1] + (container_red[1] - container_dark[1]) * gradient_factor)
        b = int(container_dark[2] + (container_red[2] - container_dark[2]) * gradient_factor)
        
        draw.rectangle([0, container_y + y, img.width, container_y + y + 1], 
                      fill=(r, g, b, alpha))
    
    # Add subtle shadow above container using transparency gradient
    shadow_height = 20
    for i in range(shadow_height):
        alpha = int(60 * (1 - i / shadow_height))  # Fade from 60 to 0
        draw.rectangle([0, container_y - shadow_height + i, img.width, container_y - shadow_height + i + 1], 
                      fill=(0, 0, 0, alpha))
    
    # Load fonts with better fallback system
    def get_font(size):
        fonts_to_try = [
            "arial.ttf", "calibri.ttf", "Arial.ttf", "Calibri.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        ]
        
        for font_path in fonts_to_try:
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
        
        return ImageFont.load_default()
    
    # Font sizes
    primary_font = get_font(38)
    secondary_font = get_font(26)
    
    # Text positioning within the container
    container_padding = 40
    text_start_x = container_padding
    text_start_y = container_y + 30
    
    # Helper function to draw text with subtle shadow
    def draw_text_with_shadow(text, position, font, text_color=white):
        x, y = position
        # Draw subtle shadow
        draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, 80))
        # Draw main text
        draw.text((x, y), text, font=font, fill=text_color)
    
    # Calculate text positions
    current_y = text_start_y
    
    # Primary text
    if primary_text:
        draw_text_with_shadow(primary_text, (text_start_x, current_y), primary_font)
        current_y += 50
    
    # Secondary text
    if secondary_text:
        draw_text_with_shadow(secondary_text, (text_start_x, current_y), secondary_font)
    
    # Composite the overlay onto the image
    final_img = Image.alpha_composite(img.convert("RGBA"), overlay)
    
    # Enhance the final image slightly
    enhancer = ImageEnhance.Contrast(final_img)
    final_img = enhancer.enhance(1.05)
    
    return final_img.convert("RGB")

# ----------------------------
# AI IMAGE GENERATION FUNCTION
# ----------------------------
def generate_ai_image(prompt):
    """Generate image using Hugging Face API with better error handling"""
    if not client:
        raise Exception("Hugging Face client not initialized")
    
    try:
        # Enhanced prompt for better quality
        enhanced_prompt = f"{prompt}, high quality, professional photography, 4k, detailed, realistic, architectural photography"
        
        # Generate image
        image = client.text_to_image(enhanced_prompt)
        
        # Ensure image is in the correct format
        if hasattr(image, 'mode') and image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        raise Exception(f"Image generation failed: {str(e)}")

# ----------------------------
# STREAMLIT UI
# ----------------------------

# Custom CSS for MagicBricks styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
    }
    .feature-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #e74c3c;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🏠 Property Image Generator</h1></div>', unsafe_allow_html=True)

# Sidebar for options
with st.sidebar:
    st.header("🎨 Design Options")
    
    st.subheader("📏 Output Settings")
    output_format = st.selectbox(
        "Image Format",
        ["PNG", "JPEG"],
        index=0
    )

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    # Image source selection
    st.subheader("🖼️ Image Source")
    image_source = st.radio(
        "Choose how to get your property image:",
        ["🤖 Generate with AI", "📁 Upload Your Own Image"],
        horizontal=True
    )
    
    if image_source == "🤖 Generate with AI":
        prompt = st.text_area(
            "📝 Describe your property",
            value="modern Indian apartment building exterior, bright daylight, professional architecture photography",
            height=100,
            help="Describe the property you want to generate"
        )
    else:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a property image"
        )

with col2:
    st.subheader("📝 Property Text")
    
    primary_text = st.text_input(
        "Primary Text",
        value="2BHK Apartment in Bangalore",
        help="Main headline text"
    )
    
    secondary_text = st.text_input(
        "Secondary Text", 
        value="₹85 Lakh • Ready to Move",
        help="Additional details text"
    )

# Generation section
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Generate Property Image", type="primary", use_container_width=True):
        # Validation
        if not primary_text.strip():
            st.error("🚫 Please enter primary text")
            st.stop()
        
        if image_source == "🤖 Generate with AI":
            if not prompt.strip():
                st.error("🚫 Please describe the property for AI generation")
                st.stop()
            
            if not client:
                st.error("🚫 AI generation is not available. Please check your HF_TOKEN in secrets.")
                st.stop()
        
        elif image_source == "📁 Upload Your Own Image":
            if not uploaded_file:
                st.error("🚫 Please upload an image file")
                st.stop()
        
        # Generate the image
        with st.spinner("🎨 Creating your property image..."):
            try:
                # Get base image
                if image_source == "🤖 Generate with AI":
                    base_image = generate_ai_image(prompt)
                else:
                    base_image = Image.open(uploaded_file)
                
                # Create the overlay
                final_image = create_property_overlay(
                    base_image,
                    primary_text.strip(),
                    secondary_text.strip()
                )
                
                # Success message
                st.markdown('<div class="success-box">✅ Property image generated successfully!</div>', unsafe_allow_html=True)
                
                # Display the image
                st.image(final_image, caption="Your Property Image", use_container_width=True)
                
                # Download and action buttons
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    # Save image to buffer
                    buf = io.BytesIO()
                    final_image.save(buf, format=output_format, quality=95)
                    byte_data = buf.getvalue()
                    
                    # Download button
                    st.download_button(
                        label=f"⬇️ Download {output_format}",
                        data=byte_data,
                        file_name=f"property_image_{primary_text.replace(' ', '_').lower()}.{output_format.lower()}",
                        mime=f"image/{output_format.lower()}",
                        use_container_width=True
                    )
                
                with col2:
                    if st.button("🔄 Regenerate", use_container_width=True):
                        st.rerun()
                
                with col3:
                    if st.button("ℹ️ Details", use_container_width=True):
                        st.info(f"Size: 1200×675px\nFormat: {output_format}")
                
                # Additional info
                with st.expander("📊 Image Information"):
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.markdown("**📐 Dimensions:** 1200 × 675 pixels")
                        st.markdown(f"**📁 Format:** {output_format}")
                    with info_col2:
                        st.markdown("**🎯 Optimized for:** Social Media")
                        st.markdown("**📱 Aspect Ratio:** 16:9")
                
            except Exception as e:
                st.error(f"❌ Error generating image: {str(e)}")
                st.info("💡 Try with a different image or check your internet connection")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 1rem;">
    <p>🏠 <strong>MagicBricks Style Property Image Generator</strong></p>
    <p>Create professional property images with MagicBricks-inspired design</p>
</div>
""", unsafe_allow_html=True)
