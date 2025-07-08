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
    page_title="MagicBricks Style Image Generator", 
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

# ----------------------------
# MAGICBRICKS STYLE OVERLAY FUNCTION
# ----------------------------
def create_magicbricks_overlay(img, title, subtitle, price, location, features_list=None):
    """Create a MagicBricks style overlay with bottom container"""
    
    # Ensure image is in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to standard social media banner size (16:9 aspect ratio)
    img = img.resize((1200, 675), Image.Resampling.LANCZOS)
    
    # Create overlay layer
    overlay = Image.new("RGBA", (1200, 675), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # MagicBricks color scheme
    mb_red = (231, 76, 60)      # MagicBricks red
    mb_dark_red = (192, 57, 43)  # Darker red for gradient
    white = (255, 255, 255)
    light_gray = (236, 240, 241)
    dark_gray = (52, 73, 94)
    
    # Container dimensions - covers bottom portion like in the image
    container_height = 200  # Increased height for better content spacing
    container_y = img.height - container_height
    
    # Create gradient background for the container (red gradient like MagicBricks)
    for y in range(container_height):
        # Create gradient from darker red at top to brighter red at bottom
        gradient_factor = y / container_height
        r = int(mb_dark_red[0] + (mb_red[0] - mb_dark_red[0]) * gradient_factor)
        g = int(mb_dark_red[1] + (mb_red[1] - mb_dark_red[1]) * gradient_factor)
        b = int(mb_dark_red[2] + (mb_red[2] - mb_dark_red[2]) * gradient_factor)
        
        draw.rectangle([0, container_y + y, img.width, container_y + y + 1], 
                      fill=(r, g, b, 240))  # Slightly transparent
    
    # Add a subtle shadow at the top of the container
    shadow_height = 8
    for i in range(shadow_height):
        alpha = int(80 * (1 - i / shadow_height))
        draw.rectangle([0, container_y - shadow_height + i, img.width, container_y - shadow_height + i + 1], 
                      fill=(0, 0, 0, alpha))
    
    # Load fonts with better fallback system
    def get_font(size, weight='normal'):
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
        
        # Final fallback
        return ImageFont.load_default()
    
    # Font sizes
    title_font = get_font(42, 'bold')
    subtitle_font = get_font(28)
    price_font = get_font(36, 'bold')
    location_font = get_font(24)
    features_font = get_font(20)
    
    # Text positioning within the container
    container_padding = 40
    text_start_x = container_padding
    text_start_y = container_y + 20
    
    # Helper function to draw text with shadow
    def draw_text_with_shadow(text, position, font, text_color=white, shadow_color=(0, 0, 0, 100)):
        x, y = position
        # Draw shadow
        draw.text((x + 2, y + 2), text, font=font, fill=shadow_color)
        # Draw main text
        draw.text((x, y), text, font=font, fill=text_color)
    
    # Calculate text positions
    current_y = text_start_y
    
    # Title (main property type)
    if title:
        draw_text_with_shadow(title, (text_start_x, current_y), title_font)
        current_y += 50
    
    # Subtitle (property details)
    if subtitle:
        draw_text_with_shadow(subtitle, (text_start_x, current_y), subtitle_font)
        current_y += 35
    
    # Price and Location on the same line
    if price:
        draw_text_with_shadow(price, (text_start_x, current_y), price_font)
        
        # Get price text width to position location next to it
        price_bbox = draw.textbbox((0, 0), price, font=price_font)
        price_width = price_bbox[2] - price_bbox[0]
        
        if location:
            location_x = text_start_x + price_width + 60  # Space between price and location
            draw_text_with_shadow(f"📍 {location}", (location_x, current_y + 8), location_font)
        
        current_y += 45
    
    # Features list (if provided)
    if features_list:
        features_text = " • ".join(features_list)
        # Wrap long features text
        max_width = img.width - (container_padding * 2)
        words = features_text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=features_font)
            if bbox[2] - bbox[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    current_line.append(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        # Draw feature lines
        for line in lines[:2]:  # Limit to 2 lines
            draw_text_with_shadow(line, (text_start_x, current_y), features_font)
            current_y += 25
    
    # Add a subtle border at the top of the container
    draw.rectangle([0, container_y, img.width, container_y + 2], fill=white)
    
    # Composite the overlay onto the image
    final_img = Image.alpha_composite(img.convert("RGBA"), overlay)
    
    # Enhance the final image
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

st.markdown('<div class="main-header"><h1>🏠 MagicBricks Style Property Image Generator</h1></div>', unsafe_allow_html=True)

# Sidebar for options
with st.sidebar:
    st.header("🎨 MagicBricks Style Options")
    st.markdown("*Create professional property images with MagicBricks-style bottom overlay*")
    
    st.subheader("📏 Output Settings")
    output_format = st.selectbox(
        "Image Format",
        ["PNG", "JPEG"],
        index=0
    )
    
    st.subheader("✨ Features")
    st.markdown("• **MagicBricks Red Theme**")
    st.markdown("• **Bottom Container Overlay**")
    st.markdown("• **Professional Typography**")
    st.markdown("• **Gradient Backgrounds**")
    st.markdown("• **Social Media Ready (1200x675)**")

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
    st.subheader("📝 Property Details")
    
    title = st.text_input(
        "🏷️ Property Type",
        value="2BHK Apartment",
        help="e.g., 2BHK Apartment, 3BHK Villa, etc."
    )
    
    subtitle = st.text_input(
        "📋 Property Details",
        value="Ready to Move • Fully Furnished",
        help="e.g., Ready to Move • Fully Furnished"
    )
    
    price = st.text_input(
        "💰 Price",
        value="₹85 Lakh",
        help="e.g., ₹85 Lakh, ₹50K/month"
    )
    
    location = st.text_input(
        "📍 Location",
        value="Whitefield, Bangalore",
        help="e.g., Whitefield, Bangalore"
    )
    
    # Features as multiselect
    feature_options = [
        "Parking", "24/7 Security", "Gym", "Swimming Pool", 
        "Garden", "Elevator", "Power Backup", "Water Supply",
        "Intercom", "Maintenance Staff", "Children's Play Area",
        "Club House", "Jogging Track", "CCTV Surveillance"
    ]
    
    selected_features = st.multiselect(
        "🎯 Property Features",
        feature_options,
        default=["Parking", "24/7 Security", "Gym"],
        help="Select key features of the property"
    )

# Generation section
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Generate MagicBricks Style Image", type="primary", use_container_width=True):
        # Validation
        if not title.strip():
            st.error("🚫 Please enter a property type")
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
        with st.spinner("🎨 Creating your MagicBricks style property image..."):
            try:
                # Get base image
                if image_source == "🤖 Generate with AI":
                    base_image = generate_ai_image(prompt)
                else:
                    base_image = Image.open(uploaded_file)
                
                # Create the MagicBricks overlay
                final_image = create_magicbricks_overlay(
                    base_image,
                    title.strip(),
                    subtitle.strip(),
                    price.strip(),
                    location.strip(),
                    selected_features
                )
                
                # Success message
                st.markdown('<div class="success-box">✅ MagicBricks style image generated successfully!</div>', unsafe_allow_html=True)
                
                # Display the image
                st.image(final_image, caption="Your MagicBricks Style Property Image", use_container_width=True)
                
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
                        file_name=f"magicbricks_property_{title.replace(' ', '_').lower()}.{output_format.lower()}",
                        mime=f"image/{output_format.lower()}",
                        use_container_width=True
                    )
                
                with col2:
                    if st.button("🔄 Regenerate", use_container_width=True):
                        st.rerun()
                
                with col3:
                    if st.button("ℹ️ Details", use_container_width=True):
                        st.info(f"Size: 1200×675px\nFormat: {output_format}\nStyle: MagicBricks")
                
                # Additional info
                with st.expander("📊 Image Information"):
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.markdown("**📐 Dimensions:** 1200 × 675 pixels")
                        st.markdown(f"**📁 Format:** {output_format}")
                        st.markdown("**🎨 Style:** MagicBricks Theme")
                    with info_col2:
                        st.markdown("**🎯 Optimized for:** Social Media")
                        st.markdown("**📱 Aspect Ratio:** 16:9")
                        st.markdown("**🌟 Quality:** Professional")
                
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
