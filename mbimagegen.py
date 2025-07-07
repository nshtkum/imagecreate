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
    page_title="Featured Image Generator", 
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
# ENHANCED TEXT OVERLAY FUNCTION
# ----------------------------
def create_modern_overlay(img, title, subtext, design_style="slanted", text_position="top-left"):
    """Create a modern, professional overlay with various design options"""
    
    # Ensure image is in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to standard social media banner size
    img = img.resize((1200, 630), Image.Resampling.LANCZOS)
    
    # Create overlay layer
    overlay = Image.new("RGBA", (1200, 630), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Load fonts with fallback
    try:
        font_title = ImageFont.truetype("arial.ttf", 64)
        font_sub = ImageFont.truetype("arial.ttf", 36)
    except:
        try:
            font_title = ImageFont.load_default()
            font_sub = ImageFont.load_default()
        except:
            font_title = ImageFont.load_default()
            font_sub = ImageFont.load_default()
    
    # Calculate text dimensions
    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    subtext_bbox = draw.textbbox((0, 0), subtext, font=font_sub)
    
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]
    subtext_width = subtext_bbox[2] - subtext_bbox[0]
    subtext_height = subtext_bbox[3] - subtext_bbox[1]
    
    # Design style implementations
    if design_style == "slanted":
        # Create slanted background box
        padding = 30
        box_width = max(title_width, subtext_width) + padding * 2
        box_height = title_height + subtext_height + padding * 3
        
        if text_position == "top-left":
            # Slanted box coordinates
            points = [
                (20, 20),
                (box_width + 40, 20),
                (box_width + 20, box_height + 20),
                (0, box_height + 20)
            ]
        elif text_position == "bottom-right":
            x_start = 1200 - box_width - 40
            y_start = 630 - box_height - 20
            points = [
                (x_start, y_start),
                (x_start + box_width + 20, y_start),
                (x_start + box_width, y_start + box_height),
                (x_start - 20, y_start + box_height)
            ]
        else:  # center
            x_start = (1200 - box_width) // 2
            y_start = (630 - box_height) // 2
            points = [
                (x_start - 20, y_start),
                (x_start + box_width + 20, y_start),
                (x_start + box_width, y_start + box_height),
                (x_start - 40, y_start + box_height)
            ]
        
        # Draw slanted box with gradient effect
        draw.polygon(points, fill=(0, 0, 0, 180))
        
        # Add border
        draw.polygon(points, outline=(255, 255, 255, 100), width=2)
        
    elif design_style == "rounded":
        # Rounded rectangle background
        padding = 40
        box_width = max(title_width, subtext_width) + padding * 2
        box_height = title_height + subtext_height + padding * 3
        
        if text_position == "top-left":
            box_coords = [20, 20, box_width + 20, box_height + 20]
        elif text_position == "bottom-right":
            box_coords = [1200 - box_width - 20, 630 - box_height - 20, 1200 - 20, 630 - 20]
        else:  # center
            x_start = (1200 - box_width) // 2
            y_start = (630 - box_height) // 2
            box_coords = [x_start, y_start, x_start + box_width, y_start + box_height]
        
        # Draw rounded rectangle
        draw.rounded_rectangle(box_coords, radius=20, fill=(0, 0, 0, 160))
        draw.rounded_rectangle(box_coords, radius=20, outline=(255, 255, 255, 80), width=2)
        
    elif design_style == "gradient":
        # Create gradient overlay
        gradient_height = title_height + subtext_height + 100
        
        if text_position == "top-left":
            for y in range(gradient_height):
                opacity = int(200 * (1 - y / gradient_height))
                draw.rectangle([0, y, 1200, y + 1], fill=(0, 0, 0, opacity))
        elif text_position == "bottom-right":
            start_y = 630 - gradient_height
            for y in range(gradient_height):
                opacity = int(200 * (y / gradient_height))
                draw.rectangle([0, start_y + y, 1200, start_y + y + 1], fill=(0, 0, 0, opacity))
    
    # Calculate text positions
    if text_position == "top-left":
        title_pos = (50, 50)
        subtext_pos = (50, 50 + title_height + 20)
    elif text_position == "bottom-right":
        title_pos = (1200 - title_width - 50, 630 - title_height - subtext_height - 70)
        subtext_pos = (1200 - subtext_width - 50, 630 - subtext_height - 30)
    else:  # center
        title_pos = ((1200 - title_width) // 2, (630 - title_height - subtext_height) // 2)
        subtext_pos = ((1200 - subtext_width) // 2, (630 - subtext_height) // 2 + title_height + 20)
    
    # Draw text with shadow/glow effect
    def draw_text_with_shadow(draw_obj, position, text, font, shadow_color=(0, 0, 0, 255), text_color=(255, 255, 255, 255)):
        x, y = position
        # Shadow
        for offset in [(2, 2), (-2, -2), (-2, 2), (2, -2), (0, 2), (2, 0), (-2, 0), (0, -2)]:
            draw_obj.text((x + offset[0], y + offset[1]), text, font=font, fill=shadow_color)
        # Main text
        draw_obj.text(position, text, font=font, fill=text_color)
    
    # Draw texts
    draw_text_with_shadow(draw, title_pos, title, font_title)
    draw_text_with_shadow(draw, subtext_pos, subtext, font_sub)
    
    # Composite the overlay onto the image
    final_img = Image.alpha_composite(img.convert("RGBA"), overlay)
    
    # Enhance the final image
    enhancer = ImageEnhance.Contrast(final_img)
    final_img = enhancer.enhance(1.1)
    
    enhancer = ImageEnhance.Sharpness(final_img)
    final_img = enhancer.enhance(1.1)
    
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
        enhanced_prompt = f"{prompt}, high quality, professional photography, 4k, detailed, realistic"
        
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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🏠 Magicbricks Featured Image Generator</h1><p>Create stunning property images with professional overlays</p></div>', unsafe_allow_html=True)

# Sidebar for advanced options
with st.sidebar:
    st.header("🎨 Design Options")
    
    design_style = st.selectbox(
        "Overlay Style",
        ["slanted", "rounded", "gradient"],
        index=0,
        help="Choose the style of your text overlay"
    )
    
    text_position = st.selectbox(
        "Text Position",
        ["top-left", "bottom-right", "center"],
        index=0,
        help="Where to place the text on the image"
    )
    
    st.header("📏 Image Settings")
    output_format = st.selectbox(
        "Output Format",
        ["PNG", "JPEG"],
        index=0,
        help="Choose the output format"
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Image source selection
    image_source = st.radio(
        "🖼️ Choose Image Source",
        ["Generate with AI", "Upload Your Own"],
        horizontal=True
    )
    
    if image_source == "Generate with AI":
        st.markdown('<div class="feature-box">🤖 <strong>AI Image Generation</strong><br>Describe your property and let AI create a professional image</div>', unsafe_allow_html=True)
        
        prompt = st.text_area(
            "📝 Describe your property",
            value="modern Indian apartment in Bangalore, garden view, bright daylight, professional photography",
            height=100,
            help="Be specific about the property type, location, and desired ambiance"
        )
        
        # Prompt suggestions
        st.markdown("**💡 Prompt Suggestions:**")
        suggestions = [
            "luxury villa with swimming pool, Mumbai, sunset lighting",
            "modern office space, glass windows, city view, professional",
            "cozy 2BHK apartment, balcony garden, natural lighting",
            "commercial shop, busy street, evening lights, attractive storefront"
        ]
        
        selected_suggestion = st.selectbox(
            "Quick prompts",
            [""] + suggestions,
            help="Select a pre-made prompt or write your own"
        )
        
        if selected_suggestion:
            prompt = selected_suggestion
    
    else:
        st.markdown('<div class="feature-box">📤 <strong>Upload Your Image</strong><br>Upload a high-quality image of your property</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear, high-resolution image for best results"
        )

with col2:
    st.markdown("### 📋 Text Overlay")
    
    title = st.text_input(
        "🏷️ Main Title",
        value="2BHK Flat in Bangalore",
        help="Property type and location"
    )
    
    subtext = st.text_input(
        "💰 Subtext",
        value="₹85 Lakh | Ready to Move",
        help="Price, status, or additional details"
    )
    
    # Preview text styling
    st.markdown("**Preview:**")
    st.markdown(f"**{title}**")
    st.markdown(f"*{subtext}*")

# Generation button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Generate Featured Image", type="primary", use_container_width=True):
        # Validation
        if not title.strip():
            st.error("Please enter a title for your image")
            st.stop()
        
        if image_source == "Generate with AI":
            if not prompt.strip():
                st.error("Please enter a description for AI generation")
                st.stop()
            
            if not client:
                st.error("AI generation is not available. Please check your HF_TOKEN in secrets.")
                st.stop()
        
        elif image_source == "Upload Your Own":
            if not uploaded_file:
                st.error("Please upload an image file")
                st.stop()
        
        # Generate or process image
        with st.spinner("Creating your featured image..."):
            try:
                if image_source == "Generate with AI":
                    base_image = generate_ai_image(prompt)
                else:
                    base_image = Image.open(uploaded_file)
                
                # Create the overlay
                final_image = create_modern_overlay(
                    base_image, 
                    title, 
                    subtext, 
                    design_style, 
                    text_position
                )
                
                # Display result
                st.markdown('<div class="success-box">✅ <strong>Success!</strong> Your featured image is ready</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(final_image, caption="Your Featured Image", use_container_width=True)
                
                with col2:
                    # Save image to buffer
                    buf = io.BytesIO()
                    final_image.save(buf, format=output_format, quality=95)
                    byte_data = buf.getvalue()
                    
                    # Download button
                    st.download_button(
                        label=f"⬇️ Download {output_format}",
                        data=byte_data,
                        file_name=f"featured_image_{title.replace(' ', '_').lower()}.{output_format.lower()}",
                        mime=f"image/{output_format.lower()}",
                        use_container_width=True
                    )
                    
                    # Image info
                    st.markdown("**📊 Image Details:**")
                    st.markdown(f"• Size: 1200 × 630 px")
                    st.markdown(f"• Format: {output_format}")
                    st.markdown(f"• Style: {design_style.title()}")
                    st.markdown(f"• Position: {text_position.replace('-', ' ').title()}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Try uploading your own image or check your internet connection for AI generation.")

# Footer with tips
st.markdown("---")
st.markdown("### 💡 Pro Tips")
tips_col1, tips_col2 = st.columns(2)

with tips_col1:
    st.markdown("""
    **For AI Generation:**
    • Be specific about property type and location
    • Mention lighting conditions (daylight, sunset, etc.)
    • Include architectural style (modern, traditional, etc.)
    • Add ambiance keywords (luxury, cozy, spacious)
    """)

with tips_col2:
    st.markdown("""
    **For Best Results:**
    • Use high-resolution images (min 800×600)
    • Keep text concise and impactful
    • Choose contrasting colors for readability
    • Test different overlay styles
    """)

st.markdown("---")
st.markdown("*Made with ❤️ for Magicbricks - Creating professional property images made easy*")
