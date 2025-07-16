import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import io
import base64
import numpy as np
import requests
from typing import Optional

# ----------------------------
# CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="Magicbricks Image Generator", 
    layout="centered",
    page_icon="🏠",
    initial_sidebar_state="collapsed"
)

# Simplified image size configurations - only landscape and portrait
IMAGE_SIZES = {
    "Landscape (16:9)": (1200, 675),
    "Portrait (9:16)": (675, 1200)
}

# Initialize OpenAI client with error handling
@st.cache_resource
def init_openai_client():
    try:
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            st.warning("⚠️ OPENAI_API_KEY not found in secrets. AI generation will be disabled.")
            st.info("💡 Please add your OpenAI API key in Streamlit secrets to enable AI image generation.")
            return None
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return None

client = init_openai_client()

def create_property_overlay(img, primary_text, secondary_text, output_size=(1200, 675)):
    """Create a property overlay with responsive design for different sizes"""
    
    # Ensure image is in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to target size
    img = img.resize(output_size, Image.Resampling.LANCZOS)
    width, height = output_size
    
    # Create overlay layer
    overlay = Image.new("RGBA", output_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Color scheme - vibrant red like in the image
    gradient_red = (231, 76, 60)
    gradient_dark = (192, 57, 43)
    white = (255, 255, 255)
    
    # Determine layout based on aspect ratio
    aspect_ratio = width / height
    
    if aspect_ratio > 1.5:  # Wide format (landscape)
        text_area_percentage = 0.30
        gradient_position = "bottom"
    elif aspect_ratio < 0.8:  # Tall format (portrait/mobile)
        text_area_percentage = 0.25
        gradient_position = "bottom"
    else:  # Square or near-square
        text_area_percentage = 0.25
        gradient_position = "bottom"
    
    # Calculate gradient dimensions
    if gradient_position == "bottom":
        gradient_height = int(height * text_area_percentage)
        gradient_start_y = height - gradient_height
        gradient_width = width
        gradient_start_x = 0
    
    # Create completely dissolved/seamless gradient
    for y in range(gradient_height):
        # Calculate gradient progression (0 to 1)
        progress = y / gradient_height
        
        # Ultra-smooth easing for completely dissolved effect
        eased_progress = progress * progress * progress * (progress * (progress * 6 - 15) + 10)
        
        # Very gradual alpha progression for dissolved effect
        alpha = int(15 + (200 * eased_progress))
        
        # Smooth color interpolation
        r = int(gradient_dark[0] + (gradient_red[0] - gradient_dark[0]) * eased_progress)
        g = int(gradient_dark[1] + (gradient_red[1] - gradient_dark[1]) * eased_progress)
        b = int(gradient_dark[2] + (gradient_red[2] - gradient_dark[2]) * eased_progress)
        
        # Draw ultra-smooth gradient line
        draw.rectangle([gradient_start_x, gradient_start_y + y, 
                       gradient_start_x + gradient_width, gradient_start_y + y + 1], 
                      fill=(r, g, b, alpha))
    
    # Add additional smoothing blur effect above gradient
    blur_height = max(20, int(height * 0.02))  # Responsive blur height
    for i in range(blur_height):
        progress = 1 - (i / blur_height)
        alpha = int(5 * progress)
        draw.rectangle([gradient_start_x, gradient_start_y - blur_height + i, 
                       gradient_start_x + gradient_width, gradient_start_y - blur_height + i + 1], 
                      fill=(gradient_dark[0], gradient_dark[1], gradient_dark[2], alpha))
    
    # Responsive font sizing based on image dimensions
    def get_responsive_font_size(base_size, width, height):
        scale_factor = min(width / 1200, height / 675)  # Base reference size
        return max(12, int(base_size * scale_factor))
    
    # Load fonts with marketing-style emphasis
    def get_marketing_font(size, bold=False):
        marketing_fonts = [
            "arialbd.ttf", "calibrib.ttf", "arial.ttf", "calibri.ttf",
            "/System/Library/Fonts/Arial Bold.ttf", "/System/Library/Fonts/Helvetica-Bold.ttf",
            "/System/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Helvetica.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        ]
        
        for font_path in marketing_fonts:
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
        
        return ImageFont.load_default()
    
    # Responsive font sizes
    primary_font_size = get_responsive_font_size(36, width, height)
    secondary_font_size = get_responsive_font_size(24, width, height)
    
    primary_font = get_marketing_font(primary_font_size, bold=True)
    secondary_font = get_marketing_font(secondary_font_size, bold=False)
    
    # Responsive text positioning
    text_padding = max(20, int(width * 0.025))
    text_start_y = gradient_start_y + (gradient_height * 0.6)
    
    # Helper function for marketing-style text with strong shadows
    def draw_marketing_text(text, position, font, text_color=white):
        x, y = position
        
        # Responsive shadow offsets
        shadow_scale = min(width / 1200, height / 675)
        shadow_offsets = [
            (int(2 * shadow_scale), int(2 * shadow_scale)),
            (int(1 * shadow_scale), int(1 * shadow_scale)),
            (int(3 * shadow_scale), int(3 * shadow_scale)),
            (int(4 * shadow_scale), int(4 * shadow_scale))
        ]
        
        for offset_x, offset_y in shadow_offsets:
            shadow_alpha = max(50, 150 - (offset_x * 30))
            draw.text((x + offset_x, y + offset_y), text, font=font, fill=(0, 0, 0, shadow_alpha))
        
        # Main text
        draw.text((x, y), text, font=font, fill=text_color)
    
    # Position text with word wrapping for smaller sizes
    def wrap_text(text, font, max_width):
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]
            
            if line_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)  # Single word too long
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    # Text positioning with responsive wrapping
    available_text_width = width - (text_padding * 2)
    
    if primary_text:
        # Wrap primary text if needed
        primary_lines = wrap_text(primary_text, primary_font, available_text_width)
        
        # Calculate total height needed
        primary_bbox = draw.textbbox((0, 0), primary_lines[0], font=primary_font)
        line_height = primary_bbox[3] - primary_bbox[1]
        total_primary_height = len(primary_lines) * line_height + (len(primary_lines) - 1) * 5
        
        # Position primary text
        current_y = text_start_y
        for line in primary_lines:
            bbox = draw.textbbox((0, 0), line, font=primary_font)
            line_width = bbox[2] - bbox[0]
            line_x = (width - line_width) // 2
            
            draw_marketing_text(line, (line_x, current_y), primary_font)
            current_y += line_height + 5
        
        if secondary_text:
            # Wrap secondary text if needed
            secondary_lines = wrap_text(secondary_text, secondary_font, available_text_width)
            
            # Position secondary text below primary
            current_y += 10  # Gap between primary and secondary
            
            for line in secondary_lines:
                bbox = draw.textbbox((0, 0), line, font=secondary_font)
                line_width = bbox[2] - bbox[0]
                line_x = (width - line_width) // 2
                
                draw_marketing_text(line, (line_x, current_y), secondary_font)
                
                secondary_bbox = draw.textbbox((0, 0), line, font=secondary_font)
                current_y += (secondary_bbox[3] - secondary_bbox[1]) + 5
    
    # Composite the overlay onto the image
    final_img = Image.alpha_composite(img.convert("RGBA"), overlay)
    
    # Apply subtle blur to the gradient area for smoother dissolution
    mask = Image.new("L", output_size, 0)
    mask_draw = ImageDraw.Draw(mask)
    
    # Create mask for gradient area only
    for y in range(gradient_height):
        alpha = int(255 * (y / gradient_height))
        mask_draw.rectangle([gradient_start_x, gradient_start_y + y, 
                           gradient_start_x + gradient_width, gradient_start_y + y + 1], 
                           fill=alpha)
    
    # Apply subtle blur to gradient area
    blur_radius = max(0.5, min(2.0, width / 2400))  # Responsive blur
    blurred = final_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    final_img = Image.composite(blurred, final_img, mask)
    
    # Enhance for marketing appeal
    enhancer = ImageEnhance.Contrast(final_img)
    final_img = enhancer.enhance(1.08)
    
    return final_img.convert("RGB")

def enhance_prompt_for_realism(user_prompt):
    """Enhance user prompt to generate realistic, natural property images"""
    
    # Keywords that indicate realistic style needed
    realism_enhancers = [
        "photorealistic",
        "natural lighting", 
        "shot with professional camera",
        "real estate photography",
        "architectural photography",
        "natural daylight",
        "authentic",
        "realistic proportions",
        "natural materials",
        "professional real estate photo",
        "high resolution",
        "sharp focus",
        "natural colors",
        "no filters",
        "documentary style"
    ]
    
    # Remove artificial/fantasy elements
    avoid_terms = [
        "artistic", "stylized", "fantasy", "dramatic lighting", 
        "oversaturated", "HDR effect", "painted", "illustration",
        "cartoon", "anime", "digital art", "concept art"
    ]
    
    # Clean the prompt from artificial terms
    clean_prompt = user_prompt
    for term in avoid_terms:
        clean_prompt = clean_prompt.replace(term, "")
    
    # Add realism enhancers
    enhanced_prompt = f"{clean_prompt}, photorealistic, natural lighting, professional real estate photography, shot with DSLR camera, natural daylight, authentic architectural details, realistic proportions, high resolution, sharp focus, natural colors, documentary photography style, no artificial effects"
    
    return enhanced_prompt

# ----------------------------
# AI IMAGE GENERATION FUNCTION
# ----------------------------
def generate_ai_image(prompt, output_size=(1200, 675)):
    """Generate realistic image using OpenAI's image generation API"""
    if not client:
        raise Exception("OpenAI client not initialized")
    
    try:
        # Enhance prompt for maximum realism
        enhanced_prompt = enhance_prompt_for_realism(prompt)
        
        # Add orientation hint based on output size
        width, height = output_size
        aspect_ratio = width / height
        
        if aspect_ratio > 1.5:  # Landscape
            orientation_hint = "wide angle view, landscape orientation"
        elif aspect_ratio < 0.8:  # Portrait
            orientation_hint = "vertical composition, portrait orientation"
        else:  # Square
            orientation_hint = "square composition, centered view"
        
        # Final prompt optimized for realism
        final_prompt = f"{enhanced_prompt}, {orientation_hint}, shot with professional DSLR camera, natural lighting, high resolution, photojournalism style, no text or graphics overlaid"
        
        # Generate image using OpenAI with highest quality settings
        result = client.images.generate(
            model="dall-e-3",
            prompt=final_prompt,
            n=1,
            size="1024x1024",
            quality="hd",  # High quality for realism
            style="natural"  # More realistic style
        )
        
        # Handle both URL and base64 responses
        if hasattr(result.data[0], 'url') and result.data[0].url:
            # Download image from URL
            import requests
            response = requests.get(result.data[0].url)
            image = Image.open(io.BytesIO(response.content))
        elif hasattr(result.data[0], 'b64_json') and result.data[0].b64_json:
            # Use base64 data
            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            raise Exception("No image data received from OpenAI")
        
        # Ensure image is in the correct format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        raise Exception(f"Image generation failed: {str(e)}")

# ----------------------------
# STREAMLIT UI
# ----------------------------

# Custom CSS for simple, clean styling
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

st.markdown('<div class="main-header"><h1>🏠 Magicbricks Image Generator</h1><p>Generate realistic property & interior images</p></div>', unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Simple image source selection
    st.subheader("🖼️ Create Image")
    image_source = st.radio(
        "Choose your method:",
        ["🤖 Generate with AI", "📁 Upload Your Own Image"],
        horizontal=True
    )
    
    if image_source == "🤖 Generate with AI":
        prompt = st.text_area(
            "📝 Describe what you want to create",
            value="modern living room with sofa and natural lighting",
            height=100,
            help="Examples: 'bedroom interior', 'apartment building exterior', 'kitchen with island', 'office space'"
        )
        
    else:  # Upload image
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload your property image"
        )

with col2:
    st.subheader("📐 Size & Text")
    
    # Simple size selection - only landscape and portrait
    selected_size = st.selectbox(
        "Image Size",
        list(IMAGE_SIZES.keys()),
        index=0,
        help="Choose landscape for wide images, portrait for tall images"
    )
    
    # Display size info
    width, height = IMAGE_SIZES[selected_size]
    st.info(f"📏 {width} × {height} pixels")
    
    # Optional text overlay
    st.markdown("**Text Overlay (Optional)**")
    primary_text = st.text_input(
        "Main text",
        value="",
        placeholder="e.g., 2BHK Apartment"
    )
    
    secondary_text = st.text_input(
        "Secondary text",
        value="",
        placeholder="e.g., ₹45 Lakh • Ready to Move"
    )

# Generation button
st.markdown("---")
if st.button("🚀 Generate Image", type="primary", use_container_width=True):
    # Validation
    if image_source == "🤖 Generate with AI":
        if not prompt.strip():
            st.error("🚫 Please describe what you want to create")
            st.stop()
        
        if not client:
            st.error("🚫 AI generation is not available. Please check your OpenAI API key.")
            st.stop()
            
    elif image_source == "📁 Upload Your Own Image":
        if not uploaded_file:
            st.error("🚫 Please upload an image file")
            st.stop()
    
    # Generate the image
    with st.spinner("🎨 Creating your image..."):
        try:
            # Get base image
            if image_source == "🤖 Generate with AI":
                output_size = IMAGE_SIZES[selected_size]
                base_image = generate_ai_image(prompt, output_size)
            else:
                base_image = Image.open(uploaded_file)
            
            # Apply text overlay if provided, otherwise just resize
            output_size = IMAGE_SIZES[selected_size]
            if primary_text.strip():
                final_image = create_property_overlay(
                    base_image,
                    primary_text.strip(),
                    secondary_text.strip(),
                    output_size
                )
            else:
                # Just resize the image without overlay
                final_image = base_image.resize(output_size, Image.Resampling.LANCZOS)
                if final_image.mode != 'RGB':
                    final_image = final_image.convert('RGB')
            
            # Success message
            st.markdown('<div class="success-box">✅ Image generated successfully!</div>', unsafe_allow_html=True)
            
            # Display the image
            st.image(final_image, caption=f"Generated Image - {selected_size}", use_container_width=True)
            
            # Download button
            buf = io.BytesIO()
            final_image.save(buf, format="PNG", quality=95)
            byte_data = buf.getvalue()
            
            filename_base = primary_text.replace(' ', '_').lower() if primary_text.strip() else "magicbricks_image"
            st.download_button(
                label="⬇️ Download Image",
                data=byte_data,
                file_name=f"{filename_base}_{selected_size.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png",
                mime="image/png",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ Error generating image: {str(e)}")
            st.info("💡 Try with a different description or check your internet connection")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 1rem;">
    <p><strong>Magicbricks Image Generator</strong></p>
    <p>Create realistic property images for listings, blogs, and marketing</p>
</div>
""", unsafe_allow_html=True)
