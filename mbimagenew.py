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

# Image size configurations
IMAGE_SIZES = {
    "Social Media Banner (16:9)": (1200, 675),
    "Photostory Standard (4:3)": (1200, 900),
    "Mobile Webstory (9:16)": (720, 1280),
    "Instagram Square (1:1)": (1080, 1080),
    "Instagram Story (9:16)": (1080, 1920),
    "Facebook Post (16:9)": (1200, 630),
    "Twitter Header (3:1)": (1500, 500)
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

# ----------------------------
# AI IMAGE GENERATION FUNCTIONS
# ----------------------------
def generate_ai_image(prompt, output_size=(1200, 675)):
    """Generate image using OpenAI's image generation API"""
    if not client:
        raise Exception("OpenAI client not initialized")
    
    try:
        # Enhanced prompt based on output size
        width, height = output_size
        aspect_ratio = width / height
        
        if aspect_ratio > 1.5:  # Landscape
            orientation_hint = "wide angle, landscape orientation"
        elif aspect_ratio < 0.8:  # Portrait
            orientation_hint = "vertical composition, portrait orientation"
        else:  # Square
            orientation_hint = "square composition, centered"
        
        enhanced_prompt = f"{prompt}, {orientation_hint}, high quality, professional photography, 4k, detailed, realistic, architectural photography, no text, no words, no letters, clean image without any text overlay"
        
        # Generate image using OpenAI
        result = client.images.generate(
            model="dall-e-3",  # Use DALL-E 3 for better quality
            prompt=enhanced_prompt,
            n=1,
            size="1024x1024",
            quality="hd"  # High quality
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

def generate_image_with_uploaded_reference(uploaded_image, prompt):
    """Generate image using OpenAI's image generation API (fallback to text-only prompt)"""
    if not client:
        raise Exception("OpenAI client not initialized")
    
    try:
        # Note: OpenAI's current API doesn't support image-to-image directly
        # This is a fallback approach using enhanced text prompts
        
        # Analyze the uploaded image to create a descriptive prompt
        # For now, we'll use the user's prompt with additional context
        enhanced_prompt = f"""
        {prompt}. 
        Create a high quality, professional property image with realistic details, 
        good lighting, and architectural photography style. 
        The image should be suitable for real estate marketing.
        No text or words should appear in the image.
        Style: photorealistic, professional architecture photography, bright natural lighting.
        """
        
        # Generate image using standard OpenAI image generation
        result = client.images.generate(
            model="dall-e-3",  # Use DALL-E 3 for better quality
            prompt=enhanced_prompt,
            n=1,
            size="1024x1024",
            quality="hd"  # High quality
        )
        
        # Get the image URL (not base64 in newer API versions)
        if hasattr(result.data[0], 'url'):
            # Download image from URL
            import requests
            response = requests.get(result.data[0].url)
            image = Image.open(io.BytesIO(response.content))
        else:
            # Fallback to base64 if available
            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))
        
        # Ensure image is in the correct format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
            
    except Exception as e:
        raise Exception(f"Image generation with reference failed: {str(e)}")

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
    .size-info {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #ffc107;
        margin: 0.5rem 0;
        font-size: 0.9em;
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
    .new-feature {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #2196f3;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🏠 Magicbricks Image Generator for Content Team</h1></div>', unsafe_allow_html=True)

# Sidebar for options
with st.sidebar:
    st.header("🎨 Design Options")
    
    st.subheader("📏 Image Size")
    selected_size = st.selectbox(
        "Choose Output Size",
        list(IMAGE_SIZES.keys()),
        index=0,
        help="Select the size that best fits your platform"
    )
    
    # Display size info
    width, height = IMAGE_SIZES[selected_size]
    aspect_ratio = round(width/height, 2)
    st.markdown(f"""
    <div class="size-info">
        <strong>📐 Dimensions:</strong> {width} × {height}px<br>
        <strong>📊 Aspect Ratio:</strong> {aspect_ratio}:1<br>
        <strong>📱 Best for:</strong> {selected_size.split('(')[0].strip()}
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("💾 Output Settings")
    output_format = st.selectbox(
        "Image Format",
        ["PNG", "JPEG"],
        index=0
    )
    
    # Batch generation option
    st.subheader("🔄 Batch Generation")
    generate_multiple = st.checkbox(
        "Generate multiple sizes",
        help="Generate images for multiple platforms at once"
    )
    
    if generate_multiple:
        selected_sizes = st.multiselect(
            "Select sizes to generate",
            list(IMAGE_SIZES.keys()),
            default=[selected_size],
            help="Choose multiple sizes for batch generation"
        )

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    # Image source selection
    st.subheader("🖼️ Image Source")
    image_source = st.radio(
        "Choose how to get your property image:",
        ["🤖 Generate with AI", "📁 Upload Your Own Image", "🎨 AI Generation with Reference Image"],
        horizontal=False
    )
    
    if image_source == "🤖 Generate with AI":
        prompt = st.text_area(
            "📝 Describe your property",
            value="modern Indian apartment building exterior, bright daylight, professional architecture photography",
            height=100,
            help="Describe the property you want to generate"
        )
        
    elif image_source == "📁 Upload Your Own Image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a property image"
        )
        
    elif image_source == "🎨 AI Generation with Reference Image":
        st.markdown('<div class="new-feature"><strong>🆕 New Feature!</strong> Upload a reference image and AI will create a new property image inspired by your description</div>', unsafe_allow_html=True)
        st.info("📝 Note: The AI will use your text description to create a new image. The reference image helps you describe what you want!")
        
        reference_image = st.file_uploader(
            "Choose a reference image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a reference image to help describe your vision",
            key="reference_upload"
        )
        
        if reference_image:
            st.image(reference_image, caption="Reference Image", use_container_width=True)
        
        reference_prompt = st.text_area(
            "📝 Describe the property image you want to create",
            value="Create a modern luxury apartment building with glass facades, contemporary architecture, professional lighting, and landscaped surroundings",
            height=100,
            help="Describe the property image you want the AI to generate (the reference image is for inspiration)"
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
    if st.button("🚀 Generate Property Image(s)", type="primary", use_container_width=True):
        # Validation
        if not primary_text.strip():
            st.error("🚫 Please enter primary text")
            st.stop()
        
        if image_source == "🤖 Generate with AI":
            if not prompt.strip():
                st.error("🚫 Please describe the property for AI generation")
                st.stop()
            
            if not client:
                st.error("🚫 AI generation is not available. Please check your OpenAI API key.")
                st.stop()
                
        elif image_source == "📁 Upload Your Own Image":
            if not uploaded_file:
                st.error("🚫 Please upload an image file")
                st.stop()
                
        elif image_source == "🎨 AI Generation with Reference Image":
            if not reference_image:
                st.error("🚫 Please upload a reference image")
                st.stop()
            if not reference_prompt.strip():
                st.error("🚫 Please describe how to modify the reference image")
                st.stop()
            if not client:
                st.error("🚫 AI generation is not available. Please check your OpenAI API key.")
                st.stop()
        
        # Determine sizes to generate
        if generate_multiple and 'selected_sizes' in locals() and selected_sizes:
            sizes_to_generate = selected_sizes
        else:
            sizes_to_generate = [selected_size]
        
        # Generate the image(s)
        with st.spinner(f"🎨 Creating your property image{'s' if len(sizes_to_generate) > 1 else ''}..."):
            try:
                # Get base image
                if image_source == "🤖 Generate with AI":
                    # Use the largest size for AI generation to maintain quality
                    max_size = max(IMAGE_SIZES[size] for size in sizes_to_generate)
                    base_image = generate_ai_image(prompt, max_size)
                elif image_source == "📁 Upload Your Own Image":
                    base_image = Image.open(uploaded_file)
                elif image_source == "🎨 AI Generation with Reference Image":
                    reference_img = Image.open(reference_image)
                    base_image = generate_image_with_uploaded_reference(reference_img, reference_prompt)
                
                # Generate images for each selected size
                generated_images = {}
                
                for size_name in sizes_to_generate:
                    output_size = IMAGE_SIZES[size_name]
                    final_image = create_property_overlay(
                        base_image,
                        primary_text.strip(),
                        secondary_text.strip(),
                        output_size
                    )
                    generated_images[size_name] = final_image
                
                # Success message
                st.markdown('<div class="success-box">✅ Property image(s) generated successfully!</div>', unsafe_allow_html=True)
                
                # Display and download options for each generated image
                for size_name, final_image in generated_images.items():
                    st.subheader(f"📱 {size_name}")
                    
                    # Display the image
                    st.image(final_image, caption=f"Property Image - {size_name}", use_container_width=True)
                    
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
                            file_name=f"property_{size_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{primary_text.replace(' ', '_').lower()}.{output_format.lower()}",
                            mime=f"image/{output_format.lower()}",
                            use_container_width=True,
                            key=f"download_{size_name}"
                        )
                    
                    with col2:
                        if st.button("ℹ️ Details", use_container_width=True, key=f"details_{size_name}"):
                            width, height = IMAGE_SIZES[size_name]
                            st.info(f"Size: {width}×{height}px\nFormat: {output_format}\nAspect: {round(width/height, 2)}:1")
                    
                    # Separator between images
                    if len(generated_images) > 1:
                        st.markdown("---")
                
                # Batch download option
                if len(generated_images) > 1:
                    st.subheader("📦 Batch Download")
                    
                    # Create a zip file with all images
                    import zipfile
                    zip_buffer = io.BytesIO()
                    
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for size_name, final_image in generated_images.items():
                            img_buffer = io.BytesIO()
                            final_image.save(img_buffer, format=output_format, quality=95)
                            
                            filename = f"property_{size_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{primary_text.replace(' ', '_').lower()}.{output_format.lower()}"
                            zip_file.writestr(filename, img_buffer.getvalue())
                    
                    st.download_button(
                        label="📦 Download All Images (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"property_images_{primary_text.replace(' ', '_').lower()}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                
                # Additional info
                with st.expander("📊 Generation Summary"):
                    st.markdown(f"**🎯 Generated:** {len(generated_images)} image{'s' if len(generated_images) > 1 else ''}")
                    st.markdown(f"**📁 Format:** {output_format}")
                    
                    if image_source == "🤖 Generate with AI":
                        st.markdown(f"**🖼️ Source:** AI Generated")
                    elif image_source == "📁 Upload Your Own Image":
                        st.markdown(f"**🖼️ Source:** Uploaded Image")
                    elif image_source == "🎨 AI Generation with Reference Image":
                        st.markdown(f"**🖼️ Source:** AI Generated with Reference")
                    
                    if len(generated_images) > 1:
                        st.markdown("**📐 Sizes Generated:**")
                        for size_name in generated_images.keys():
                            width, height = IMAGE_SIZES[size_name]
                            st.markdown(f"• {size_name}: {width}×{height}px")
                
                # Quick regenerate button
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("🔄 Generate Again", use_container_width=True):
                        st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating image: {str(e)}")
                st.info("💡 Try with a different image or check your internet connection")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 1rem;">
    <p><strong>Magicbricks Image Generator for Content Team</strong></p>
    <p>Create professional property images in multiple sizes for all platforms</p>
    <p><em>Now with OpenAI-powered AI generation and reference image support!</em></p>
</div>
""", unsafe_allow_html=True)
