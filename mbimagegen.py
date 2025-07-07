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
def create_modern_overlay(img, title, subtext, design_style="rectangle", text_position="top-left", 
                         overlay_color="Black", transparency=75, text_color="White", 
                         custom_color=None, custom_text_color=None, show_overlay=True):
    """Create a modern, professional overlay with various design options"""
    
    # Convert color selections to RGB values
    color_map = {
        "Black": (0, 0, 0),
        "Dark Blue": (25, 42, 86),
        "Dark Green": (21, 71, 52),
        "Dark Red": (139, 0, 0),
        "White": (255, 255, 255)
    }
    
    if overlay_color == "Custom" and custom_color:
        # Convert hex to RGB
        hex_color = custom_color.lstrip('#')
        bg_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    else:
        bg_color = color_map.get(overlay_color, (0, 0, 0))
    
    # Calculate alpha from transparency percentage
    alpha = int(255 * (transparency / 100))
    
    # Text color conversion
    text_color_map = {
        "White": (255, 255, 255, 255),
        "Black": (0, 0, 0, 255),
        "Yellow": (255, 255, 0, 255)
    }
    
    if text_color == "Custom" and custom_text_color:
        hex_text = custom_text_color.lstrip('#')
        final_text_color = tuple(int(hex_text[i:i+2], 16) for i in (0, 2, 4)) + (255,)
    else:
        final_text_color = text_color_map.get(text_color, (255, 255, 255, 255))
    
    # Ensure image is in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to standard social media banner size
    img = img.resize((1200, 630), Image.Resampling.LANCZOS)
    
    # Create overlay layer
    overlay = Image.new("RGBA", (1200, 630), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Smart font sizing - balance between visibility and fitting
    # Start with optimal sizes and adjust based on text length
    base_title_size = int(img.width * 0.08)  # ~96px for 1200px width
    base_sub_size = int(img.width * 0.045)   # ~54px for 1200px width
    
    # Adjust font size based on text length to ensure it fits
    title_length_factor = max(0.6, min(1.0, 30 / len(title))) if title else 1.0
    sub_length_factor = max(0.7, min(1.0, 40 / len(subtext))) if subtext else 1.0
    
    title_font_size = int(base_title_size * title_length_factor)
    sub_font_size = int(base_sub_size * sub_length_factor)
    
    # Ensure minimum readable sizes
    title_font_size = max(title_font_size, 48)  # Minimum 48px
    sub_font_size = max(sub_font_size, 28)      # Minimum 28px
    
    # Maximum sizes to prevent overflow
    title_font_size = min(title_font_size, int(img.width * 0.1))  # Max 10% of width
    sub_font_size = min(sub_font_size, int(img.width * 0.06))     # Max 6% of width
    
    # Load fonts with much larger sizes and better fallback
    try:
        font_title = ImageFont.truetype("arial.ttf", title_font_size)
        font_sub = ImageFont.truetype("arial.ttf", sub_font_size)
    except:
        try:
            # Try other common system fonts with large sizes
            font_title = ImageFont.truetype("calibri.ttf", title_font_size)
            font_sub = ImageFont.truetype("calibri.ttf", sub_font_size)
        except:
            try:
                # Try with different font names for cross-platform compatibility
                font_title = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", title_font_size)
                font_sub = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", sub_font_size)
            except:
                # Fallback: create a large default font
                try:
                    font_title = ImageFont.load_default().font_variant(size=title_font_size)
                    font_sub = ImageFont.load_default().font_variant(size=sub_font_size)
                except:
                    # Final fallback
                    font_title = ImageFont.load_default()
                    font_sub = ImageFont.load_default()
    
    # Calculate text dimensions with the adjusted font sizes
    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    subtext_bbox = draw.textbbox((0, 0), subtext, font=font_sub)
    
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]
    subtext_width = subtext_bbox[2] - subtext_bbox[0]
    subtext_height = subtext_bbox[3] - subtext_bbox[1]
    
    # Smart text wrapping if text is still too wide
    max_width = img.width * 0.8  # Use 80% of image width max
    
    if title_width > max_width:
        # Split title into two lines if too long
        words = title.split()
        if len(words) > 1:
            mid = len(words) // 2
            title_line1 = " ".join(words[:mid])
            title_line2 = " ".join(words[mid:])
            title = title_line1 + "\n" + title_line2
            # Recalculate dimensions
            title_bbox = draw.multiline_textbbox((0, 0), title, font=font_title)
            title_width = title_bbox[2] - title_bbox[0]
            title_height = title_bbox[3] - title_bbox[1]
    
    if subtext_width > max_width:
        # Split subtext if too long
        words = subtext.split()
        if len(words) > 1:
            mid = len(words) // 2
            sub_line1 = " ".join(words[:mid])
            sub_line2 = " ".join(words[mid:])
            subtext = sub_line1 + "\n" + sub_line2
            # Recalculate dimensions
            subtext_bbox = draw.multiline_textbbox((0, 0), subtext, font=font_sub)
            subtext_width = subtext_bbox[2] - subtext_bbox[0]
            subtext_height = subtext_bbox[3] - subtext_bbox[1]
    
    # Design style implementations - only if overlay is enabled
    if show_overlay:
        if design_style == "rectangle":
            # Create simple rectangle background box - no slant, no outline
            padding = int(img.width * 0.04)
            box_width = max(title_width, subtext_width) + padding * 3
            box_height = title_height + subtext_height + padding * 4
            
            # Ensure minimum box size for visibility
            min_box_width = int(img.width * 0.4)
            min_box_height = int(img.height * 0.25)
            box_width = max(box_width, min_box_width)
            box_height = max(box_height, min_box_height)
            
            if text_position == "top-left":
                # Simple rectangle coordinates
                box_coords = [30, 30, box_width + 30, box_height + 30]
            elif text_position == "bottom-right":
                box_coords = [img.width - box_width - 30, img.height - box_height - 30, img.width - 30, img.height - 30]
            else:  # center
                x_start = (img.width - box_width) // 2
                y_start = (img.height - box_height) // 2
                box_coords = [x_start, y_start, x_start + box_width, y_start + box_height]
            
            # Draw simple rectangle with custom color and transparency - NO OUTLINE
            draw.rectangle(box_coords, fill=(*bg_color, alpha))
            
        elif design_style == "rounded":
            # Rounded rectangle background - larger and more prominent
            padding = int(img.width * 0.05)
            box_width = max(title_width, subtext_width) + padding * 3
            box_height = title_height + subtext_height + padding * 4
            
            # Ensure minimum box size for visibility
            min_box_width = int(img.width * 0.4)
            min_box_height = int(img.height * 0.25)
            box_width = max(box_width, min_box_width)
            box_height = max(box_height, min_box_height)
            
            if text_position == "top-left":
                box_coords = [30, 30, box_width + 30, box_height + 30]
            elif text_position == "bottom-right":
                box_coords = [img.width - box_width - 30, img.height - box_height - 30, img.width - 30, img.height - 30]
            else:  # center
                x_start = (img.width - box_width) // 2
                y_start = (img.height - box_height) // 2
                box_coords = [x_start, y_start, x_start + box_width, y_start + box_height]
            
            # Draw rounded rectangle with custom color and transparency
            draw.rounded_rectangle(box_coords, radius=25, fill=(*bg_color, alpha))
            border_color = (255, 255, 255, 120) if sum(bg_color) < 400 else (0, 0, 0, 120)
            draw.rounded_rectangle(box_coords, radius=25, outline=border_color, width=3)
            
        elif design_style == "gradient":
            # Create stronger gradient overlay for better text visibility
            gradient_height = max(title_height + subtext_height + 150, int(img.height * 0.35))
            
            if text_position == "top-left":
                for y in range(gradient_height):
                    gradient_alpha = int(alpha * (1 - y / gradient_height))
                    draw.rectangle([0, y, img.width, y + 1], fill=(*bg_color, gradient_alpha))
            elif text_position == "bottom-right":
                start_y = img.height - gradient_height
                for y in range(gradient_height):
                    gradient_alpha = int(alpha * (y / gradient_height))
                    draw.rectangle([0, start_y + y, img.width, start_y + y + 1], fill=(*bg_color, gradient_alpha))
    
    # Calculate text positions with adjusted positioning - slightly more towards center from left
    text_padding = int(img.width * 0.04)
    line_spacing = int(title_font_size * 0.3)
    
    if text_position == "top-left":
        # Move text slightly more towards center and down from current position
        title_pos = (text_padding + 60, text_padding + 80)  # Moved right by 40px and down by 60px
        subtext_pos = (text_padding + 60, text_padding + 80 + title_height + line_spacing)
    elif text_position == "bottom-right":
        title_pos = (img.width - title_width - text_padding - 20, img.height - title_height - subtext_height - line_spacing - text_padding - 20)
        subtext_pos = (img.width - subtext_width - text_padding - 20, img.height - subtext_height - text_padding - 20)
    else:  # center
        title_pos = ((img.width - title_width) // 2, (img.height - title_height - subtext_height - line_spacing) // 2)
        subtext_pos = ((img.width - subtext_width) // 2, (img.height - subtext_height - line_spacing) // 2 + title_height + line_spacing)
    
    # Draw text with enhanced shadow/glow effect
    def draw_text_with_shadow(draw_obj, position, text, font, text_color=final_text_color):
        x, y = position
        shadow_offset = max(2, int(font.size * 0.03))
        
        # Smart shadow color based on text color
        if sum(text_color[:3]) > 400:  # Light text
            shadow_color = (0, 0, 0, 255)  # Dark shadow
        else:  # Dark text
            shadow_color = (255, 255, 255, 255)  # Light shadow
        
        # Multiple shadow layers
        for offset in range(shadow_offset, 0, -1):
            shadow_alpha = int(255 * (offset / shadow_offset) * 0.8)
            for dx, dy in [(offset, offset), (-offset, -offset), (-offset, offset), (offset, -offset), 
                          (0, offset), (offset, 0), (-offset, 0), (0, -offset)]:
                if '\n' in text:
                    draw_obj.multiline_text((x + dx, y + dy), text, font=font, fill=(*shadow_color[:3], shadow_alpha))
                else:
                    draw_obj.text((x + dx, y + dy), text, font=font, fill=(*shadow_color[:3], shadow_alpha))
        
        # Main text
        if '\n' in text:
            draw_obj.multiline_text(position, text, font=font, fill=text_color)
        else:
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

st.markdown('<div class="main-header"><h1>🏠 Property Image Generator</h1></div>', unsafe_allow_html=True)

# Sidebar for design options
with st.sidebar:
    st.header("🎨 Design Options")
    
    design_style = st.selectbox(
        "Overlay Style",
        ["rectangle", "rounded", "gradient"],
        index=0
    )
    
    # Toggle for overlay visibility
    show_overlay = st.checkbox("Show Background Overlay", value=True)
    
    text_position = st.selectbox(
        "Text Position",
        ["top-left", "bottom-right", "center"],
        index=0
    )
    
    # Color and transparency options
    st.subheader("Color & Transparency")
    
    overlay_color = st.selectbox(
        "Overlay Color",
        ["Black", "Dark Blue", "Dark Green", "Dark Red", "White", "Custom"],
        index=0
    )
    
    if overlay_color == "Custom":
        custom_color = st.color_picker("Pick Color", "#000000")
    
    transparency = st.slider(
        "Transparency",
        min_value=0,
        max_value=100,
        value=75,
        help="0 = Fully transparent, 100 = Fully opaque"
    )
    
    # Text color
    text_color = st.selectbox(
        "Text Color",
        ["White", "Black", "Yellow", "Custom"],
        index=0
    )
    
    if text_color == "Custom":
        custom_text_color = st.color_picker("Pick Text Color", "#FFFFFF")
    
    st.header("📏 Output")
    output_format = st.selectbox(
        "Format",
        ["PNG", "JPEG"],
        index=0
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
        prompt = st.text_area(
            "📝 Describe your property",
            value="modern Indian apartment in Bangalore, garden view, bright daylight",
            height=80
        )
    
    else:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg']
        )

with col2:
    st.markdown("### 📋 Text Content")
    
    title = st.text_input(
        "🏷️ Main Title",
        value="2BHK Flat in Bangalore"
    )
    
    subtext = st.text_input(
        "💰 Subtitle",
        value="₹85 Lakh | Ready to Move"
    )

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
                    text_position,
                    overlay_color,
                    transparency,
                    text_color,
                    custom_color if overlay_color == "Custom" else None,
                    custom_text_color if text_color == "Custom" else None,
                    show_overlay
                )
                
                # Display result
                st.markdown('<div class="success-box">✅ Image generated successfully!</div>', unsafe_allow_html=True)
                
                # Large image display - prioritize image viewing
                st.image(final_image, caption="Your Property Image", use_container_width=True)
                
                # Compact controls below the image
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
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
                
                with col2:
                    # Regenerate button
                    if st.button("🔄 Regenerate", use_container_width=True):
                        st.rerun()
                
                with col3:
                    # Info button
                    if st.button("ℹ️ Info", use_container_width=True):
                        st.info(f"Size: 1200×630px • Format: {output_format} • Style: {design_style.title()}")
                
                # Image details in an expander to save space
                with st.expander("📊 Image Details & Info"):
                    detail_col1, detail_col2 = st.columns(2)
                    with detail_col1:
                        st.markdown(f"**Size:** 1200 × 630 px")
                        st.markdown(f"**Format:** {output_format}")
                    with detail_col2:
                        st.markdown(f"**Style:** {design_style.title()}")
                        st.markdown(f"**Position:** {text_position.replace('-', ' ').title()}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Professional property image generator*")
