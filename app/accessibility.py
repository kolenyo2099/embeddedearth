"""
Accessibility Helpers Module

Provides utilities for WCAG 2.1 compliance in Streamlit,
including focus management and screen reader support.
"""

import streamlit as st
from typing import Optional


def inject_accessibility_css():
    """
    Inject CSS for accessibility improvements.
    
    Adds:
    - Focus indicators
    - Skip navigation links
    - Improved contrast
    """
    st.markdown("""
    <style>
    /* Focus indicators */
    *:focus {
        outline: 3px solid #4A90A4 !important;
        outline-offset: 2px !important;
    }
    
    /* Skip to content link - Removed */
    
    /* Improved button contrast */
    .stButton > button {
        min-height: 44px;  /* Touch target size */
        font-size: 16px;
    }
    
    /* Form field labels */
    .stTextInput label,
    .stSelectbox label,
    .stSlider label {
        font-weight: 600;
        color: #333;
    }
    
    /* Result cards */
    .result-card {
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 12px;
        background: #fff;
    }
    
    .result-card:focus-within {
        border-color: #003262;
    }
    
    /* High contrast mode support */
    @media (prefers-contrast: high) {
        .stButton > button {
            border: 3px solid black;
        }
        
        .result-card {
            border-width: 3px;
        }
    }
    
    /* Reduced motion support */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation: none !important;
            transition: none !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def add_skip_link(target_id: str = "main-content"):
    """
    Deprecated: Skip link removed per user request.
    """
    pass


def announce_to_screen_reader(message: str):
    """
    Announce a message to screen readers.
    
    Uses ARIA live region for dynamic updates.
    
    Args:
        message: Message to announce.
    """
    st.markdown(f"""
    <div aria-live="polite" aria-atomic="true" class="sr-only" 
         style="position: absolute; left: -10000px; width: 1px; height: 1px; overflow: hidden;">
        {message}
    </div>
    """, unsafe_allow_html=True)


def create_accessible_image(
    image_data,
    alt_text: str,
    caption: Optional[str] = None
):
    """
    Display an image with proper accessibility attributes.
    
    Args:
        image_data: Image data (file path, URL, or array).
        alt_text: Descriptive alt text for screen readers.
        caption: Optional visible caption.
    """
    # Use Streamlit's image with caption
    st.image(image_data, caption=caption or alt_text, use_container_width=True)
    
    # Add hidden description for complex images
    st.markdown(f"""
    <div class="sr-only" style="position: absolute; left: -10000px;">
        Image description: {alt_text}
    </div>
    """, unsafe_allow_html=True)


def create_form_field(
    label: str,
    field_type: str = "text",
    help_text: Optional[str] = None,
    required: bool = False,
    **kwargs
):
    """
    Create an accessible form field.
    
    Args:
        label: Field label.
        field_type: Type of input (text, number, date, etc.).
        help_text: Additional help text.
        required: Whether field is required.
        **kwargs: Additional arguments for the field.
        
    Returns:
        Field value.
    """
    # Add required indicator to label
    if required:
        display_label = f"{label} *"
    else:
        display_label = label
    
    # Create field based on type
    if field_type == "text":
        value = st.text_input(display_label, help=help_text, **kwargs)
    elif field_type == "number":
        value = st.number_input(display_label, help=help_text, **kwargs)
    elif field_type == "date":
        value = st.date_input(display_label, help=help_text, **kwargs)
    elif field_type == "select":
        value = st.selectbox(display_label, help=help_text, **kwargs)
    elif field_type == "slider":
        value = st.slider(display_label, help=help_text, **kwargs)
    elif field_type == "file":
        value = st.file_uploader(display_label, help=help_text, **kwargs)
    else:
        value = st.text_input(display_label, help=help_text, **kwargs)
    
    return value


def validate_form(required_fields: dict) -> tuple:
    """
    Validate required form fields.
    
    Args:
        required_fields: Dict of {field_name: field_value}.
        
    Returns:
        Tuple of (is_valid, error_messages).
    """
    errors = []
    
    for name, value in required_fields.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"{name} is required")
    
    return len(errors) == 0, errors
