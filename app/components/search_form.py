"""
Search Form Component

Accessible search form using st.form for batch submission,
preventing page reloads on each input change.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from config import search_config, gee_config


@dataclass
class SearchParameters:
    """Container for search form parameters."""
    
    query: str = ""
    search_type: str = "text"  # "text" or "image"
    reference_image: Optional[bytes] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    top_k: int = 10
    similarity_threshold: float = 0.3
    resolution: float = 10.0
    submitted: bool = False


def render_search_form(key_prefix: str = "search") -> SearchParameters:
    """
    Render the accessible search form.
    
    Uses st.form to batch all inputs and submit at once,
    preventing focus loss on each change (WCAG 2.4.3).
    
    Args:
        key_prefix: Prefix for form element keys.
        
    Returns:
        SearchParameters with user inputs.
    """
    params = SearchParameters()
    
    st.markdown("### 🔍 Search Parameters")
        
    # Search type selector (Must be outside form to trigger rerun)
    search_type = st.radio(
        "Search Method",
        options=["Text Query", "Reference Image"],
        horizontal=True,
        help="Search by describing what you're looking for, or upload a reference image."
    )
    params.search_type = "text" if search_type == "Text Query" else "image"
    
    # Text input OUTSIDE the form to prevent Enter key submission
    # The form will only submit when clicking the button
    if params.search_type == "text":
        params.query = st.text_input(
            "Search Query *",
            placeholder="e.g., solar panels, deforestation, circular irrigation",
            help="Describe the features you want to find. Press the Search button below to start."
        )
    else:
        uploaded_file = st.file_uploader(
            "Reference Image *",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            help="Upload a satellite image to find similar areas."
        )
        if uploaded_file:
            params.reference_image = uploaded_file.read()
    
    with st.form(key=f"{key_prefix}_form"):
        # st.markdown("### 🔍 Search Parameters") # Moved title up
        
        # Show what the user entered (read-only feedback)
        if params.search_type == "text" and params.query:
            st.info(f"🔍 Query: **{params.query}**")
        
        st.divider()
        
        # Date range
        col1, col2 = st.columns(2)
        
        with col1:
            default_start = datetime.now() - timedelta(days=gee_config.default_days_back)
            params.start_date = st.date_input(
                "Start Date",
                value=default_start,
                help="Beginning of the date range for imagery."
            )
        
        with col2:
            params.end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="End of the date range for imagery."
            )
        
        st.divider()
        
        # Advanced options in expander
        with st.expander("⚙️ Advanced Options"):
            params.top_k = st.slider(
                "Number of Results",
                min_value=1,
                max_value=50,
                value=search_config.top_k,
                help="Maximum number of matching tiles to return."
            )
            
            params.similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.001,
                format="%.3f",
                help="Minimum similarity score (0 to 1). Adjust to filter results."
            )
            
            st.markdown("#### 🔍 Search Resolution")
            params.resolution = st.slider(
                "Resolution (meters/pixel)",
                min_value=10.0,
                max_value=60.0,
                value=10.0,
                step=10.0,
                help="Resolution in meters per pixel. 10m is standard Sentinel-2 (High Detail). Higher values (e.g. 20m, 60m) are faster but less detailed."
            )
        
        st.divider()
        
        # Submit button
        submitted = st.form_submit_button(
            "🚀 Search",
            use_container_width=True,
            type="primary"
        )
        
        params.submitted = submitted
    
    return params


def validate_search_params(params: SearchParameters) -> Tuple[bool, str]:
    """
    Validate search parameters.
    
    Args:
        params: SearchParameters to validate.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    if params.search_type == "text":
        if not params.query or not params.query.strip():
            return False, "Please enter a search query."
    else:
        if params.reference_image is None:
            return False, "Please upload a reference image."
    
    if params.start_date and params.end_date:
        if params.start_date > params.end_date:
            return False, "Start date must be before end date."
    
    return True, ""
