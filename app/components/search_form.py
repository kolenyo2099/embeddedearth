"""
Search Form Component

Accessible search form for query-time parameters only. Load-time parameters
(dates, sensor, resolution, chip coverage) live in the area panel now — they
identify a LoadedArea, not a search. top_k and similarity_threshold sit
outside the form: with cached embeddings a rerun costs milliseconds, so
these re-rank live against the cached similarity array.
"""

import streamlit as st
from typing import Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from config import search_config


@dataclass
class SearchParameters:
    """Container for search form parameters (query-time only)."""

    query: str = ""
    search_type: str = "text"  # "text" or "image"
    reference_image: Optional[bytes] = None
    top_k: int = 10
    similarity_threshold: float = 0.1  # matches search_config.similarity_threshold
    submitted: bool = False


def render_search_form(key_prefix: str = "search") -> SearchParameters:
    """
    Render the accessible search form.

    Search method/query/image are inside an st.form so they batch-submit;
    top_k and similarity_threshold live outside it so dragging them re-ranks
    already-cached results instantly instead of waiting for a submit click.

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

    # Live re-rank controls: outside the form so changes take effect
    # immediately against the cached similarity array (no re-encoding).
    params.top_k = st.slider(
        "Number of Results",
        min_value=1,
        max_value=50,
        value=search_config.top_k,
        key=f"{key_prefix}_top_k",
        help="Maximum number of matching tiles to return."
    )

    threshold_pct = st.slider(
        "Minimum Match Confidence (%)",
        min_value=0,
        max_value=100,
        value=int(search_config.similarity_threshold * 100) if search_config.similarity_threshold else 10,
        step=1,
        format="%d%%",
        key=f"{key_prefix}_threshold",
        help="Minimum similarity percentage. Note: For satellite AI, >15% is often a strong match."
    )
    params.similarity_threshold = threshold_pct / 100.0

    with st.form(key=f"{key_prefix}_form"):
        if params.search_type == "text" and params.query:
            st.info(f"🔍 Query: **{params.query}**")

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

    return True, ""
