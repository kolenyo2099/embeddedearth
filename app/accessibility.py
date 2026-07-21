"""
Accessibility Helpers Module

Provides utilities for WCAG 2.1 compliance in Streamlit,
including focus management and screen reader support.
"""

import streamlit as st


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
    /* Focus indicators (amber, matches the theme accent) */
    *:focus {
        outline: 2px solid #f59e0b !important;
        outline-offset: 2px !important;
    }

    /* Skip to content link - Removed */

    /* Improved button contrast */
    .stButton > button {
        min-height: 44px;  /* Touch target size */
        font-size: 16px;
    }

    /* Form field labels (inherit the dark-theme text color) */
    .stTextInput label,
    .stSelectbox label,
    .stSlider label {
        font-weight: 600;
    }

    /* Result cards */
    .result-card {
        border: 1px solid #2a2f3a;
        border-radius: 10px;
        padding: 12px;
        background: #181b22;
    }

    .result-card:focus-within {
        border-color: #f59e0b;
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
