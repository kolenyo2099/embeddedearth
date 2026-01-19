import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import io
import torch

from models.dinov3 import DINOv3Wrapper
from pipeline.zero_shot_pipeline import run_zero_shot_pipeline
from app.components.search_form import validate_search_params # reuse date/aoi validation logic if possible

def render_zero_shot_form():
    """
    Render the Zero-Shot Object Detection sidebar form.
    Returns params object if search submitted, else None.
    """
    st.header("🎯 Zero-Shot Detection")
    st.caption("Find objects based on a visual example (Query Patch).")
    
    # 1. Hugging Face Token (Required for DINOv3)
    if 'hf_token' not in st.session_state:
        st.session_state.hf_token = ""
        
    # Check env var first
    import os
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        st.session_state.hf_token = env_token
    
    with st.expander("🔑 Model Access (Required)", expanded=not bool(st.session_state.hf_token)):
        token_input = st.text_input(
            "Hugging Face Token", 
            value=st.session_state.hf_token,
            type="password",
            help="Access token for downloading DINOv3 model (facebook/dinov3...)"
        )
        if token_input:
            st.session_state.hf_token = token_input
            
    if not st.session_state.hf_token:
        st.info("No Hugging Face Token provided. Will attempt to use cached model or public access.")
        # Do NOT return None here. Let it proceed.

    st.markdown("---")
    
    # 2. Reference Selection
    st.subheader("1. Select Reference Object")
    
    uploaded_file = st.file_uploader("Upload Reference Image", type=['png', 'jpg', 'jpeg', 'tif'])
    
    query_vector = None
    
    if uploaded_file is not None:
        # Load Image safely into memory
        bytes_data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        img_array = np.array(image)
        
        # Persist image execution to avoid GC issues
        # st.image(image, caption="Reference", use_column_width=True) # Debug check
        
        st.markdown("Draw a box around the object:")
        
        # Calculate canvas dimensions to fit sidebar
        aspect_ratio = img_array.shape[1] / img_array.shape[0]
        canvas_width = 280
        canvas_height = int(canvas_width / aspect_ratio)
        
        # Canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Orange translucent
            stroke_width=2,
            stroke_color="#FFA500",
            background_image=image,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="rect",
            key="canvas_zero_shot",
        )
        
        # Process Selection
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                # Get last drawn object
                obj = objects[-1]
                left = int(obj["left"] * (img_array.shape[1] / canvas_width))
                top = int(obj["top"] * (img_array.shape[0] / canvas_height))
                width = int(obj["width"] * (img_array.shape[1] / canvas_width))
                height = int(obj["height"] * (img_array.shape[0] / canvas_height))
                
                bbox = [left, top, width, height]
                
                # Extract Query Vector Button
                if st.button("Extract Features from Patch"):
                    with st.spinner("Extracting DINOv3 features..."):
                        try:
                            wrapper = DINOv3Wrapper(token=st.session_state.hf_token)
                            q_vec = wrapper.get_patch_embedding(img_array, bbox)
                            st.session_state.zero_shot_query = q_vec
                            st.success("✅ Query Vector Extracted!")
                        except Exception as e:
                            st.error(f"Extraction failed: {e}")
            else:
                st.info("Please draw a box around the target.")
    
    # Check if we have a query vector in session
    if 'zero_shot_query' in st.session_state:
        st.success("Reference Pattern Ready")
        query_vector = st.session_state.zero_shot_query
    else:
        # st.stop() # Stop here until reference is ready - CAUSES MAIN SCRIPT TO STOP
        return None

    st.markdown("---")
    
    # 3. Search Parameters
    st.subheader("2. Search Parameters")
    
    from datetime import datetime, timedelta
    
    sensor = st.selectbox("Sensor", ["Sentinel-2", "Sentinel-1"], index=0)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
        
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.65, 0.05)
    
    # 4. Run Button
    if st.button("🚀 Run Detection", type="primary"):
        return {
            "start_date": start_date,
            "end_date": end_date,
            "sensor": sensor,
            "threshold": threshold,
            "query_vector": query_vector,
            "token": st.session_state.hf_token,
            "submitted": True
        }
        
    return None
