"""
Area Panel Component

Sidebar UI for Phase A of the two-phase workflow: load (fetch + tile +
embed) an AOI once, then reuse it for as many Phase B searches as you like.
Three sections: the load form, the current-area status, and the saved-areas
list (load from disk — no GEE required — or delete).
"""

import os
from datetime import datetime, timedelta

import streamlit as st

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from config import gee_config
from pipeline.area_store import (
    AreaParams,
    delete_area,
    list_saved_areas,
    load_area_from_disk,
    save_area,
)
from pipeline.ingest import MODEL_FAMILIES, embed_area_with_model, load_area

_CHIP_OPTIONS = {
    "Broad – ~3.8km/chip (default, fast)": 384,
    "Narrow – ~1.9km/chip (2× more tiles)": 192,
    "Precise – ~960m/chip (4× more tiles)": 96,
    "Ultra – ~480m/chip (8× more tiles)": 48,
}


def _model_short_label(model: str) -> str:
    """First token of the family label, e.g. 'DOFA-CLIP'."""
    return MODEL_FAMILIES.get(model, {}).get("label", model).split(" — ")[0]


def _area_status_line(meta_or_params: dict, num_tiles: int, name: str) -> str:
    sensor = meta_or_params.get("sensor")
    resolution = meta_or_params.get("resolution")
    start_date = meta_or_params.get("start_date")
    end_date = meta_or_params.get("end_date")
    model = meta_or_params.get("model", "dofa")
    return (
        f"**{name}** — {num_tiles} tiles · {_model_short_label(model)} · {sensor} · "
        f"{resolution:g} m · {start_date} to {end_date}"
    )


def _do_load_and_embed(params: AreaParams, name: str, hf_token: str = None) -> bool:
    """Runs load_area + embed (with the chosen model) with a progress bar."""
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def progress_cb(msg, frac=None):
        status_text.text(msg)
        if frac is not None:
            progress_bar.progress(min(1.0, max(0.0, frac)))

    try:
        area = load_area(params, progress_cb=progress_cb, name=name)

        embed_area_with_model(area, params.model, hf_token=hf_token, progress_cb=progress_cb)

        status_text.text("💾 Saving area to disk...")
        save_area(area)

        st.session_state.current_area = area
        st.session_state.last_similarities = None

        status_text.empty()
        progress_bar.empty()
        st.success(f"✅ Loaded and saved '{area.name}' — {area.num_tiles} tiles.")
        return True
    except Exception as e:
        status_text.empty()
        progress_bar.empty()
        st.error(f"Failed to load area: {e}")
        return False


def render_load_form():
    """Load-time parameters that identify a LoadedArea (removed from search_form)."""
    st.markdown("#### 📥 Load Area")

    aoi = st.session_state.get("aoi_geojson")

    # Model selection lives OUTSIDE the form so choosing DINOv3 can immediately
    # reveal its token field. The area is embedded with this one model, and the
    # search UI is then constrained to the matching tab.
    model = st.selectbox(
        "Embedding model",
        options=list(MODEL_FAMILIES.keys()),
        format_func=lambda k: MODEL_FAMILIES[k]["label"],
        help=(
            "Pick the model to embed this area with. Each model powers one kind "
            "of search: DOFA-CLIP → Semantic Search, DINOv3 → Zero-Shot "
            "Detection, CopernicusFM → Copernicus FM. You can load the same area "
            "again with a different model later."
        ),
    )

    hf_token = None
    if MODEL_FAMILIES[model]["needs_hf_token"]:
        env_token = os.environ.get("HF_TOKEN", "")
        hf_token = st.text_input(
            "Hugging Face token",
            value=env_token,
            type="password",
            help="Required to download the DINOv3 model (facebook/dinov3…).",
        )

    with st.form(key="area_load_form"):
        name = st.text_input(
            "Area name",
            value=f"Area {datetime.now().strftime('%Y-%m-%d')}",
            help="A label to recognize this area later in the saved-areas list."
        )

        sensor = st.radio(
            "Sensor",
            options=["Sentinel-2", "Sentinel-1"],
            horizontal=True,
            help="Choose between Optical (Sentinel-2) or Radar (Sentinel-1) imagery."
        )

        col1, col2 = st.columns(2)
        with col1:
            default_start = datetime.now() - timedelta(days=gee_config.default_days_back)
            start_date = st.date_input("Start Date", value=default_start)
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())

        resolution = st.slider(
            "Resolution (meters/pixel)",
            min_value=10.0,
            max_value=60.0,
            value=10.0,
            step=10.0,
            help="10m is standard Sentinel-2 (High Detail). Higher values are faster but less detailed."
        )

        tile_mode = st.selectbox(
            "Chip Coverage",
            options=list(_CHIP_OPTIONS.keys()),
            index=0,
            help=(
                "How much geographic area each tile sent to the model covers. "
                "Smaller chips are better for small/local features; larger chips "
                "preserve scene context and are faster."
            )
        )
        chip_size = _CHIP_OPTIONS[tile_mode]

        submitted = st.form_submit_button("📥 Load & Embed Area", use_container_width=True, type="primary")

    if submitted:
        if aoi is None:
            st.error("Please draw an area of interest on the map first.")
        elif sensor == "Sentinel-2" and start_date > end_date:
            st.error("Start date must be before end date.")
        else:
            params = AreaParams(
                aoi_geojson=aoi,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                sensor=sensor,
                resolution=resolution,
                chip_size=chip_size,
                model=model,
            )
            _do_load_and_embed(params, name, hf_token=hf_token)
            st.rerun()


def render_current_area_status():
    """Shows the loaded area currently held in session state, if any."""
    area = st.session_state.get("current_area")
    if area is None:
        return

    st.markdown("#### ✅ Current Area")
    st.markdown(_area_status_line(vars(area.params), area.num_tiles, area.name))

    if st.button("🗑️ Unload area", key="unload_current_area"):
        st.session_state.current_area = None
        st.session_state.last_similarities = None
        st.rerun()


def render_saved_areas_list():
    """Saved areas on disk — load without GEE, or delete."""
    st.markdown("#### 💾 Saved Areas")

    saved = list_saved_areas()
    if not saved:
        st.caption("No saved areas yet.")
        return

    current_area_id = getattr(st.session_state.get("current_area"), "area_id", None)

    for meta in saved:
        area_id = meta["area_id"]
        is_current = area_id == current_area_id

        with st.container():
            label = _area_status_line(meta["params"], meta.get("num_tiles", 0), meta["name"])
            if is_current:
                label = f"▶️ {label} (loaded)"
            st.markdown(label)

            col_load, col_delete = st.columns(2)
            with col_load:
                if st.button("Load", key=f"load_saved_{area_id}", disabled=is_current, use_container_width=True):
                    area = load_area_from_disk(area_id)
                    st.session_state.current_area = area
                    st.session_state.last_similarities = None
                    if area.tile_bounds:
                        lons = [b[0] for b in area.tile_bounds] + [b[2] for b in area.tile_bounds]
                        lats = [b[1] for b in area.tile_bounds] + [b[3] for b in area.tile_bounds]
                        st.session_state.map_center = [
                            (min(lats) + max(lats)) / 2,
                            (min(lons) + max(lons)) / 2,
                        ]
                    st.rerun()

            with col_delete:
                confirm_key = f"confirm_delete_{area_id}"
                if st.session_state.get(confirm_key):
                    if st.button("⚠️ Confirm delete", key=f"confirm_btn_{area_id}", use_container_width=True):
                        delete_area(area_id)
                        st.session_state.pop(confirm_key, None)
                        if is_current:
                            st.session_state.current_area = None
                        st.rerun()
                else:
                    if st.button("✕ Delete", key=f"delete_saved_{area_id}", use_container_width=True):
                        st.session_state[confirm_key] = True
                        st.rerun()

        st.divider()


def render_area_panel():
    """Renders the full area management panel (load form + status + saved list)."""
    render_load_form()
    st.markdown("---")
    render_current_area_status()
    st.markdown("---")
    render_saved_areas_list()
