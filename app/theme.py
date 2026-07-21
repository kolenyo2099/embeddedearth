"""
EmbeddedEarth — visual theme ("Swiss data grid").

Editorial / Bauhaus identity: off-white paper, near-black ink, one vermilion
accent, zero border-radius, hard rules. `config.toml` sets the base light
palette so Streamlit themes its widgets natively; this module layers on the
grid structure, typography (Archivo + Space Mono), and the signature device:
numbered section headers with thick rules — legitimate here because the app's
workflow is a real sequence (01 select area -> 02 search -> 03 results) — plus
monospace coordinate/data readouts.

Exposed helpers:
- inject_theme_css(): global styles. Call once, right after set_page_config.
- render_hero(): the main-pane wordmark.
- render_section(number, title, note=None): a numbered section header.
"""

import streamlit as st

_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800;900&family=Space+Mono:wght@400;700&display=swap');

:root{
  --ee-bg:#FAFAF8;
  --ee-panel:#FFFFFF;
  --ee-panel-2:#F2F1EC;
  --ee-ink:#171714;
  --ee-muted:#6B6B63;
  --ee-rule:#171714;
  --ee-rule-soft:#D8D7CF;
  --ee-accent:#DA2818;
  --ee-mono:'Space Mono',ui-monospace,SFMono-Regular,monospace;
  --ee-sans:'Archivo',system-ui,-apple-system,'Helvetica Neue',sans-serif;
}

/* ---- Base ------------------------------------------------------------- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"], .stMarkdown{
  font-family:var(--ee-sans);
  color:var(--ee-ink);
}
[data-testid="stAppViewContainer"]{ background:var(--ee-bg); }

/* Reclaim vertical space; wide, gridded content column. */
[data-testid="stMainBlockContainer"], .block-container{
  padding-top:1.2rem !important;
  padding-bottom:4rem !important;
  max-width:1340px;
}
[data-testid="stHeader"]{ background:transparent; }

/* Kill rounded corners everywhere that matters — Swiss is sharp. */
.stButton > button, [data-baseweb="input"], [data-baseweb="textarea"],
[data-baseweb="select"] > div, [data-testid="stExpander"], [data-testid="stMetric"],
[data-testid="stAlert"], [data-testid="stAlertContainer"], [data-baseweb="tab-list"],
[data-baseweb="tab"], iframe, [data-testid="stIFrame"], .ee-readout,
[data-testid="stNotification"], img{
  border-radius:0 !important;
}

/* Headings: heavy grotesque, tight. */
h1,h2,h3,h4{ font-family:var(--ee-sans); font-weight:800; letter-spacing:-0.015em; color:var(--ee-ink); }

/* Dividers become hard ink rules. */
hr{ border:none !important; border-top:1.5px solid var(--ee-rule) !important; margin:1.1rem 0 !important; }

/* ---- Hero ------------------------------------------------------------- */
.ee-hero{ padding:2px 0 12px; }
.ee-eyebrow{
  font-family:var(--ee-mono); text-transform:uppercase; letter-spacing:0.16em;
  font-size:0.72rem; font-weight:700; color:var(--ee-accent);
}
.ee-title{
  font-family:var(--ee-sans); font-weight:900; font-size:clamp(2.4rem,5vw,3.4rem);
  line-height:0.96; letter-spacing:-0.03em; margin:0.28rem 0 0.4rem; color:var(--ee-ink);
}
.ee-sub{ color:var(--ee-muted); font-size:1.02rem; max-width:56ch; font-weight:500; }

/* ---- Numbered section header (the signature) -------------------------- */
.ee-section{
  display:flex; align-items:baseline; gap:16px;
  border-top:2.5px solid var(--ee-rule); padding-top:9px; margin:6px 0 2px;
}
.ee-section .ee-num{
  font-family:var(--ee-mono); font-weight:700; font-size:0.98rem; color:var(--ee-accent);
  letter-spacing:0.02em; flex:0 0 auto;
}
.ee-section .ee-label{
  font-family:var(--ee-sans); font-weight:800; text-transform:uppercase;
  letter-spacing:0.02em; font-size:1.32rem; line-height:1; color:var(--ee-ink);
}
.ee-section .ee-note{ margin-left:auto; font-family:var(--ee-mono); font-size:0.78rem; color:var(--ee-muted); align-self:center; }

/* ---- Telemetry readouts ----------------------------------------------- */
.ee-readout{
  font-family:var(--ee-mono); color:var(--ee-ink); background:var(--ee-panel);
  border:1.5px solid var(--ee-rule); padding:2px 9px; font-size:0.82rem; white-space:nowrap;
}
[data-testid="stMetric"]{
  background:var(--ee-panel); border:1.5px solid var(--ee-rule); padding:12px 14px;
}
[data-testid="stMetricValue"]{ font-family:var(--ee-mono); font-weight:700; color:var(--ee-ink); }
[data-testid="stMetricLabel"]{ text-transform:uppercase; letter-spacing:0.05em; font-size:0.72rem; color:var(--ee-muted); }
code, [data-testid="stMarkdownContainer"] code{
  font-family:var(--ee-mono); background:var(--ee-panel-2);
  color:var(--ee-accent); padding:1px 6px; font-size:0.85em; border:1px solid var(--ee-rule-soft);
}

/* ---- Sidebar = gridded control column --------------------------------- */
[data-testid="stSidebar"]{ background:var(--ee-panel-2); border-right:2px solid var(--ee-rule); }
[data-testid="stSidebar"] h1{ font-family:var(--ee-sans); font-weight:900; letter-spacing:-0.02em; font-size:1.5rem; }

/* ---- Buttons: hard rectangles ----------------------------------------- */
.stButton > button, [data-testid="stBaseButton-secondary"]{
  font-family:var(--ee-sans); font-weight:700; text-transform:uppercase; letter-spacing:0.03em;
  font-size:0.8rem; border:1.5px solid var(--ee-ink); background:var(--ee-panel); color:var(--ee-ink);
  transition:background .12s, color .12s;
}
.stButton > button:hover{ background:var(--ee-ink); color:var(--ee-bg); border-color:var(--ee-ink); }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"]{
  background:var(--ee-accent); color:#fff; border-color:var(--ee-accent);
}
.stButton > button[kind="primary"]:hover{ background:var(--ee-ink); border-color:var(--ee-ink); color:#fff; }

/* ---- Inputs ----------------------------------------------------------- */
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div{
  background:var(--ee-panel) !important; border:1.5px solid var(--ee-ink) !important;
}
.stTextInput input, .stNumberInput input, .stDateInput input, [data-baseweb="textarea"] textarea{ color:var(--ee-ink); }
.stTextInput label, .stSelectbox label, .stSlider label, .stDateInput label, .stNumberInput label, .stRadio label{
  text-transform:uppercase; letter-spacing:0.04em; font-size:0.74rem !important; font-weight:700 !important; color:var(--ee-muted) !important;
}

/* ---- Tabs: hard segmented control ------------------------------------- */
[data-baseweb="tab-list"]{ gap:0; border-bottom:2px solid var(--ee-rule); }
[data-baseweb="tab"]{
  font-family:var(--ee-sans); font-weight:700; text-transform:uppercase; letter-spacing:0.03em;
  font-size:0.82rem; color:var(--ee-muted); padding:10px 18px;
}
[data-baseweb="tab"]:hover{ color:var(--ee-ink); }
[data-baseweb="tab"][aria-selected="true"]{ color:var(--ee-ink); }
[data-baseweb="tab-highlight"]{ background:var(--ee-accent) !important; height:3px; }

/* ---- Expanders -------------------------------------------------------- */
[data-testid="stExpander"]{ border:1.5px solid var(--ee-ink); background:var(--ee-panel); overflow:hidden; }
[data-testid="stExpander"] summary{ font-weight:700; }
[data-testid="stExpander"] summary:hover{ color:var(--ee-accent); }

/* ---- Alerts: flat paper, hard rule down the left ---------------------- */
[data-testid="stAlert"], [data-testid="stAlertContainer"], [data-baseweb="notification"]{
  background:var(--ee-panel) !important; border:1.5px solid var(--ee-ink) !important;
  border-left:5px solid var(--ee-accent) !important; color:var(--ee-ink) !important;
}
[data-testid="stAlert"] p, [data-testid="stAlertContainer"] p{ color:var(--ee-ink) !important; }

/* ---- Sliders ---------------------------------------------------------- */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]{ background:var(--ee-accent) !important; border-radius:0 !important; }

/* ---- Map framed as a chart plate -------------------------------------- */
[data-testid="stIFrame"], iframe[title="streamlit_folium.st_folium"]{ border:1.5px solid var(--ee-ink); }
/* st_folium mis-sizes its iframe: a ~2970px height (map is 650) leaves a white
   void, and a hardcoded narrow width (from an early column measurement) leaves
   a gray tile band on the right. Pin height to the real map height and force
   full width; map_viewer's invalidateSize nudge then loads the tiles.
   Keep the height in sync with render_map_viewer(height=...). */
iframe[title="streamlit_folium.st_folium"]{ height:654px !important; width:100% !important; }

.stProgress > div > div > div > div{ background-color:transparent; }
</style>
"""


def inject_theme_css():
    """Inject global theme styles. Call once, right after set_page_config."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def render_hero():
    """Main-pane wordmark: eyebrow + title + one-line orientation."""
    st.markdown(
        '<div class="ee-hero">'
        '<div class="ee-eyebrow">Earth Observation / Semantic Search</div>'
        '<div class="ee-title">EmbeddedEarth</div>'
        '<div class="ee-sub">Draw an area, embed its imagery once, then search '
        'it in plain language — as many times as you like.</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_section(number: str, title: str, note: str | None = None):
    """A numbered section header (thick rule + mono number + heavy label).

    `number` like "01"; `note` is an optional right-aligned mono caption.
    """
    note_html = f'<span class="ee-note">{note}</span>' if note else ""
    st.markdown(
        f'<div class="ee-section"><span class="ee-num">{number}</span>'
        f'<span class="ee-label">{title}</span>{note_html}</div>',
        unsafe_allow_html=True,
    )
