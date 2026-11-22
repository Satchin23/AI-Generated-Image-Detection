"""
AI-Generated Image Detector
Grad-CAM Enabled Detection of AI-Generated Images Using Spatio-Frequency Features.
Designed for Academic AI Mini-Project Viva Examination & Live Demonstration.
Main Application Entry Point with Top Horizontal Navigation.
"""

import os
import sys
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from ui.styles import get_forensic_css
from ui.pipeline import init_session_state
from ui.components import render_top_navbar

# Import Modular Views
from views.home import render_home_view
from views.detect import render_detect_view
from views.analysis import render_analysis_view
from views.performance import render_performance_view
from views.reports import render_reports_view
from views.about import render_about_view
from views.quadrant_view import render_quadrant_view
from views.celebrity_analysis import render_celebrity_analysis_view

# -----------------------------------------------------------------------------
# Global Page Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Generated Image Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject Forensic Cyber-Lab Dark CSS Theme & Hide Sidebar
st.markdown(get_forensic_css(), unsafe_allow_html=True)

# Initialize Session State
init_session_state()

# -----------------------------------------------------------------------------
# Top Horizontal Navigation Shell
# -----------------------------------------------------------------------------
current_page = st.session_state.get("current_page", "Home")
render_top_navbar(current_page)

st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Page View Router
# -----------------------------------------------------------------------------
page = st.session_state.get("current_page", "Home")

if page == "Home":
    render_home_view()
elif page == "Detect Image":
    render_detect_view()
elif page == "Quadrant Split" or page == "Quadrant Forensics":
    render_quadrant_view()
elif page == "Analysis":
    render_analysis_view()
elif page == "Performance":
    render_performance_view()
elif page == "History & Reports":
    render_reports_view()
elif page == "About":
    render_about_view()
elif page == "Segment Analysis":
    render_celebrity_analysis_view()
else:
    render_home_view()
