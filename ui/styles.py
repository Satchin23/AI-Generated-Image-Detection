"""
UI Styling and Cyber-Forensic Lab Theme for Streamlit.
Provides custom CSS, tactical fonts, cyber grid background, glowing indicators,
and forensic analyst aesthetics.
"""

def get_forensic_css() -> str:
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-main: #070a12;
        --bg-card: rgba(13, 19, 32, 0.88);
        --bg-card-hover: rgba(18, 26, 44, 0.95);
        --border-subtle: rgba(56, 189, 248, 0.15);
        --border-active: rgba(0, 240, 255, 0.5);
        --accent-cyan: #00f0ff;
        --accent-blue: #0284c7;
        --accent-red: #ff2d55;
        --accent-amber: #f59e0b;
        --accent-emerald: #10b981;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-mono: 'Inter', sans-serif;
    }

    /* Overall Application Background & Grid */
    .stApp {
        background-color: var(--bg-main);
        background-image: 
            linear-gradient(rgba(0, 240, 255, 0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 240, 255, 0.02) 1px, transparent 1px),
            radial-gradient(ellipse 80% 50% at 50% -20%, rgba(14, 165, 233, 0.12), transparent);
        background-size: 32px 32px, 32px 32px, 100% 100%;
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Scanline Overlay (subtle) */
    .stApp::before {
        content: " ";
        display: block;
        position: fixed;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.18) 50%);
        background-size: 100% 4px;
        z-index: 0;
        pointer-events: none;
        opacity: 0.45;
    }

    /* Header styling */
    .forensic-header {
        position: relative;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(8, 12, 22, 0.9));
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 8px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4), inset 0 0 15px rgba(0, 240, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    
    .forensic-header::before {
        content: "";
        position: absolute;
        top: -1px; left: 20px;
        width: 80px; height: 3px;
        background: var(--accent-cyan);
        box-shadow: 0 0 10px var(--accent-cyan);
    }

    .forensic-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1.65rem;
        letter-spacing: -0.02em;
        margin: 0;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }

    .forensic-subtitle {
        font-family: var(--text-mono);
        font-size: 0.8rem;
        color: var(--accent-cyan);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 0.35rem;
    }

    .forensic-specs {
        font-family: var(--text-mono);
        font-size: 0.75rem;
        color: var(--text-secondary);
        display: flex;
        flex-wrap: wrap;
        gap: 1.25rem;
        margin-top: 0.6rem;
        padding-top: 0.6rem;
        border-top: 1px dashed rgba(255, 255, 255, 0.1);
    }

    .forensic-spec-item {
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }

    .spec-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: var(--accent-cyan);
        box-shadow: 0 0 6px var(--accent-cyan);
    }

    /* Cards with tactical corners */
    .forensic-card {
        position: relative;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(8px);
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }

    .forensic-card:hover {
        border-color: var(--border-active);
        box-shadow: 0 6px 20px rgba(0, 240, 255, 0.08);
    }

    /* Corner accents */
    .corner-bracket {
        position: relative;
    }
    .corner-bracket::before {
        content: "";
        position: absolute;
        top: -1px; left: -1px;
        width: 10px; height: 10px;
        border-top: 2px solid var(--accent-cyan);
        border-left: 2px solid var(--accent-cyan);
        pointer-events: none;
    }
    .corner-bracket::after {
        content: "";
        position: absolute;
        bottom: -1px; right: -1px;
        width: 10px; height: 10px;
        border-bottom: 2px solid var(--accent-cyan);
        border-right: 2px solid var(--accent-cyan);
        pointer-events: none;
    }

    /* Verdict Banners & Badges */
    .badge-real {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(0, 240, 255, 0.1);
        color: #00f0ff;
        border: 1px solid rgba(0, 240, 255, 0.4);
        padding: 0.35rem 0.85rem;
        border-radius: 4px;
        font-family: var(--text-mono);
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.06em;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.15);
    }

    .badge-ai {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255, 45, 85, 0.12);
        color: #ff3366;
        border: 1px solid rgba(255, 45, 85, 0.45);
        padding: 0.35rem 0.85rem;
        border-radius: 4px;
        font-family: var(--text-mono);
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.06em;
        text-shadow: 0 0 10px rgba(255, 45, 85, 0.5);
        box-shadow: 0 0 15px rgba(255, 45, 85, 0.2);
    }

    /* Verdict Card Highlight */
    .verdict-box-real {
        background: linear-gradient(135deg, rgba(0, 240, 255, 0.08), rgba(13, 19, 32, 0.95));
        border: 1px solid rgba(0, 240, 255, 0.35);
        border-left: 4px solid #00f0ff;
        border-radius: 6px;
        padding: 1.25rem;
    }

    .verdict-box-ai {
        background: linear-gradient(135deg, rgba(255, 45, 85, 0.1), rgba(13, 19, 32, 0.95));
        border: 1px solid rgba(255, 45, 85, 0.4);
        border-left: 4px solid #ff2d55;
        border-radius: 6px;
        padding: 1.25rem;
    }

    /* Telemetry data readout */
    .telemetry-label {
        font-family: var(--text-mono);
        font-size: 0.72rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.25rem;
    }

    .telemetry-value {
        font-family: var(--text-mono);
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    /* Tactical radar spinner */
    .forensic-scanner {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2.5rem;
        background: rgba(11, 16, 28, 0.85);
        border: 1px solid rgba(0, 240, 255, 0.3);
        border-radius: 8px;
        text-align: center;
        margin: 1.5rem 0;
    }

    .radar-circle {
        position: relative;
        width: 70px;
        height: 70px;
        border-radius: 50%;
        border: 2px solid rgba(0, 240, 255, 0.3);
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.2);
        animation: radarPulse 2s infinite linear;
        margin-bottom: 1rem;
    }

    .radar-circle::after {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        border-radius: 50%;
        border: 2px solid transparent;
        border-top-color: var(--accent-cyan);
        animation: radarSpin 1.2s infinite linear;
    }

    @keyframes radarSpin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    @keyframes radarPulse {
        0% { transform: scale(0.95); opacity: 0.8; }
        50% { transform: scale(1.05); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.8; }
    }

    /* Fade-in result animation */
    @keyframes fadeInForensic {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-reveal {
        animation: fadeInForensic 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* Streamlit Components Restyling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(13, 19, 32, 0.95);
        padding: 0.4rem;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 2.5rem;
        padding: 0 1.2rem;
        background: transparent;
        border-radius: 6px;
        color: var(--text-secondary);
        font-family: var(--text-mono);
        font-size: 0.82rem;
        letter-spacing: 0.04em;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(0, 240, 255, 0.12) !important;
        color: #ffffff !important;
        border: 1px solid rgba(0, 240, 255, 0.35) !important;
        box-shadow: 0 0 12px rgba(0, 240, 255, 0.15);
    }

    /* Sidebar Restyling */
    section[data-testid="stSidebar"] {
        background-color: #05080e !important;
        border-right: 1px solid rgba(0, 240, 255, 0.12);
    }

    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }

    /* Streamlit Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.15), rgba(8, 12, 22, 0.85)) !important;
        border: 1px solid rgba(0, 240, 255, 0.35) !important;
        color: #e2e8f0 !important;
        font-family: var(--text-mono) !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        border-radius: 4px !important;
        padding: 0.45rem 1rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, rgba(0, 240, 255, 0.25), rgba(14, 165, 233, 0.2)) !important;
        border-color: var(--accent-cyan) !important;
        color: #ffffff !important;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.25) !important;
        transform: translateY(-1px);
    }

    /* Download button */
    div.stDownloadButton > button {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18), rgba(8, 12, 22, 0.85)) !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        color: #a7f3d0 !important;
        font-family: var(--text-mono) !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.05em !important;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.15) !important;
    }

    div.stDownloadButton > button:hover {
        border-color: #34d399 !important;
        color: #ffffff !important;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.3) !important;
    }

    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-family: var(--text-mono) !important;
        color: #ffffff !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-family: var(--text-mono) !important;
        font-size: 0.75rem !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
    }

    /* File Uploader */
    section[data-testid="stFileUploadDropzone"] {
        background: rgba(13, 19, 32, 0.7) !important;
        border: 1.5px dashed rgba(0, 240, 255, 0.25) !important;
        border-radius: 8px !important;
    }
    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: var(--accent-cyan) !important;
        background: rgba(13, 19, 32, 0.9) !important;
    }

    /* Custom Table for Batch and Logs */
    .forensic-table {
        width: 100%;
        border-collapse: collapse;
        font-family: var(--text-mono);
        font-size: 0.82rem;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        overflow: hidden;
    }

    .forensic-table th {
        background: rgba(18, 26, 44, 0.95);
        color: var(--accent-cyan);
        text-align: left;
        padding: 0.65rem 0.9rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        border-bottom: 1px solid rgba(0, 240, 255, 0.2);
    }

    .forensic-table td {
        padding: 0.6rem 0.9rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        color: #e2e8f0;
    }

    .forensic-table tr:hover {
        background: rgba(0, 240, 255, 0.04);
    }

    /* Monospace Code blocks */
    pre, code {
        font-family: var(--text-mono) !important;
        background: rgba(8, 12, 22, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Image display container */
    .image-plate-box {
        background: rgba(10, 15, 26, 0.9);
        border: 1px solid rgba(0, 240, 255, 0.18);
        border-radius: 6px;
        padding: 0.6rem;
        text-align: center;
        box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.5);
    }
    
    .image-plate-label {
        font-family: var(--text-mono);
        font-size: 0.74rem;
        color: var(--accent-cyan);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
    /* Completely remove the left-side navigation sidebar from main UI */
    [data-testid="stSidebar"], 
    section[data-testid="stSidebar"], 
    div[data-testid="collapsedControl"],
    button[data-testid="stSidebarCollapseButton"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        margin-left: -9999px !important;
    }

    /* Maximize container layout */
    .main .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1240px !important;
    }

    /* Top Navigation Bar Container */
    .top-nav-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(8, 12, 22, 0.92));
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 8px;
        padding: 0.65rem 1.25rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4), inset 0 0 15px rgba(0, 240, 255, 0.04);
    }

    /* Style Streamlit Pills for Top Navigation */
    div[data-testid="stPills"] {
        display: flex;
        justify-content: center;
        gap: 0.4rem;
        background: rgba(13, 19, 32, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 30px;
        padding: 0.25rem 0.6rem;
    }

    div[data-testid="stPills"] button {
        border-radius: 20px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.4rem 0.95rem !important;
        color: #94a3b8 !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stPills"] button:hover {
        color: #00f0ff !important;
        background: rgba(0, 240, 255, 0.08) !important;
        border-color: rgba(0, 240, 255, 0.3) !important;
    }

    div[data-testid="stPills"] button[aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(135deg, rgba(2, 132, 199, 0.85), rgba(14, 165, 233, 0.95)) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 14px rgba(14, 165, 233, 0.4) !important;
    }
    </style>
    """
