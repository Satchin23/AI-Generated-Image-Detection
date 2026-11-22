"""
Forensic Confidence Gauge and Telemetry Meter.
Generates interactive, lightweight SVG radial meters and forensic telemetry bars
color-coded for REAL (Cyan/Emerald) vs AI-GENERATED (Amber/Crimson).
"""

import math

def render_confidence_gauge(label: str, confidence: float, threshold: float = 0.55) -> str:
    """
    Renders a custom SVG semi-circular forensic gauge showing detection confidence.
    
    Args:
        label: 'REAL' or 'AI-GENERATED'
        confidence: float between 0.0 and 1.0 (confidence of the winning class)
        threshold: float decision threshold (default 0.55)
    
    Returns:
        HTML string containing self-contained, responsive SVG gauge.
    """
    # Calculate synthetic probability (0.0 = completely real, 1.0 = completely synthetic)
    if label == "AI-GENERATED":
        prob_synthetic = confidence
        is_ai = True
        accent_color = "#ff2d55"
        glow_color = "rgba(255, 45, 85, 0.4)"
        status_text = "SYNTHETIC ARTIFACT DETECTED"
    else:
        prob_synthetic = 1.0 - confidence
        is_ai = False
        accent_color = "#00f0ff"
        glow_color = "rgba(0, 240, 255, 0.4)"
        status_text = "ORGANIC SPATIO-FREQUENCY SIGNATURE"

    # Clamp between 0.02 and 0.98 for visual aesthetics
    clamped_prob = max(0.02, min(0.98, prob_synthetic))

    # Semi-circle needle angle: 0.0 -> -90 deg (left), 0.5 -> 0 deg (top), 1.0 -> +90 deg (right)
    # In SVG polar coordinates for radius R=100, center at (150, 140)
    cx, cy, r = 150, 135, 95
    angle_deg = 180 + (clamped_prob * 180) # 180 deg (left) to 360 deg (right)
    angle_rad = math.radians(angle_deg)
    
    # Needle tip coordinates
    needle_len = r - 10
    nx = cx + needle_len * math.cos(angle_rad)
    ny = cy + needle_len * math.sin(angle_rad)

    # Threshold marker position (e.g. tau = 0.55 -> 180 + 0.55*180 = 279 deg)
    thresh_deg = 180 + (threshold * 180)
    thresh_rad = math.radians(thresh_deg)
    tx1 = cx + (r - 18) * math.cos(thresh_rad)
    ty1 = cy + (r - 18) * math.sin(thresh_rad)
    tx2 = cx + (r + 8) * math.cos(thresh_rad)
    ty2 = cy + (r + 8) * math.sin(thresh_rad)

    # Arc paths for background zones
    # Left zone (Real): 180 to (180 + thresh*180)
    # Right zone (AI): (180 + thresh*180) to 360
    
    html = f"""
    <div style="background: rgba(13, 19, 32, 0.9); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 1.25rem; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.5); position: relative;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 0.5rem; margin-bottom: 0.75rem;">
            <span style="font-family: 'Inter', sans-serif; font-size: 0.72rem; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase;">
                CALIBRATED TELEMETRY GAUGE
            </span>
            <span style="font-family: 'Inter', sans-serif; font-size: 0.72rem; color: #38bdf8; background: rgba(56,189,248,0.1); padding: 2px 6px; border-radius: 3px; border: 1px solid rgba(56,189,248,0.3);">
                &tau; = {threshold:.2f}
            </span>
        </div>

        <svg viewBox="0 0 300 175" style="width: 100%; max-width: 320px; margin: 0 auto; display: block; overflow: visible;">
            <defs>
                <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#00f0ff" />
                    <stop offset="45%" stop-color="#38bdf8" />
                    <stop offset="55%" stop-color="#f59e0b" />
                    <stop offset="100%" stop-color="#ff2d55" />
                </linearGradient>
                <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="3" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
            </defs>

            <!-- Background Arc Track -->
            <path d="M 45 135 A 105 105 0 0 1 255 135" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="22" stroke-linecap="round" />

            <!-- Colored Gradient Spectrum Arc -->
            <path d="M 55 135 A 95 95 0 0 1 245 135" fill="none" stroke="url(#gaugeGrad)" stroke-width="14" stroke-linecap="round" opacity="0.85" />

            <!-- Threshold Division Line -->
            <line x1="{tx1:.1f}" y1="{ty1:.1f}" x2="{tx2:.1f}" y2="{ty2:.1f}" stroke="#ffffff" stroke-width="2.5" stroke-dasharray="2,2" opacity="0.8" />
            <text x="{tx2 + 4:.1f}" y="{ty2 - 2:.1f}" fill="#94a3b8" font-family="'Inter', sans-serif" font-size="9" text-anchor="start">&tau; Cutoff</text>

            <!-- Major Zone Labels -->
            <text x="52" y="156" fill="#00f0ff" font-family="'Inter', sans-serif" font-size="10" font-weight="bold" text-anchor="middle">REAL</text>
            <text x="248" y="156" fill="#ff2d55" font-family="'Inter', sans-serif" font-size="10" font-weight="bold" text-anchor="middle">SYNTHETIC</text>

            <!-- Pivot Center Base -->
            <circle cx="{cx}" cy="{cy}" r="14" fill="#0f172a" stroke="{accent_color}" stroke-width="3" filter="url(#glow)" />
            <circle cx="{cx}" cy="{cy}" r="5" fill="{accent_color}" />

            <!-- Needle Pointer -->
            <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{accent_color}" stroke-width="3.5" stroke-linecap="round" filter="url(#glow)" />
        </svg>

        <!-- Digital Metric Display -->
        <div style="margin-top: 0.25rem;">
            <div style="font-family: 'Inter', sans-serif; font-size: 2.1rem; font-weight: 700; color: {accent_color}; letter-spacing: -0.02em; text-shadow: 0 0 16px {glow_color};">
                {confidence * 100:.1f}%
            </div>
            <div style="font-family: 'Inter', sans-serif; font-size: 0.8rem; font-weight: 600; color: #ffffff; letter-spacing: 0.05em; margin-top: 0.15rem;">
                VERDICT: <span style="color: {accent_color};">{label}</span>
            </div>
            <div style="font-family: 'Inter', sans-serif; font-size: 0.7rem; color: #64748b; margin-top: 0.35rem; letter-spacing: 0.04em;">
                {status_text}
            </div>
        </div>

        <!-- Linear Sub-Band Telemetry -->
        <div style="margin-top: 1rem; padding-top: 0.85rem; border-top: 1px dashed rgba(255,255,255,0.08);">
            <div style="display: flex; justify-content: space-between; font-family: 'Inter', sans-serif; font-size: 0.7rem; color: #94a3b8; margin-bottom: 0.35rem;">
                <span>Authentic Probability: { (1.0 - prob_synthetic) * 100:.1f}%</span>
                <span>Synthetic Probability: {prob_synthetic * 100:.1f}%</span>
            </div>
            <div style="height: 6px; width: 100%; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden; display: flex;">
                <div style="width: {(1.0 - prob_synthetic) * 100:.1f}%; background: #00f0ff; box-shadow: 0 0 8px rgba(0,240,255,0.5);"></div>
                <div style="width: {prob_synthetic * 100:.1f}%; background: #ff2d55; box-shadow: 0 0 8px rgba(255,45,85,0.5);"></div>
            </div>
        </div>
    </div>
    """
    return html


def render_compact_gauge_bar(label: str, confidence: float) -> str:
    """
    Renders a compact telemetry bar for tables or batch items.
    """
    is_ai = (label == "AI-GENERATED")
    color = "#ff2d55" if is_ai else "#00f0ff"
    pct = confidence * 100

    return f"""
    <div style="display: flex; align-items: center; gap: 0.5rem; font-family: 'Inter', sans-serif; font-size: 0.78rem;">
        <div style="flex-grow: 1; height: 6px; background: rgba(255,255,255,0.08); border-radius: 3px; overflow: hidden; min-width: 60px;">
            <div style="width: {pct:.1f}%; height: 100%; background: {color}; box-shadow: 0 0 6px {color};"></div>
        </div>
        <span style="color: {color}; font-weight: 600; min-width: 45px; text-align: right;">{pct:.1f}%</span>
    </div>
    """
