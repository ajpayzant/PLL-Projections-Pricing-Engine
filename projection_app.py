"""
PLL Projection App
==================
Entry point. Run with:  streamlit run projection_app.py
"""
import streamlit as st

st.set_page_config(
    page_title="PLL Projections",
    page_icon="🥍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main .block-container { padding-top:1rem; padding-bottom:2rem; max-width:1800px; }
  h1,h2,h3 { letter-spacing:-0.02em; }
  .pll-card {
    border:1px solid rgba(148,163,184,.20); border-radius:12px; padding:12px 16px;
    background:linear-gradient(160deg,rgba(255,255,255,.04),rgba(255,255,255,.01));
    box-shadow:0 4px 16px rgba(0,0,0,.10); margin-bottom:8px;
  }
  .pll-card-label { color:#94a3b8; font-size:.78rem; font-weight:600;
    text-transform:uppercase; letter-spacing:.05em; margin-bottom:4px; }
  .pll-card-value { font-size:1.5rem; font-weight:800; color:#f1f5f9; line-height:1.1; }
  .pll-card-sub   { color:#94a3b8; font-size:.78rem; margin-top:3px; }
  .note-text { color:#64748b; font-size:.80rem; font-style:italic; }
</style>
""", unsafe_allow_html=True)

st.title("🥍 PLL Projections")
st.markdown("##### Monte Carlo projection system for Premier Lacrosse League games")
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="pll-card"><div class="pll-card-label">📊 Page 1</div>'
                '<div class="pll-card-value">Projections</div>'
                '<div class="pll-card-sub">Select a game, run projections, view team stats and sim distributions</div></div>',
                unsafe_allow_html=True)
with c2:
    st.markdown('<div class="pll-card"><div class="pll-card-label">👤 Page 2</div>'
                '<div class="pll-card-value">Player Props</div>'
                '<div class="pll-card-sub">Goals, assists, points, SOG, saves, FO wins with American odds</div></div>',
                unsafe_allow_html=True)
with c3:
    st.markdown('<div class="pll-card"><div class="pll-card-label">📋 Page 3</div>'
                '<div class="pll-card-value">Depth Charts</div>'
                '<div class="pll-card-sub">Mark players inactive, set starters, adjust usage multipliers</div></div>',
                unsafe_allow_html=True)
with c4:
    st.markdown('<div class="pll-card"><div class="pll-card-label">💰 Page 4</div>'
                '<div class="pll-card-value">Game Lines</div>'
                '<div class="pll-card-sub">Final moneyline, spread, total with score probability grid</div></div>',
                unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
**Workflow:**
1. **Projections** → pick a game → click Run Projection
2. **Depth Charts** → mark any scratched players inactive
3. **Projections** → re-run to apply roster changes
4. **Player Props** → review every player's prop lines
5. **Game Lines** → final market output with optional line overrides

<span class="note-text">Engine v3 · Possession-chain model · 20,000 Monte Carlo simulations · Zero-inflated distributions · Calibrated to 2022–2025 seasons</span>
""", unsafe_allow_html=True)
