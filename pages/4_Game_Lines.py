"""Page 4 — Game Lines"""
from __future__ import annotations

import sys
from pathlib import Path

_PAGES_DIR = Path(__file__).resolve().parent
_ROOT      = _PAGES_DIR.parent
for _p in [str(_ROOT), str(_PAGES_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _engine_state import (
    SHARED_CSS, card, fmt_prob,
    init_session,
    team_color, team_name,
)
from projection_engine_v3 import PricingEngine

st.set_page_config(page_title="Game Lines · PLL", page_icon="🥍", layout="wide")
init_session()
st.markdown(SHARED_CSS, unsafe_allow_html=True)

result = st.session_state.get("last_result")
if result is None:
    st.warning("No projection loaded. Go to **Projections** first.")
    st.stop()

home_id  = result.home_proj.team_id
away_id  = result.away_proj.team_id
home_nm  = team_name(home_id)
away_nm  = team_name(away_id)
game     = st.session_state.selected_game or {}
gs       = result.game_sim
gm       = result.game_market
hold_pct = st.session_state.get("hold_pct", 0.045)

st.title("💰 Game Lines")
st.markdown(
    f"**{away_nm} @ {home_nm}** · "
    f"Game {game.get('game_number','—')} · "
    f"{str(game.get('game_date',''))[:10]}"
)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Line Overrides")
    st.markdown(
        '<span class="note-text">'
        "Override any line to see how the model prices it at a different number."
        "</span>",
        unsafe_allow_html=True,
    )

    use_tot  = st.checkbox("Override total line",  key="ov_tot_chk")
    cust_tot = st.number_input(
        "Total line", 10.0, 40.0, float(gm.total_line), 0.5,
        key="cust_tot", disabled=not use_tot,
        help="Set a custom total. Probabilities and odds update to show model's view at that line.",
    )

    use_spd  = st.checkbox("Override spread", key="ov_spd_chk")
    cust_spd = st.number_input(
        f"Spread ({home_nm})", -15.0, 15.0, float(gm.spread_home), 0.5,
        key="cust_spd", disabled=not use_spd,
        help=f"Positive = {home_nm} favored. Negative = {away_nm} favored.",
    )

    show_alt_spreads = st.checkbox("Show alternate spread table", value=False)
    show_alt_totals  = st.checkbox("Show alternate total table",  value=False)

    st.markdown("---")
    hold_slider = st.slider("Hold %", 2.0, 8.0, hold_pct * 100, 0.5, key="gl_hold") / 100.0

pricing = PricingEngine(hold_pct=hold_slider)

# ── Helpers ───────────────────────────────────────────────────────────────
def _am(prob: float) -> str:
    prob = min(max(prob, 0.001), 0.999)
    if prob >= 0.50:
        return str(int(-round((prob / (1 - prob)) * 100)))
    return "+" + str(int(round(((1 - prob) / prob) * 100)))

def _hold_odds(p1: float, p2: float):
    t = p1 + p2
    if t <= 0:
        return _am(0.5 + hold_slider / 2), _am(0.5 + hold_slider / 2)
    tgt = 1 + hold_slider
    return _am((p1 / t) * tgt), _am((p2 / t) * tgt)

total_arr  = gs.total_distribution
margin_arr = gs.margin_distribution

act_total  = cust_tot if use_tot else gm.total_line
act_spread = cust_spd if use_spd else gm.spread_home

p_over   = float(np.mean(total_arr  > act_total))
p_hcover = float(np.mean(margin_arr > act_spread))

over_odds,  under_odds = _hold_odds(p_over,    1 - p_over)
hspd_odds,  aspd_odds  = _hold_odds(p_hcover,  1 - p_hcover)

# ── Market display ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Moneyline")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(card(f"{away_nm}", gm.away_ml, f"Win prob: {fmt_prob(gm.away_win_prob)}"),
                unsafe_allow_html=True)
with c2:
    st.markdown(card("Model", "—", f"{gs.n_sims:,} simulations"), unsafe_allow_html=True)
with c3:
    st.markdown(card(f"{home_nm}", gm.home_ml, f"Win prob: {fmt_prob(gm.home_win_prob)}"),
                unsafe_allow_html=True)

st.markdown("### Spread")
away_spd = -act_spread
c1, c2, c3 = st.columns(3)
with c1:
    lbl = f"{away_nm} {away_spd:+.1f}" if away_spd != 0 else f"{away_nm} PK"
    st.markdown(card(lbl, aspd_odds, f"P(cover): {1-p_hcover:.1%}"),
                unsafe_allow_html=True)
with c2:
    note = " ⚡" if use_spd else ""
    st.markdown(card(f"Spread{note}", f"{act_spread:+.1f}", "model spread"),
                unsafe_allow_html=True)
with c3:
    lbl = f"{home_nm} {act_spread:+.1f}" if act_spread != 0 else f"{home_nm} PK"
    st.markdown(card(lbl, hspd_odds, f"P(cover): {p_hcover:.1%}"),
                unsafe_allow_html=True)

st.markdown("### Total")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(card("Over", over_odds, f"P(over): {p_over:.1%}"), unsafe_allow_html=True)
with c2:
    note = " ⚡" if use_tot else ""
    st.markdown(card(f"Line{note}", str(act_total), f"Sim median: {gs.expected_total:.1f}"),
                unsafe_allow_html=True)
with c3:
    st.markdown(card("Under", under_odds, f"P(under): {1-p_over:.1%}"), unsafe_allow_html=True)

st.markdown("---")

# ── Summary table ─────────────────────────────────────────────────────────
st.markdown("### Full Market Summary")
summary = pd.DataFrame([
    {"Market": f"{away_nm} ML",              "Line": "—",               "Odds": gm.away_ml,
     "Fair Prob": fmt_prob(gm.away_win_prob), "Hold": f"{hold_slider*100:.1f}%"},
    {"Market": f"{home_nm} ML",              "Line": "—",               "Odds": gm.home_ml,
     "Fair Prob": fmt_prob(gm.home_win_prob), "Hold": f"{hold_slider*100:.1f}%"},
    {"Market": f"{away_nm} {away_spd:+.1f}", "Line": f"{away_spd:+.1f}", "Odds": aspd_odds,
     "Fair Prob": fmt_prob(1-p_hcover),       "Hold": f"{hold_slider*100:.1f}%"},
    {"Market": f"{home_nm} {act_spread:+.1f}","Line":f"{act_spread:+.1f}","Odds": hspd_odds,
     "Fair Prob": fmt_prob(p_hcover),          "Hold": f"{hold_slider*100:.1f}%"},
    {"Market": "Total Over",                 "Line": str(act_total),    "Odds": over_odds,
     "Fair Prob": fmt_prob(p_over),           "Hold": f"{hold_slider*100:.1f}%"},
    {"Market": "Total Under",                "Line": str(act_total),    "Odds": under_odds,
     "Fair Prob": fmt_prob(1-p_over),         "Hold": f"{hold_slider*100:.1f}%"},
])
st.dataframe(summary, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Alternate spread table ────────────────────────────────────────────────
if show_alt_spreads:
    st.markdown("### Alternate Spread Lines")
    st.markdown(
        f'<span class="note-text">'
        f"Model's probability at every half-point spread for this matchup. "
        f"Use to find value when the market line differs from the model spread."
        f"</span>",
        unsafe_allow_html=True,
    )
    spd_min = float(np.percentile(margin_arr, 2))
    spd_max = float(np.percentile(margin_arr, 98))
    alt_spreads = [round(s * 0.5, 1) for s in range(int(spd_min * 2) - 2, int(spd_max * 2) + 3)]
    alt_spd_rows = []
    for asp in alt_spreads:
        p_h = float(np.mean(margin_arr > asp))
        h_o, a_o = _hold_odds(p_h, 1 - p_h)
        alt_spd_rows.append({
            f"{home_nm} Spread": f"{asp:+.1f}",
            f"P({home_nm} covers)": f"{p_h:.3f}",
            f"{home_nm} odds": h_o,
            f"P({away_nm} covers)": f"{1-p_h:.3f}",
            f"{away_nm} odds": a_o,
            "Model spread": f"{gm.spread_home:+.1f}",
        })
    st.dataframe(pd.DataFrame(alt_spd_rows), use_container_width=True, hide_index=True)
    st.markdown("---")

# ── Alternate totals table ────────────────────────────────────────────────
if show_alt_totals:
    st.markdown("### Alternate Total Lines")
    st.markdown(
        '<span class="note-text">'
        "Model's over/under probability at every half-point total."
        "</span>",
        unsafe_allow_html=True,
    )
    tot_min = max(float(np.percentile(total_arr, 1)), 5.0)
    tot_max = float(np.percentile(total_arr, 99)) + 3.0
    alt_totals = [round(t * 0.5, 1) for t in range(int(tot_min * 2), int(tot_max * 2) + 2)]
    alt_tot_rows = []
    for at_val in alt_totals:
        p_ov = float(np.mean(total_arr > at_val))
        o_o, u_o = _hold_odds(p_ov, 1 - p_ov)
        alt_tot_rows.append({
            "Total Line": at_val,
            "P(Over)":  f"{p_ov:.3f}",
            "Over Odds":  o_o,
            "P(Under)": f"{1-p_ov:.3f}",
            "Under Odds": u_o,
            "Model Total": f"{gs.expected_total:.1f}",
        })
    st.dataframe(pd.DataFrame(alt_tot_rows), use_container_width=True, hide_index=True)
    st.markdown("---")

# ── Score probability grid ────────────────────────────────────────────────
st.markdown("### Score Probability Grid")
hs  = gs.home_scores.astype(int)
as_ = gs.away_scores.astype(int)
h_vals = sorted(set(hs.clip(0, 30)))[:16]
a_vals = sorted(set(as_.clip(0, 30)))[:16]
n = len(hs)
grid = [[float(np.mean((hs == hv) & (as_ == av))) * 100
         for hv in h_vals] for av in a_vals]

fig = go.Figure(go.Heatmap(
    z=grid,
    x=[str(v) for v in h_vals],
    y=[str(v) for v in a_vals],
    colorscale="Blues",
    text=[[f"{grid[i][j]:.1f}%" for j in range(len(h_vals))] for i in range(len(a_vals))],
    texttemplate="%{text}", textfont=dict(size=9),
    showscale=True, colorbar=dict(title="Prob %"),
))
fig.update_layout(
    height=400, margin=dict(l=0,r=0,t=28,b=0),
    xaxis_title=f"{home_nm} Score",
    yaxis_title=f"{away_nm} Score",
    title=f"Score Probability Grid ({n//1000}k sims)",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f1f5f9"),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Total distribution ─────────────────────────────────────────────────────
st.markdown("### Score Total Distribution")
fig2 = go.Figure(go.Histogram(x=total_arr, nbinsx=35, marker_color="#3b82f6", opacity=0.7))
fig2.add_vline(
    x=act_total, line_dash="dash", line_color="#f59e0b", line_width=2,
    annotation_text=f"Line: {act_total}  O{over_odds}/U{under_odds}",
    annotation_position="top right",
)
fig2.update_layout(
    height=250, margin=dict(l=0,r=0,t=6,b=0),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f1f5f9"),
    xaxis_title="Combined Score", yaxis_title="Simulations",
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    f'<span class="note-text">'
    f"Engine v3 · {gs.n_sims:,} sims · Hold: {hold_slider*100:.1f}%"
    f"</span>",
    unsafe_allow_html=True,
)
