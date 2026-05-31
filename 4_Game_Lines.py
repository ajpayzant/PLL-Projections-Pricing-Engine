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
    get_engine, init_session,
    team_color, team_name,
    render_global_projection_runner,
)
from projection_engine_v3 import PricingEngine

st.set_page_config(page_title="Game Lines · PLL", page_icon="🥍", layout="wide")
init_session()
st.markdown(SHARED_CSS, unsafe_allow_html=True)

engine = get_engine()
render_global_projection_runner(engine=engine, key_prefix="lines")

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
        "Total line", 10.5, 40.5, float(gm.total_line), 1.0,
        key="cust_tot", disabled=not use_tot,
        help="Set a custom total. Lines are forced to x.5 to avoid pushes.",
    )

    use_spd  = st.checkbox("Override spread", key="ov_spd_chk")
    cust_spd = st.number_input(
        f"Spread ({home_nm})", -15.5, 15.5, float(gm.spread_home), 1.0,
        key="cust_spd", disabled=not use_spd,
        help=f"Positive = {home_nm} favored. Lines are forced to x.5 to avoid pushes.",
    )

    show_alt_spreads = st.checkbox("Show alternate spread table", value=False)
    show_alt_totals  = st.checkbox("Show alternate total table",  value=False)
    show_alt_team_totals = st.checkbox("Show alternate team-total tables", value=False)

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

def _force_half_only(x: float) -> float:
    """Snap to the nearest x.5 line and never display x.0."""
    try:
        v = float(x)
    except Exception:
        return 0.5
    if not np.isfinite(v):
        return 0.5
    lower = np.floor(v) + 0.5
    upper = lower + (1.0 if lower < v else -1.0)
    return round(float(min([lower, upper], key=lambda c: abs(c - v))), 1)

def _half_only_lines(lo: float, hi: float):
    """Generate alternate lines with decimal .5 only; whole numbers are excluded."""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return [0.5]
    if hi < lo:
        lo, hi = hi, lo
    start = np.floor(lo) + 0.5
    if start < lo - 1e-9:
        start += 1.0
    end = np.ceil(hi) + 0.5
    return [round(float(v), 1) for v in np.arange(start, end + 1e-9, 1.0)]

def _opt_half_line(arr, allow_negative=False) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.5
    lo = float(np.percentile(arr, 1)) - (3.0 if allow_negative else 0.0)
    hi = float(np.percentile(arr, 99)) + 5.0
    if not allow_negative:
        lo = max(0.5, lo)
    best, best_d = 0.5, float("inf")
    for line in _half_only_lines(lo, hi):
        d = abs(float(np.mean(arr > line)) - 0.50)
        if d < best_d:
            best, best_d = line, d
    return round(float(best), 1)

def _price_arr(arr, line):
    line = _force_half_only(line)
    p_over_local = float(np.mean(np.asarray(arr, dtype=float) > line))
    o, u = _hold_odds(p_over_local, 1 - p_over_local)
    return line, p_over_local, o, u

total_arr  = gs.total_distribution
margin_arr = gs.margin_distribution
home_team_total_arr = gs.home_scores
away_team_total_arr = gs.away_scores

act_total  = _force_half_only(cust_tot if use_tot else gm.total_line)
act_spread = _force_half_only(cust_spd if use_spd else gm.spread_home)

p_over   = float(np.mean(total_arr  > act_total))
p_hcover = float(np.mean(margin_arr > act_spread))

over_odds,  under_odds = _hold_odds(p_over,    1 - p_over)
hspd_odds,  aspd_odds  = _hold_odds(p_hcover,  1 - p_hcover)

away_tt_line = _opt_half_line(away_team_total_arr)
home_tt_line = _opt_half_line(home_team_total_arr)
away_tt_line, p_away_tt_over, away_tt_over_odds, away_tt_under_odds = _price_arr(away_team_total_arr, away_tt_line)
home_tt_line, p_home_tt_over, home_tt_over_odds, home_tt_under_odds = _price_arr(home_team_total_arr, home_tt_line)

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
    st.markdown(card(f"Line{note}", f"{act_total:.1f}", f"Sim median: {gs.expected_total:.1f}"),
                unsafe_allow_html=True)
with c3:
    st.markdown(card("Under", under_odds, f"P(under): {1-p_over:.1%}"), unsafe_allow_html=True)

st.markdown("### Team Totals")
c1, c2 = st.columns(2)
with c1:
    st.markdown(card(
        f"{away_nm} Team Total",
        f"{away_tt_line:.1f}",
        f"Over {away_tt_over_odds} / Under {away_tt_under_odds} · P(O): {p_away_tt_over:.1%}",
    ), unsafe_allow_html=True)
with c2:
    st.markdown(card(
        f"{home_nm} Team Total",
        f"{home_tt_line:.1f}",
        f"Over {home_tt_over_odds} / Under {home_tt_under_odds} · P(O): {p_home_tt_over:.1%}",
    ), unsafe_allow_html=True)

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
    {"Market": "Total Over",                 "Line": f"{act_total:.1f}", "Odds": over_odds,
     "Fair Prob": fmt_prob(p_over),           "Hold": f"{hold_slider*100:.1f}%"},
    {"Market": "Total Under",                "Line": f"{act_total:.1f}", "Odds": under_odds,
     "Fair Prob": fmt_prob(1-p_over),         "Hold": f"{hold_slider*100:.1f}%"},
    {"Market": f"{away_nm} Team Total Over",  "Line": f"{away_tt_line:.1f}", "Odds": away_tt_over_odds,
     "Fair Prob": fmt_prob(p_away_tt_over),   "Hold": f"{hold_slider*100:.1f}%"},
    {"Market": f"{away_nm} Team Total Under", "Line": f"{away_tt_line:.1f}", "Odds": away_tt_under_odds,
     "Fair Prob": fmt_prob(1-p_away_tt_over), "Hold": f"{hold_slider*100:.1f}%"},
    {"Market": f"{home_nm} Team Total Over",  "Line": f"{home_tt_line:.1f}", "Odds": home_tt_over_odds,
     "Fair Prob": fmt_prob(p_home_tt_over),   "Hold": f"{hold_slider*100:.1f}%"},
    {"Market": f"{home_nm} Team Total Under", "Line": f"{home_tt_line:.1f}", "Odds": home_tt_under_odds,
     "Fair Prob": fmt_prob(1-p_home_tt_over), "Hold": f"{hold_slider*100:.1f}%"},
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
    spd_min = float(np.percentile(margin_arr, 2)) - 2.0
    spd_max = float(np.percentile(margin_arr, 98)) + 2.0
    alt_spreads = _half_only_lines(spd_min, spd_max)
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
    alt_totals = _half_only_lines(tot_min, tot_max)
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

# ── Alternate team-total tables ───────────────────────────────────────────
if show_alt_team_totals:
    st.markdown("### Alternate Team Total Lines")
    team_total_rows = []
    for label, arr, model_line in [
        (away_nm, away_team_total_arr, away_tt_line),
        (home_nm, home_team_total_arr, home_tt_line),
    ]:
        lo = max(float(np.percentile(arr, 2)) - 1.0, 0.5)
        hi = float(np.percentile(arr, 98)) + 2.0
        for line in _half_only_lines(lo, hi):
            p_ov = float(np.mean(arr > line))
            o_o, u_o = _hold_odds(p_ov, 1 - p_ov)
            team_total_rows.append({
                "Team": label,
                "Team Total Line": f"{line:.1f}",
                "P(Over)": f"{p_ov:.3f}",
                "Over Odds": o_o,
                "P(Under)": f"{1-p_ov:.3f}",
                "Under Odds": u_o,
                "Main Line": f"{model_line:.1f}",
            })
    st.dataframe(pd.DataFrame(team_total_rows), use_container_width=True, hide_index=True)
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
    annotation_text=f"Line: {act_total:.1f}  O{over_odds}/U{under_odds}",
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
