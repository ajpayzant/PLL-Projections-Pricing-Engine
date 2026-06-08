"""Page 2 -- Player Props"""
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
    SHARED_CSS, pos_badge, fmt_prob,
    get_engine, init_session,
    team_color, team_name,
    render_update_projection_btn,
)
from projection_engine_v3 import PricingEngine

st.set_page_config(page_title="Player Props · PLL", page_icon="🥍", layout="wide")
init_session()
st.markdown(SHARED_CSS, unsafe_allow_html=True)

result = st.session_state.get("last_result")
if result is None:
    st.warning("No projection loaded. Go to **Projections** and run a game first.")
    st.stop()

game    = st.session_state.selected_game or {}
home_id = result.home_proj.team_id
away_id = result.away_proj.team_id
home_nm = team_name(home_id)
away_nm = team_name(away_id)
hold_pct = st.session_state.get("hold_pct", 0.045)
pricing  = PricingEngine(hold_pct=hold_pct)

st.title("👤 Player Prop Markets")
st.markdown(
    f"**{away_nm} @ {home_nm}** · "
    f"Game {game.get('game_number','--')} · "
    f"{str(game.get('game_date',''))[:10]}"
)

STAT_LABELS = {
    "goals": "Goals", "assists": "Assists", "points": "Points",
    "shots": "Shots", "shots_on_goal": "SOG", "two_pt_goals": "2PT Goals",
    "one_pt_goals": "1PT Goals", "saves": "Saves", "faceoff_wins": "FO Wins",
    "ground_balls": "Ground Balls",
}
FIELD_STATS  = ["goals", "assists", "points", "shots_on_goal", "two_pt_goals"]
GOALIE_STATS = ["saves"]
FO_STATS     = ["faceoff_wins"]
MILE_DEFS    = {"goals": [1, 2, 3], "assists": [1, 2], "saves": [10, 12, 14]}

# -- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filters")
    show_team = st.radio("Team", ["Both", away_nm, home_nm], key="prop_team")
    show_pos  = st.multiselect(
        "Positions", ["A","M","D","FO","SSDM","LSM","G"],
        default=["A","M","FO","G"], key="prop_pos",
    )
    min_pts   = st.slider("Min projected points", 0.0, 3.0, 0.3, 0.1, key="prop_min_pts")
    show_miles = st.checkbox("Show milestone props (1+, 2+, 3+)", value=True)
    show_alt   = st.checkbox("Show alternate line pricing", value=False)

    st.markdown("---")
    st.markdown("### Hold %")
    hcol1, hcol2 = st.columns([3, 1])
    with hcol1:
        hold_sl = st.slider("Hold %", 2.0, 8.0, float(hold_pct * 100), 0.5, key="pp_hold_slider",
                            label_visibility="collapsed")
    with hcol2:
        hold_num = st.number_input("", 2.0, 8.0, hold_sl, 0.5, key="pp_hold_num",
                                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Quick Line Override")
    st.markdown('<span class="note-text">Price any player at a custom line.</span>',
                unsafe_allow_html=True)
    ov_player = st.text_input("Player name (partial)", key="ov_player")
    ov_stat   = st.selectbox("Stat", ["goals","assists","points","shots_on_goal",
                                       "saves","faceoff_wins"], key="ov_stat")
    ov_line   = st.number_input(
        "Line", 0.5, 25.5, 0.5, 1.0, key="ov_line",
        help="Lines are forced to x.5 values to avoid pushes."
    )

    st.markdown("---")
    engine = get_engine()
    render_update_projection_btn(engine, key="p2")

# hold_pct and pricing defined OUTSIDE sidebar so accessible to prop pricing below
new_hold_pct = (hold_num if abs(hold_num - hold_sl) > 0.1 else hold_sl) / 100.0
st.session_state.hold_pct = new_hold_pct
pricing = PricingEngine(hold_pct=new_hold_pct)

# -- Collect sims ----------------------------------------------------------
all_projs = {p.player_id: p for p in result.home_players + result.away_players}
markets   = result.player_markets

def _half_only_lines(lo: float, hi: float):
    """Generate alternate prop lines with decimal .5 only; whole numbers are excluded."""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return [0.5]
    if hi < lo:
        lo, hi = hi, lo
    lo = max(0.5, lo)
    start = np.floor(lo) + 0.5
    if start < lo - 1e-9:
        start += 1.0
    end = np.ceil(hi) + 0.5
    return [round(float(v), 1) for v in np.arange(start, end + 1e-9, 1.0)]

def _alt_width(stat: str) -> float:
    return 5.0 if stat in {"saves", "faceoff_wins"} else 3.0

def _keep(pid: str) -> bool:
    pm  = markets.get(pid, {})
    pv  = pm.get("proj_values", {})
    pts = max(pv.get("points",0), pv.get("saves",0), pv.get("faceoff_wins",0))
    if pts < min_pts:
        return False
    proj = all_projs.get(pid)
    if proj is None or not proj.active:
        return False
    if proj.position not in show_pos:
        return False
    if show_team == away_nm and proj.team_id != away_id:
        return False
    if show_team == home_nm and proj.team_id != home_id:
        return False
    return True

sims_filtered = sorted(
    [s for s in (result.home_player_sims + result.away_player_sims) if _keep(s.player_id)],
    key=lambda s: markets.get(s.player_id, {}).get("proj_values", {}).get("points", 0),
    reverse=True,
)

if not sims_filtered:
    st.info("No players match the current filters.")
    st.stop()

st.markdown(f"**{len(sims_filtered)} players shown** · hold: {new_hold_pct*100:.1f}%")
st.markdown("---")

# -- Player prop cards -----------------------------------------------------
for ps in sims_filtered:
    pid  = ps.player_id
    pm   = markets.get(pid, {})
    proj = all_projs.get(pid)
    if proj is None:
        continue

    pv  = pm.get("proj_values", {})
    nm  = proj.full_name or pid
    pos = proj.position
    tid = proj.team_id

    # -- Get primary stat line for collapsed summary row ------------------
    if pos == "G":
        pri_stat, pri_proj = "saves", proj.proj_saves
    elif pos == "FO":
        pri_stat, pri_proj = "faceoff_wins", proj.proj_faceoff_wins
    else:
        pri_stat, pri_proj = "points", proj.proj_points

    pm_data = markets.get(pid, {})
    pri_market = pm_data.get("markets", {}).get(pri_stat, {})
    if pri_market:
        line_val = pri_market.get("line")
        line_str = f"{line_val:.1f}" if isinstance(line_val, (int, float)) else "?"
        over_str = pri_market.get("over_odds", "--")
        under_str = pri_market.get("under_odds", "--")
        expander_label = (
            f"{nm}  ·  {pos}  ·  {team_name(tid)}  |  "
            f"Proj: {pri_proj:.2f}  |  Line: {line_str}  |  "
            f"O {over_str} / U {under_str}"
        )
    else:
        expander_label = f"{nm}  ·  {pos}  ·  {team_name(tid)}  |  Proj: {pri_proj:.2f}"

    with st.expander(expander_label, expanded=False):
        col_info, col_dist = st.columns([1, 2])

        with col_info:
            st.markdown(f"**Pos:** {pos_badge(pos)}", unsafe_allow_html=True)
            st.markdown(f"**Team:** {team_name(tid)}")
            if pos == "G":
                st.markdown(f"Proj Saves: **{proj.proj_saves:.2f}**")
                st.markdown(f"Save%: **{proj.proj_save_pct:.3f}**")
            elif pos == "FO":
                st.markdown(f"FO Wins: **{proj.proj_faceoff_wins:.2f}**")
                st.markdown(f"FO%: **{proj.proj_faceoff_pct:.3f}**")
            else:
                st.markdown(f"Goals: **{proj.proj_goals:.3f}**")
                st.markdown(f"Assists: **{proj.proj_assists:.3f}**")
                st.markdown(f"Points: **{proj.proj_points:.3f}**")
                st.markdown(f"Shots: **{proj.proj_shots:.2f}**  SOG: **{proj.proj_sog:.2f}**")
                if proj.proj_2pt_goals > 0.02:
                    rate = proj.proj_2pt_goals / max(proj.proj_goals, 0.01)
                    st.markdown(f"2PT Rate: **{rate:.1%}**")
                st.markdown(f"Zero-score prob: **{proj.zero_prob_goals:.1%}**")

        with col_dist:
            pri = "saves" if pos == "G" else ("faceoff_wins" if pos == "FO" else "points")
            if pri in ps.stat_distributions:
                dist = ps.stat_distributions[pri]
                fig  = go.Figure(go.Histogram(x=dist, nbinsx=20,
                                              marker_color=team_color(tid), opacity=0.75))
                pv_val = pv.get(pri, 0)
                fig.add_vline(x=pv_val, line_dash="dash", line_color="#f59e0b",
                              annotation_text=f"Proj: {pv_val:.2f}")
                fig.update_layout(
                    height=170, margin=dict(l=0,r=0,t=4,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f1f5f9"), showlegend=False,
                    xaxis_title=STAT_LABELS.get(pri, pri), yaxis_title="",
                )
                st.plotly_chart(fig, use_container_width=True)

        # -- Model line table ----------------------------------------------
        stat_list = GOALIE_STATS if pos == "G" else (FO_STATS if pos == "FO" else FIELD_STATS)
        rows = []
        for stat in stat_list:
            if stat not in ps.stat_distributions:
                continue
            dist = ps.stat_distributions[stat]
            custom_line = None
            if ov_player and ov_player.lower() in nm.lower() and ov_stat == stat:
                custom_line = ov_line
            ml  = pricing.price_prop(ps, stat, line=custom_line)
            pct = float(np.percentile(dist, 75)) - float(np.percentile(dist, 25))
            rows.append({
                "Stat":     STAT_LABELS.get(stat, stat),
                "Proj":     f"{pv.get(stat,0):.3f}",
                "Line":     f"{ml.line:.1f}",
                "P(Over)":  f"{ml.fair_over_prob:.3f}",
                "Over":     ml.over_odds,
                "P(Under)": f"{ml.fair_under_prob:.3f}",
                "Under":    ml.under_odds,
                "IQR":      f"{pct:.2f}",
                "P10":      f"{np.percentile(dist,10):.1f}",
                "Median":   f"{np.percentile(dist,50):.1f}",
                "P90":      f"{np.percentile(dist,90):.1f}",
            })
        if rows:
            st.markdown("**Model Lines**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # -- Alternate line pricing -----------------------------------------
        if show_alt:
            for stat in stat_list:
                if stat not in ps.stat_distributions:
                    continue
                dist = ps.stat_distributions[stat]
                proj_v = pv.get(stat, 0)
                st.markdown(f"**Alternate Lines -- {STAT_LABELS.get(stat, stat)}**")

                # Build a compact ladder around the main model line.
                # Lines are x.5 only (0.5, 1.5, 2.5...) so integer outcomes cannot push.
                main_ml = pricing.price_prop(ps, stat)
                width = _alt_width(stat)
                lo = main_ml.line - width
                hi = main_ml.line + width
                alt_lines = _half_only_lines(lo, hi)
                alt_rows = []
                for al in alt_lines:
                    ml_a = pricing.price_prop(ps, stat, line=al)
                    alt_rows.append({
                        "Line": f"{al:.1f}",
                        "P(Over)":  f"{ml_a.fair_over_prob:.3f}",
                        "Over Odds":  ml_a.over_odds,
                        "P(Under)": f"{ml_a.fair_under_prob:.3f}",
                        "Under Odds": ml_a.under_odds,
                        "Main Line": f"{main_ml.line:.1f}",
                        "Model Proj": f"{proj_v:.3f}",
                    })
                if alt_rows:
                    st.dataframe(pd.DataFrame(alt_rows), use_container_width=True, hide_index=True)

        # -- Milestone props ------------------------------------------------
        if show_miles:
            for stat, levels in MILE_DEFS.items():
                if stat not in ps.stat_distributions:
                    continue
                dist = ps.stat_distributions[stat]
                m_rows = []
                for lvl in levels:
                    ml_m = pricing.price_prop(ps, stat, line=lvl - 0.5)
                    m_rows.append({
                        "Milestone": f"{STAT_LABELS.get(stat,stat)} {lvl}+",
                        "P(Hit)":    f"{float(np.mean(dist >= lvl)):.3f}",
                        "Yes odds":  ml_m.over_odds,
                        "No odds":   ml_m.under_odds,
                    })
                if m_rows:
                    st.markdown(f"**Milestones -- {STAT_LABELS.get(stat,stat)}**")
                    st.dataframe(pd.DataFrame(m_rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
    f'<span class="note-text">'
    f'Hold: {new_hold_pct*100:.1f}% · 20,000 sims · '
    f'Adjust hold in sidebar · '
    f'Enable "Alternate line pricing" in sidebar for full line grids'
    f'</span>',
    unsafe_allow_html=True,
)
