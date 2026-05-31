"""Page 1 — Game Projections"""
from __future__ import annotations

import sys
from pathlib import Path

_PAGES_DIR = Path(__file__).resolve().parent
_ROOT      = _PAGES_DIR.parent
for _p in [str(_ROOT), str(_PAGES_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _engine_state import (
    SHARED_CSS, TEAM_RATING_DEFS,
    card, fmt_prob, fmt_goals,
    get_engine, init_session,
    team_color, team_name,
    build_overrides, build_active_players,
    get_team_rating_overrides, set_team_rating_override,
    sorted_upcoming, default_game_index,
)

st.set_page_config(page_title="Projections · PLL", page_icon="🥍", layout="wide")
init_session()
st.markdown(SHARED_CSS, unsafe_allow_html=True)

engine   = get_engine()
raw_games = engine.upcoming_games()

# ── Attach season to each game dict (for sorting) ─────────────────────────
def _season_from_game(g: dict) -> int:
    gdate = str(g.get("game_date", ""))
    try:
        return int(gdate[:4])
    except Exception:
        return 0

for g in raw_games:
    g["game_number_season"] = _season_from_game(g)

upcoming = sorted_upcoming(raw_games)

st.title("📊 Game Projections")

if not upcoming:
    st.warning("No upcoming games found. The data warehouse may need to be refreshed "
               "(run the GitHub Action: Update PLL Data Warehouse).")
    st.stop()

today = dt.date.today()
current_year = today.year

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Game selector ─────────────────────────────────────────────────────
    st.markdown("### Select Game")

    # Group by season for display
    seasons_in_list = sorted(set(g["game_number_season"] for g in upcoming), reverse=True)
    current_season = max(seasons_in_list) if seasons_in_list else current_year

    season_filter = st.selectbox(
        "Season",
        options=seasons_in_list,
        format_func=lambda s: f"{s} Season",
        index=0,
        key="season_filter",
    )
    season_games = [g for g in upcoming if g["game_number_season"] == season_filter]

    if not season_games:
        st.warning("No upcoming games for that season.")
        st.stop()

    # Build display labels: "Week XX · Away @ Home · Date"
    def _game_label(g: dict) -> str:
        ht   = team_name(g.get("home_team_id",""))
        at   = team_name(g.get("away_team_id",""))
        gnum = g.get("game_number","?")
        date = str(g.get("game_date",""))[:10]
        return f"Game {gnum} · {at} @ {ht} · {date}"

    game_labels = [_game_label(g) for g in season_games]

    # Default to next game whose date >= today in the selected season
    default_idx = 0
    for i, g in enumerate(season_games):
        try:
            gd = dt.date.fromisoformat(str(g.get("game_date",""))[:10])
            if gd >= today:
                default_idx = i
                break
        except Exception:
            pass

    game_idx = st.selectbox(
        "Game",
        options=range(len(season_games)),
        format_func=lambda i: game_labels[i],
        index=default_idx,
        key="game_idx",
    )
    game = season_games[game_idx]
    st.session_state.selected_game = game

    home_id = str(game.get("home_team_id",""))
    away_id = str(game.get("away_team_id",""))
    home_nm = team_name(home_id)
    away_nm = team_name(away_id)

    st.markdown("---")

    # ── Team rating adjustments ───────────────────────────────────────────
    st.markdown("### Team Rating Adjustments")
    st.markdown(
        '<span class="note-text">'
        "Adjust the model's actual input ratings. Each slider shows the model's "
        "current value — move it to reflect information the stats don't capture "
        "(injury, travel, matchup advantage). The projection recalculates "
        "automatically when you run it."
        "</span>",
        unsafe_allow_html=True,
    )

    # Fetch current model ratings for both teams so we can show them as defaults
    rb = engine.rating_builder
    hf_current = rb.get_team_rating(home_id) if rb else {}
    af_current = rb.get_team_rating(away_id) if rb else {}

    def _rating_sliders(team_id: str, team_nm: str, current_ratings: dict):
        """Render adjustable sliders for one team's key ratings."""
        st.markdown(f"**{team_nm}**")
        overrides = get_team_rating_overrides(team_id)

        for key, meta in TEAM_RATING_DEFS.items():
            model_val = float(current_ratings.get(key, 0.0))
            if model_val == 0.0:
                continue  # skip if rating not available

            current_override = overrides.get(key, model_val)

            new_val = st.slider(
                label=meta["label"],
                min_value=meta["min"],
                max_value=meta["max"],
                value=float(current_override),
                step=meta["step"],
                help=meta["help"],
                key=f"tr_{team_id}_{key}",
            )
            # Show model value as reference
            changed = abs(new_val - model_val) > meta["step"] * 0.5
            model_str = meta["fmt"].format(model_val)
            if changed:
                st.markdown(
                    f'<span class="rating-changed note-text">Model: {model_str} → '
                    f'You: {meta["fmt"].format(new_val)}</span>',
                    unsafe_allow_html=True,
                )
                set_team_rating_override(team_id, key, new_val)
            else:
                st.markdown(
                    f'<span class="note-text">Model value: {model_str}</span>',
                    unsafe_allow_html=True,
                )
                # Remove override if it matches model (user reset slider)
                if key in get_team_rating_overrides(team_id):
                    del st.session_state.team_rating_overrides[team_id][key]

    _rating_sliders(home_id, home_nm + " (Home)", hf_current)
    st.markdown("")
    _rating_sliders(away_id, away_nm + " (Away)", af_current)

    st.markdown("---")
    hold_pct = st.slider("Market hold %", 2.0, 8.0, 4.5, 0.5, key="hold_slider") / 100.0
    st.session_state.hold_pct = hold_pct

    if st.button("Reset all adjustments", key="reset_adj"):
        st.session_state.team_rating_overrides = {}
        st.rerun()

    run_btn = st.button("▶  Run Projection", type="primary", use_container_width=True)

# ── Run projection ────────────────────────────────────────────────────────
team_rating_overrides = {}
for tid in [home_id, away_id]:
    ov = get_team_rating_overrides(tid)
    if ov:
        team_rating_overrides[tid] = ov

if run_btn or st.session_state.last_result is None:
    with st.spinner("Running 20,000 simulations…"):
        result = engine.project(
            home_team_id=home_id,
            away_team_id=away_id,
            player_overrides=build_overrides() or None,
            active_players=build_active_players() or None,
            team_rating_overrides=team_rating_overrides or None,
        )
        st.session_state.last_result = result

result = st.session_state.last_result
if result is None:
    st.info("Click **▶ Run Projection** in the sidebar.")
    st.stop()

hp = result.home_proj
ap = result.away_proj
gs = result.game_sim
gm = result.game_market

# ── Game header ───────────────────────────────────────────────────────────
st.markdown(
    f'<h2 style="text-align:center;margin-bottom:2px;">'
    f'<span style="color:{team_color(away_id)}">{away_nm}</span>'
    f' &nbsp;@&nbsp; '
    f'<span style="color:{team_color(home_id)}">{home_nm}</span>'
    f'</h2>'
    f'<p style="text-align:center;color:#94a3b8;margin-top:0;">'
    f'Game {game.get("game_number","?")} · {str(game.get("game_date",""))[:10]}'
    f'</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Win probability row ───────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 2])
with c1:
    st.markdown(card(f"{away_nm} Win Prob", fmt_prob(gm.away_win_prob), f"ML: {gm.away_ml}"),
                unsafe_allow_html=True)
with c2:
    st.markdown(card("Spread", f"{gm.spread_home:+.1f}",
                     f"{gm.spread_home_odds}/{gm.spread_away_odds}"), unsafe_allow_html=True)
with c3:
    st.markdown(card("Total Line", str(gm.total_line),
                     f"O{gm.over_odds} / U{gm.under_odds}"), unsafe_allow_html=True)
with c4:
    st.markdown(card("Exp. Total", f"{gs.expected_total:.1f}", "sim median"),
                unsafe_allow_html=True)
with c5:
    st.markdown(card(f"{home_nm} Win Prob", fmt_prob(gm.home_win_prob), f"ML: {gm.home_ml}"),
                unsafe_allow_html=True)

fig_wp = go.Figure(go.Bar(
    x=[gm.away_win_prob * 100, gm.home_win_prob * 100],
    y=[away_nm, home_nm],
    orientation="h",
    marker_color=[team_color(away_id), team_color(home_id)],
    text=[f"{gm.away_win_prob*100:.1f}%", f"{gm.home_win_prob*100:.1f}%"],
    textposition="auto",
))
fig_wp.update_layout(
    height=110, margin=dict(l=0,r=0,t=2,b=0),
    xaxis=dict(range=[0,100], showticklabels=False, showgrid=False),
    yaxis=dict(showgrid=False),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f1f5f9"), showlegend=False,
)
st.plotly_chart(fig_wp, use_container_width=True)
st.markdown("---")

# ── Team stats table ──────────────────────────────────────────────────────
st.markdown("### Team Projections")

# Show any active adjustments as a note
if team_rating_overrides:
    parts = []
    for tid, ov in team_rating_overrides.items():
        for key, val in ov.items():
            lbl = TEAM_RATING_DEFS.get(key, {}).get("label", key)
            fmt = TEAM_RATING_DEFS.get(key, {}).get("fmt", "{:.3f}")
            parts.append(f"{team_name(tid)}: {lbl} → {fmt.format(val)}")
    st.markdown(
        f'<span class="rating-changed note-text">⚡ Active adjustments: {" · ".join(parts)}</span>',
        unsafe_allow_html=True,
    )

proj_df = pd.DataFrame([
    {"Team": away_nm, "Goals": round(ap.proj_goals,1), "Score": round(ap.proj_scores,1),
     "Shots": round(ap.proj_shots,1), "SOG": round(ap.proj_sog,1),
     "FO%": f"{ap.proj_faceoff_pct:.3f}", "FO Wins": round(ap.proj_faceoff_wins,1),
     "2PT": round(ap.proj_2pt_goals,2), "Assists": round(ap.proj_assists,1),
     "Saves": round(ap.proj_saves,1), "Save%": f"{ap.proj_save_pct:.3f}",
     "TOs": round(ap.proj_turnovers,1), "GBs": round(ap.proj_ground_balls,1)},
    {"Team": home_nm, "Goals": round(hp.proj_goals,1), "Score": round(hp.proj_scores,1),
     "Shots": round(hp.proj_shots,1), "SOG": round(hp.proj_sog,1),
     "FO%": f"{hp.proj_faceoff_pct:.3f}", "FO Wins": round(hp.proj_faceoff_wins,1),
     "2PT": round(hp.proj_2pt_goals,2), "Assists": round(hp.proj_assists,1),
     "Saves": round(hp.proj_saves,1), "Save%": f"{hp.proj_save_pct:.3f}",
     "TOs": round(hp.proj_turnovers,1), "GBs": round(hp.proj_ground_balls,1)},
]).set_index("Team")
st.dataframe(proj_df, use_container_width=True)
st.markdown("---")

# ── Sim distributions ─────────────────────────────────────────────────────
st.markdown("### Simulation Distributions  (20,000 sims)")
t1, t2, t3 = st.tabs(["Score Total", "Margin", "Goals by Team"])

with t1:
    tot = gs.total_distribution
    fig = go.Figure(go.Histogram(x=tot, nbinsx=40, marker_color="#3b82f6", opacity=0.7))
    fig.add_vline(x=gm.total_line, line_dash="dash", line_color="#f59e0b",
                  annotation_text=f"Line: {gm.total_line}")
    fig.update_layout(height=280, margin=dict(l=0,r=0,t=6,b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#f1f5f9"),
                      xaxis_title="Combined Score (incl. 2pt bonus)", yaxis_title="Sims")
    st.plotly_chart(fig, use_container_width=True)
    p_ov = float(np.mean(tot > gm.total_line))
    a, b, c = st.columns(3)
    a.metric("P(Over)",  f"{p_ov:.1%}")
    b.metric("P(Under)", f"{1-p_ov:.1%}")
    c.metric("P10 / P90", f"{np.percentile(tot,10):.0f} / {np.percentile(tot,90):.0f}")

with t2:
    mar = gs.margin_distribution
    fig2 = go.Figure(go.Histogram(x=mar, nbinsx=40, marker_color="#8b5cf6", opacity=0.7))
    fig2.add_vline(x=0, line_color="#64748b", line_dash="dash")
    fig2.add_vline(x=gm.spread_home, line_color="#f59e0b", line_dash="dot",
                   annotation_text=f"Priced spread: {gm.spread_home:+.1f}")
    fig2.update_layout(height=280, margin=dict(l=0,r=0,t=6,b=0),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#f1f5f9"),
                       xaxis_title=f"Margin (+ = {home_nm} wins)", yaxis_title="Sims")
    st.plotly_chart(fig2, use_container_width=True)

with t3:
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=gs.away_goals, name=away_nm,
                                marker_color=team_color(away_id), opacity=0.6, nbinsx=25))
    fig3.add_trace(go.Histogram(x=gs.home_goals, name=home_nm,
                                marker_color=team_color(home_id), opacity=0.6, nbinsx=25))
    fig3.update_layout(barmode="overlay", height=280, margin=dict(l=0,r=0,t=6,b=0),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#f1f5f9"),
                       xaxis_title="Goals", yaxis_title="Sims")
    st.plotly_chart(fig3, use_container_width=True)
    a, b, c, d = st.columns(4)
    a.metric(f"{away_nm} median", f"{np.median(gs.away_goals):.1f}")
    b.metric(f"{away_nm} P10/P90", f"{np.percentile(gs.away_goals,10):.0f}/{np.percentile(gs.away_goals,90):.0f}")
    c.metric(f"{home_nm} median", f"{np.median(gs.home_goals):.1f}")
    d.metric(f"{home_nm} P10/P90", f"{np.percentile(gs.home_goals,10):.0f}/{np.percentile(gs.home_goals,90):.0f}")

st.markdown("---")

# ── Player summary ────────────────────────────────────────────────────────
st.markdown("### Player Projection Summary")
st.markdown('<span class="note-text">Full prop lines → Player Props page. '
            'Roster adjustments → Depth Charts page.</span>', unsafe_allow_html=True)

for nm, players in [(away_nm, result.away_players), (home_nm, result.home_players)]:
    active = [p for p in players if p.active]
    if not active:
        continue
    st.markdown(f"**{nm}**")
    rows = [
        {"Player": p.full_name or p.player_id, "Pos": p.position,
         "Proj G": round(p.proj_goals,2), "Proj A": round(p.proj_assists,2),
         "Proj Pts": round(p.proj_points,2), "Proj Sh": round(p.proj_shots,1),
         "Proj SOG": round(p.proj_sog,1),
         "2PT Rate": f"{p.proj_2pt_goals/max(p.proj_goals,0.01):.0%}" if p.proj_goals > 0.05 else "—",
         "Proj SV": round(p.proj_saves,1) if p.position == "G" else "—"}
        for p in sorted(active, key=lambda x: x.proj_points, reverse=True)[:14]
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
