"""Page 1 -- Game Projections"""
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
    build_overrides, build_active_players, build_starter_goalies,
    get_team_rating_overrides, set_team_rating_override,
    sorted_upcoming, default_game_index,
    render_update_projection_btn,
    session_to_json, session_from_json,
)

st.set_page_config(page_title="Projections · PLL", page_icon="🥍", layout="wide")
init_session()
st.markdown(SHARED_CSS, unsafe_allow_html=True)

engine   = get_engine()
raw_games = engine.upcoming_games()

# -- Attach season to each game dict (for sorting) -------------------------
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

# -- Sidebar ---------------------------------------------------------------
with st.sidebar:

    # -- Game selector -----------------------------------------------------
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

    # Build display labels: "Game XX · Away @ Home · Date"
    def _game_label(g: dict) -> str:
        ht   = team_name(g.get("home_team_id",""))
        at   = team_name(g.get("away_team_id",""))
        gnum = g.get("game_number","?")
        gdate = str(g.get("game_date",""))[:10]
        return f"Game {gnum} · {at} @ {ht} · {gdate}"

    game_labels = [_game_label(g) for g in season_games]

    # Persist selected game across page navigations
    persisted = st.session_state.get("selected_game")
    default_idx = 0
    if isinstance(persisted, dict):
        for i, g in enumerate(season_games):
            if (g.get("home_team_id") == persisted.get("home_team_id") and
                    g.get("away_team_id") == persisted.get("away_team_id")):
                default_idx = i
                break
    else:
        # find next upcoming
        for i, g in enumerate(season_games):
            try:
                if dt.date.fromisoformat(str(g.get("game_date",""))[:10]) >= today:
                    default_idx = i
                    break
            except Exception:
                pass

    game_idx = st.selectbox(
        "Game",
        options=range(len(season_games)),
        format_func=lambda i: game_labels[i],
        index=default_idx,
        key="game_idx_p1",
    )
    game = season_games[game_idx]

    # Clear result when game changes
    prev = st.session_state.get("selected_game") or {}
    if (game.get("home_team_id") != prev.get("home_team_id") or
            game.get("away_team_id") != prev.get("away_team_id")):
        st.session_state.last_result = None
    st.session_state.selected_game = game

    home_id = str(game.get("home_team_id",""))
    away_id = str(game.get("away_team_id",""))
    home_nm = team_name(home_id)
    away_nm = team_name(away_id)

    st.markdown("---")

    # -- Team rating adjustments -------------------------------------------
    st.markdown("### Team Rating Adjustments")
    st.markdown(
        '<span class="note-text">'
        "Adjust the model's actual input ratings. Each slider shows the model's "
        "current value -- move it to reflect information the stats don't capture "
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
        """Render number inputs for one team's key ratings."""
        st.markdown(f"**{team_nm}**")
        overrides = get_team_rating_overrides(team_id)

        for key, meta in TEAM_RATING_DEFS.items():
            model_val = float(current_ratings.get(key, 0.0))
            if model_val == 0.0:
                continue  # skip if rating not available

            current_override = overrides.get(key, model_val)

            new_val = st.number_input(
                meta["label"],
                min_value=meta["min"],
                max_value=meta["max"],
                value=float(current_override),
                step=meta["step"],
                help=meta["help"],
                key=f"tr_num_{team_id}_{key}",
            )

            # Show model value as reference
            changed = abs(new_val - model_val) > meta["step"] * 0.5
            model_str = meta["fmt"].format(model_val)
            if changed:
                st.markdown(
                    f'<span class="rating-changed note-text">Model: {model_str} &rarr; '
                    f'You: {meta["fmt"].format(new_val)}</span>',
                    unsafe_allow_html=True,
                )
                set_team_rating_override(team_id, key, new_val)
            else:
                st.markdown(
                    f'<span class="note-text">Model value: {model_str}</span>',
                    unsafe_allow_html=True,
                )
                # Remove override if it matches model (user reset to model value)
                if key in get_team_rating_overrides(team_id):
                    del st.session_state.team_rating_overrides[team_id][key]

    _rating_sliders(home_id, home_nm + " (Home)", hf_current)
    st.markdown("")
    _rating_sliders(away_id, away_nm + " (Away)", af_current)

    st.markdown("---")

    # Hold % — single number input, synced globally via session state
    hold_pct_pct = st.number_input(
        "Market margin %",
        min_value=2.0, max_value=15.0,
        value=float(st.session_state.get("hold_pct", 0.075) * 100),
        step=0.5,
        key="hold_num_p1",
        help="Vig/margin applied to all priced markets. 7.5% = standard sportsbook. Updates across all pages.",
    )
    hold_pct = hold_pct_pct / 100.0
    st.session_state.hold_pct = hold_pct

    if st.button("Reset all adjustments", key="reset_adj"):
        st.session_state.team_rating_overrides = {}
        st.rerun()

    run_btn = st.button("▶  Run Projection", type="primary", use_container_width=True)

    # Update Projection button (also reruns when clicked)
    render_update_projection_btn(engine, key="p1")

    # -- Save / Load session state -----------------------------------------
    st.markdown("---")
    st.markdown("### Save / Load")
    st.markdown('<span class="note-text">Save all overrides, depth chart, and selected game to a file. Load it back anytime to restore instantly.</span>',
                unsafe_allow_html=True)

    # Save button -- generates JSON download
    save_data = session_to_json()
    _game = st.session_state.get("selected_game") or {}
    _gnum = _game.get("game_number", "?")
    _gdate = str(_game.get("game_date", ""))[:10].replace("-", "")
    _save_name = f"PLL_session_game{_gnum}_{_gdate}.json"
    st.download_button(
        label="💾 Save session",
        data=save_data,
        file_name=_save_name,
        mime="application/json",
        key="save_session_btn",
        use_container_width=True,
        help="Downloads a JSON file with your current game selection, depth chart, and rating overrides.",
    )

    # Load from uploaded file
    uploaded = st.file_uploader("Load session", type="json",
                                 key="load_session_file",
                                 help="Upload a previously saved session file to restore all settings.",
                                 label_visibility="collapsed")
    if uploaded is not None:
        try:
            json_str = uploaded.read().decode("utf-8")
            if session_from_json(json_str):
                st.success("Session restored. Click Run Projection to reload.")
                st.rerun()
            else:
                st.error("File format not recognised.")
        except Exception as _e:
            st.error(f"Could not load: {_e}")

# -- Run projection --------------------------------------------------------
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
            game_date=game.get("game_date"),
            player_overrides=build_overrides() or None,
            active_players=build_active_players() or None,
            starter_goalies=build_starter_goalies() or None,
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

# -- Game header -----------------------------------------------------------
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

# -- Win probability row ---------------------------------------------------
# home gets + when underdog (home_displayed_spd = -spread_home)
# away lays - when favored (away_spd_display = spread_home)
home_spd_display = -gm.spread_home   # home gets + when underdog
away_spd_display = gm.spread_home    # away lays - when favored

c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 2])
with c1:
    st.markdown(card(f"{away_nm} Win Prob", fmt_prob(gm.away_win_prob),
        f"ML: {gm.away_ml}  ·  Spread: {away_spd_display:+.1f} ({gm.spread_away_odds})"),
        unsafe_allow_html=True)
with c2:
    st.markdown(card("Total", f"{gm.total_line:.1f}",
        f"O{gm.over_odds} / U{gm.under_odds}"), unsafe_allow_html=True)
with c3:
    st.markdown(card("Exp Total", f"{gs.expected_total:.1f}", "sim median"),
        unsafe_allow_html=True)
with c4:
    st.markdown(card("Margin", f"{gm.spread_home:+.1f}", f"home perspective"),
        unsafe_allow_html=True)
with c5:
    st.markdown(card(f"{home_nm} Win Prob", fmt_prob(gm.home_win_prob),
        f"ML: {gm.home_ml}  ·  Spread: {home_spd_display:+.1f} ({gm.spread_home_odds})"),
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

# -- Team stats table ------------------------------------------------------
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

# -- Sim distributions -----------------------------------------------------
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

# -- Player summary --------------------------------------------------------
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
         "2PT Rate": f"{p.proj_2pt_goals/max(p.proj_goals,0.01):.0%}" if p.proj_goals > 0.05 else "--",
         "Proj SV": round(p.proj_saves,1) if p.position == "G" else "--"}
        for p in sorted(active, key=lambda x: x.proj_points, reverse=True)[:14]
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# -- Export -------------------------------------------------------------------
st.markdown("---")
st.markdown("### Download Projection Package")

import io, datetime as _dt

def _build_export(result, game, hold_pct, engine):
    """Build a multi-tab Excel export of the current projection."""
    from projection_engine_v3 import PricingEngine
    pricing = PricingEngine(hold_pct=hold_pct)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:

        # ── Tab 1: Metadata ────────────────────────────────────────────────
        pm = engine.player_model
        filter_details = getattr(pm, "last_roster_filter_details", {}) or {}
        h_src = filter_details.get(result.home_proj.team_id, {}).get("reason", "unknown")
        a_src = filter_details.get(result.away_proj.team_id, {}).get("reason", "unknown")
        meta_rows = [
            ("Game",         f"{team_name(result.away_proj.team_id)} @ {team_name(result.home_proj.team_id)}"),
            ("Game Number",  game.get("game_number", "")),
            ("Game Date",    str(game.get("game_date", ""))[:10]),
            ("Home Team",    team_name(result.home_proj.team_id)),
            ("Away Team",    team_name(result.away_proj.team_id)),
            ("Generated At", _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")),
            ("Hold %",       f"{hold_pct*100:.1f}%"),
            ("Sims",         result.game_sim.n_sims),
            ("Home Roster Source", h_src),
            ("Away Roster Source", a_src),
            ("Model",        result.home_proj.model_used),
        ]
        pd.DataFrame(meta_rows, columns=["Field", "Value"]).to_excel(
            xl, sheet_name="Metadata", index=False)

        # ── Tab 2: Game Lines ──────────────────────────────────────────────
        gs = result.game_sim
        gm = result.game_market
        import numpy as np
        home_tt_line = round(float(np.median(gs.home_scores)) * 2) / 2
        away_tt_line = round(float(np.median(gs.away_scores)) * 2) / 2
        # snap to x.5
        def snap(v):
            return round(round(v * 2) / 2 + (0 if round(v*2)%2==1 else 0.5 if v-int(v)<0.5 else -0.5), 1)
        home_tt = snap(home_tt_line) if (home_tt_line % 1) == 0 else home_tt_line
        away_tt = snap(away_tt_line) if (away_tt_line % 1) == 0 else away_tt_line

        lines_rows = [
            (f"{team_name(result.away_proj.team_id)} ML", "--", gm.away_ml,
             f"{gm.away_win_prob*100:.1f}%"),
            (f"{team_name(result.home_proj.team_id)} ML", "--", gm.home_ml,
             f"{gm.home_win_prob*100:.1f}%"),
            (f"{team_name(result.away_proj.team_id)} Spread", f"{gm.spread_home:+.1f}",
             gm.spread_away_odds, "--"),
            (f"{team_name(result.home_proj.team_id)} Spread", f"{-gm.spread_home:+.1f}",
             gm.spread_home_odds, "--"),
            ("Total Over",  f"{gm.total_line:.1f}", gm.over_odds,
             f"{float(np.mean(gs.total_distribution>gm.total_line))*100:.1f}%"),
            ("Total Under", f"{gm.total_line:.1f}", gm.under_odds,
             f"{float(np.mean(gs.total_distribution<=gm.total_line))*100:.1f}%"),
            (f"{team_name(result.home_proj.team_id)} Team Total O", f"{home_tt:.1f}", "--",
             f"{float(np.mean(gs.home_scores>home_tt))*100:.1f}%"),
            (f"{team_name(result.home_proj.team_id)} Team Total U", f"{home_tt:.1f}", "--",
             f"{float(np.mean(gs.home_scores<=home_tt))*100:.1f}%"),
            (f"{team_name(result.away_proj.team_id)} Team Total O", f"{away_tt:.1f}", "--",
             f"{float(np.mean(gs.away_scores>away_tt))*100:.1f}%"),
            (f"{team_name(result.away_proj.team_id)} Team Total U", f"{away_tt:.1f}", "--",
             f"{float(np.mean(gs.away_scores<=away_tt))*100:.1f}%"),
        ]
        pd.DataFrame(lines_rows,
                     columns=["Market", "Line", "Odds", "Fair Prob"]
                     ).to_excel(xl, sheet_name="Game Lines", index=False)

        # ── Tab 3: Player Props ────────────────────────────────────────────
        all_players = {p.player_id: p
                       for p in result.home_players + result.away_players}
        markets = result.player_markets
        sims_all = result.home_player_sims + result.away_player_sims
        prop_rows = []
        STAT_LABELS = {"goals":"Goals","assists":"Assists","points":"Points",
                       "shots_on_goal":"SOG","saves":"Saves","faceoff_wins":"FO Wins"}

        for ps in sims_all:
            proj = all_players.get(ps.player_id)
            if proj is None or not proj.active:
                continue
            pm_data = markets.get(ps.player_id, {})
            pv = pm_data.get("proj_values", {})
            ms = pm_data.get("markets", {})

            stats = (["saves"] if proj.position == "G"
                     else ["faceoff_wins"] if proj.position == "FO"
                     else ["goals", "assists", "points", "shots_on_goal"])
            for stat in stats:
                if stat not in ps.stat_distributions:
                    continue
                mkt = ms.get(stat, {})
                proj_val = round(float(pv.get(stat, 0)), 3)
                if proj_val < 0.05 and proj.position not in ("G","FO"):
                    continue
                prop_rows.append({
                    "Player":       proj.full_name or proj.player_id,
                    "Team":         team_name(proj.team_id),
                    "Pos":          proj.position,
                    "Stat":         STAT_LABELS.get(stat, stat),
                    "Projection":   proj_val,
                    "Main Line":    mkt.get("line", ""),
                    "Over Odds":    mkt.get("over_odds", ""),
                    "Under Odds":   mkt.get("under_odds", ""),
                    "Fair P(Over)": round(float(mkt.get("fair_over_prob", 0)), 3),
                    "P10":  round(float(np.percentile(ps.stat_distributions[stat], 10)), 2),
                    "P50":  round(float(np.percentile(ps.stat_distributions[stat], 50)), 2),
                    "P90":  round(float(np.percentile(ps.stat_distributions[stat], 90)), 2),
                    "Actual Result": "",   # fill in later for tracking
                    "Hit/Miss":      "",
                })

        (pd.DataFrame(prop_rows)
           .sort_values(["Team","Pos","Player","Stat"])
           .to_excel(xl, sheet_name="Player Props", index=False))

        # ── Tab 4: Depth Chart (active roster used) ────────────────────────
        depth_rows = []
        for tid, players in [
            (result.home_proj.team_id, result.home_players),
            (result.away_proj.team_id, result.away_players),
        ]:
            for p in sorted(players, key=lambda x: (x.position, -x.proj_points)):
                depth_rows.append({
                    "Team":       team_name(tid),
                    "Player":     p.full_name or p.player_id,
                    "Pos":        p.position,
                    "Active":     "Yes" if p.active else "No",
                    "Starter":    "Yes" if p.is_starter else "",
                    "Usage Mult": round(p.usage_multiplier, 2),
                    "Proj Goals": round(p.proj_goals, 3) if p.active else 0,
                    "Proj Asst":  round(p.proj_assists, 3) if p.active else 0,
                    "Proj Pts":   round(p.proj_points, 3) if p.active else 0,
                    "Proj Saves": round(p.proj_saves, 1) if p.position == "G" else "",
                    "Proj FOW":   round(p.proj_faceoff_wins, 1) if p.position == "FO" else "",
                })
        pd.DataFrame(depth_rows).to_excel(xl, sheet_name="Depth Chart", index=False)

        # ── Tab 5: Team Projections ────────────────────────────────────────
        team_rows = []
        for proj in [result.home_proj, result.away_proj]:
            team_rows.append({
                "Team":     team_name(proj.team_id),
                "Goals":    round(proj.proj_goals, 2),
                "Score":    round(proj.proj_scores, 2),
                "Shots":    round(proj.proj_shots, 1),
                "SOG":      round(proj.proj_sog, 1),
                "FO%":      round(proj.proj_faceoff_pct, 3),
                "FO Wins":  round(proj.proj_faceoff_wins, 1),
                "Assists":  round(proj.proj_assists, 1),
                "Saves":    round(proj.proj_saves, 1),
                "Save%":    round(proj.proj_save_pct, 3),
                "2PT Goals":round(proj.proj_2pt_goals, 2),
                "TOs":      round(proj.proj_turnovers, 1),
                "GBs":      round(proj.proj_ground_balls, 1),
            })
        pd.DataFrame(team_rows).to_excel(xl, sheet_name="Team Projections", index=False)

    buf.seek(0)
    return buf.getvalue()


game_date_str = str(game.get("game_date", ""))[:10].replace("-", "")
fname = (f"PLL_{team_name(away_id)}_{team_name(home_id)}_"
         f"Game{game.get('game_number','?')}_{game_date_str}.xlsx")

if st.button("📥 Download Projection Package (Excel)", type="secondary",
             use_container_width=False):
    with st.spinner("Building Excel export..."):
        try:
            xlsx_bytes = _build_export(result, game, hold_pct, engine)
            st.download_button(
                label="⬇ Click to download",
                data=xlsx_bytes,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_xlsx",
            )
            st.success(f"Ready: {fname}")
        except Exception as e:
            st.error(f"Export failed: {e}")

st.markdown(
    '<span class="note-text">Export includes: Game Lines · Player Props · '
    'Depth Chart · Team Projections · Metadata. '
    'Fill in "Actual Result" column after games to track model accuracy.</span>',
    unsafe_allow_html=True,
)

# -- Roster source info -------------------------------------------------------
with st.expander("Roster source details", expanded=False):
    pm = engine.player_model
    if pm is not None:
        filter_details = getattr(pm, "last_roster_filter_details", {})
        for tid in [home_id, away_id]:
            d = filter_details.get(tid, {})
            src = d.get("reason", "unknown")
            count = d.get("final_projection_roster_count", "?")
            synthetic = d.get("synthetic_current_roster_added", 0)
            st.write(f"**{team_name(tid)}**: {count} players | source: {src}"
                     + (f" | {synthetic} new additions (no historical data)" if synthetic else ""))
    else:
        st.write("Engine not loaded")
