"""Page 3 -- Depth Charts"""
from __future__ import annotations

import sys
from pathlib import Path

_PAGES_DIR = Path(__file__).resolve().parent
_ROOT      = _PAGES_DIR.parent
for _p in [str(_ROOT), str(_PAGES_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

from _engine_state import (
    SHARED_CSS, PLAYER_RATING_DEFS,
    get_engine, init_session,
    team_name,
    get_depth_chart, set_player_override,
    set_player_rating,
    render_update_projection_btn,
)

st.set_page_config(page_title="Depth Charts · PLL", page_icon="🥍", layout="wide")
init_session()

# -- Extra CSS for compact depth chart layout --------------------------------
st.markdown(SHARED_CSS + """
<style>
.dc-group-header {
    font-size: .72rem; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; color: #64748b;
    margin: 10px 0 2px; padding: 3px 6px;
    border-left: 3px solid #334155;
    background: rgba(51,65,85,.18); border-radius: 0 4px 4px 0;
}
.dc-inactive { opacity: .45; }
.dc-proj { font-size: .82rem; color: #94a3b8; }
.dc-proj-hi { color: #34d399; font-weight: 600; }
.dc-modified { font-size: .70rem; color: #fbbf24; font-weight: 700; }
.dc-starter-badge {
    background: #0891b2; color: #fff; border-radius: 3px;
    padding: 1px 5px; font-size: .68rem; font-weight: 700;
}
.dc-roster-badge {
    display: inline-block; font-size: .70rem; font-weight: 600;
    padding: 2px 8px; border-radius: 10px; margin-bottom: 6px;
}
.dc-roster-gameday { background: #166534; color: #bbf7d0; }
.dc-roster-current  { background: #1e3a5f; color: #bae6fd; }
.dc-roster-fallback { background: #3f3f46; color: #d4d4d8; }
</style>
""", unsafe_allow_html=True)

engine = get_engine()
result = st.session_state.get("last_result")
if result is None:
    st.warning("No projection loaded. Go to **Projections** first and run a game.")
    st.stop()

home_id = result.home_proj.team_id
away_id = result.away_proj.team_id
home_nm = team_name(home_id)
away_nm = team_name(away_id)
game    = st.session_state.selected_game or {}

st.title("📋 Depth Charts")
st.markdown(
    f"**{away_nm} @ {home_nm}** · "
    f"Game {game.get('game_number','--')} · "
    f"{str(game.get('game_date',''))[:10]}"
)

# -- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Controls")
    render_update_projection_btn(engine, key="p3")

    st.markdown("---")
    st.markdown("### Roster source")
    filter_details = getattr(getattr(engine, "player_model", None),
                             "last_roster_filter_details", {}) or {}
    for tid in [home_id, away_id]:
        d = filter_details.get(tid, {})
        reason = d.get("reason", "unknown")
        count  = d.get("final_projection_roster_count", "?")
        if "gameday" in str(reason).lower():
            badge_cls, label = "dc-roster-gameday", "Gameday roster"
        elif "official_current" in str(reason).lower():
            badge_cls, label = "dc-roster-current", "Official current roster"
        else:
            badge_cls, label = "dc-roster-fallback", "Historical fallback"
        st.markdown(
            f'<div class="dc-roster-badge {badge_cls}">'
            f'{team_name(tid)}: {label} ({count} players)</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Bulk actions")
    bulk_team = st.radio("Team", [away_nm, home_nm], key="bulk_team", horizontal=True)
    bulk_tid  = away_id if bulk_team == away_nm else home_id

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Activate all", key="bulk_act", use_container_width=True):
            for p in (result.away_players if bulk_team == away_nm else result.home_players):
                set_player_override(bulk_tid, p.player_id, "active", True)
            st.rerun()
        if st.button("Reset usage", key="bulk_use", use_container_width=True):
            for p in (result.away_players if bulk_team == away_nm else result.home_players):
                set_player_override(bulk_tid, p.player_id, "usage_multiplier", 1.0)
            st.rerun()
    with b2:
        if st.button("Clear overrides", key="bulk_clr", use_container_width=True):
            st.session_state.depth_charts[bulk_tid] = {}
            st.rerun()

st.markdown("---")

# -- Position group ordering and labels -------------------------------------
POS_ORDER  = {"A": 0, "M": 1, "FO": 2, "SSDM": 3, "LSM": 4, "D": 5, "G": 6}
POS_LABELS = {
    "A": "Attack", "M": "Midfield", "FO": "Faceoff",
    "SSDM": "Short-Stick Def. Mid", "LSM": "Long-Stick Mid",
    "D": "Defense", "G": "Goalies",
}
ALL_POSITIONS = ["A", "M", "FO", "SSDM", "LSM", "D", "G"]


def _model_val_for(pid: str, key: str, p) -> float:
    """Get the model's current value for a rating key."""
    pm = engine.player_model
    if pm is not None and not pm.pr.empty:
        rows = pm.pr[pm.pr["player_id"] == pid]
        if not rows.empty and key in rows.columns:
            v = float(rows[key].iloc[-1])
            if v != 0.0:
                return v
    # Fallback from projection
    team_proj = result.home_proj if p.team_id == home_id else result.away_proj
    proj_map = {
        "share_goals_ewm":   p.proj_goals   / max(team_proj.proj_goals,   1.0),
        "share_assists_ewm": p.proj_assists / max(team_proj.proj_assists, 1.0),
        "two_pt_rate_ewm":   p.proj_2pt_goals / max(p.proj_goals, 0.01),
        "bayes_save_pct":    p.proj_save_pct,
        "bayes_fo_pct":      p.proj_faceoff_pct,
        "shot_pct_ewm":      p.proj_goals / max(p.proj_shots, 0.01),
    }
    return proj_map.get(key, 0.0)


def _effective_pos(p, dc: dict) -> str:
    """Return the position to display/group by, respecting any position override."""
    return dc.get(p.player_id, {}).get("position_override", p.position)


def _render_team(team_id: str, team_nm: str, players):
    dc = get_depth_chart(team_id)

    sorted_players = sorted(
        players,
        key=lambda p: (POS_ORDER.get(_effective_pos(p, dc), 9), -p.proj_points)
    )

    goalies = [p for p in sorted_players if p.position == "G"]
    current_starter = next(
        (p.player_id for p in goalies if dc.get(p.player_id, {}).get("is_starter", False)),
        max(goalies, key=lambda p: p.proj_save_pct).player_id if goalies else None,
    )

    # -- Column header row ---------------------------------------------------
    h = st.columns([3.5, 0.8, 0.7, 0.7, 1.2, 1.0, 1.0, 1.0, 0.8])
    for col, lbl in zip(h, ["Player", "Pos", "Active", "Start", "Usage ×",
                              "Proj G", "Proj A", "Proj Pts", ""]):
        col.markdown(f"<span style='font-size:.75rem;font-weight:700;color:#64748b;'>{lbl}</span>",
                     unsafe_allow_html=True)
    st.markdown(
        '<hr style="margin:3px 0 6px;border-color:rgba(148,163,184,.18);">',
        unsafe_allow_html=True,
    )

    current_group = None

    for p in sorted_players:
        pid       = p.player_id
        existing  = dc.get(pid, {})
        is_active = existing.get("active", True)
        usage_val = float(existing.get("usage_multiplier", 1.0))
        eff_pos   = _effective_pos(p, dc)
        is_goalie = eff_pos == "G"
        has_ov    = bool(existing.get("rating_overrides") or "position_override" in existing)
        nm        = p.full_name or pid

        # -- Position group header -------------------------------------------
        if eff_pos != current_group:
            current_group = eff_pos
            label = POS_LABELS.get(eff_pos, eff_pos)
            st.markdown(f'<div class="dc-group-header">{label}</div>',
                        unsafe_allow_html=True)

        # -- Player row -------------------------------------------------------
        opacity = "" if is_active else "dc-inactive"
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(
            [3.5, 0.8, 0.7, 0.7, 1.2, 1.0, 1.0, 1.0, 0.8]
        )

        # Name + badges
        with c1:
            name_style = "text-decoration:line-through;color:#475569;" if not is_active else ""
            starter_html = (' <span class="dc-starter-badge">STARTER</span>'
                            if is_goalie and current_starter == pid else "")
            mod_html = (' <span class="dc-modified">⚡</span>'
                        if has_ov or usage_val != 1.0 else "")
            st.markdown(
                f'<span style="{name_style}font-size:.88rem;">{nm}</span>'
                f'{starter_html}{mod_html}',
                unsafe_allow_html=True,
            )

        # Position selector (shows current effective position; allows override)
        with c2:
            pos_idx = ALL_POSITIONS.index(eff_pos) if eff_pos in ALL_POSITIONS else 0
            new_pos = st.selectbox(
                "",
                options=ALL_POSITIONS,
                index=pos_idx,
                key=f"pos_{team_id}_{pid}",
                label_visibility="collapsed",
            )
            if new_pos != p.position:
                set_player_override(team_id, pid, "position_override", new_pos)
            elif "position_override" in existing and new_pos == p.position:
                # User reset back to native position — clear the override
                del st.session_state.depth_charts[team_id][pid]["position_override"]

        # Active checkbox
        with c3:
            new_active = st.checkbox(
                "", value=is_active,
                key=f"act_{team_id}_{pid}",
                label_visibility="collapsed",
            )
            if new_active != is_active:
                set_player_override(team_id, pid, "active", new_active)
                new_usage_val = 0.0 if not new_active else 1.0
                set_player_override(team_id, pid, "usage_multiplier", new_usage_val)
                # Force the number_input widget to show the new value immediately
                # by writing directly to its session state key
                st.session_state[f"use_{team_id}_{pid}"] = new_usage_val

        # Starter checkbox (goalies only)
        with c4:
            if is_goalie:
                is_starter_now = (current_starter == pid)
                new_starter = st.checkbox(
                    "", value=is_starter_now,
                    key=f"start_{team_id}_{pid}",
                    label_visibility="collapsed",
                )
                if new_starter and not is_starter_now:
                    for g in goalies:
                        set_player_override(team_id, g.player_id, "is_starter", False)
                    set_player_override(team_id, pid, "is_starter", True)
                    current_starter = pid

        # Usage multiplier
        with c5:
            new_usage = st.number_input(
                "", min_value=0.0, max_value=2.5, step=0.05,
                value=usage_val,
                key=f"use_{team_id}_{pid}",
                label_visibility="collapsed",
                disabled=not is_active,
                help="1.0=normal · 1.3=elevated · 0.7=limited · 0.0=inactive",
            )
            if abs(new_usage - usage_val) > 0.001:
                set_player_override(team_id, pid, "usage_multiplier", new_usage)

        # Projected stats (compact)
        color_g = "#34d399" if p.proj_goals > 1.0 else "#94a3b8"
        color_p = "#34d399" if p.proj_points > 1.5 else "#94a3b8"
        with c6:
            st.markdown(
                f'<span style="font-size:.82rem;color:{color_g};">'
                f'{"--" if not is_active else f"{p.proj_goals:.2f}"}</span>',
                unsafe_allow_html=True,
            )
        with c7:
            st.markdown(
                f'<span style="font-size:.82rem;color:#94a3b8;">'
                f'{"--" if not is_active else f"{p.proj_assists:.2f}"}</span>',
                unsafe_allow_html=True,
            )
        with c8:
            if eff_pos == "G":
                lbl = f"{p.proj_saves:.1f}sv" if is_active else "--"
                st.markdown(f'<span style="font-size:.82rem;color:#94a3b8;">{lbl}</span>',
                            unsafe_allow_html=True)
            elif eff_pos == "FO":
                lbl = f"{p.proj_faceoff_wins:.1f}fw" if is_active else "--"
                st.markdown(f'<span style="font-size:.82rem;color:#94a3b8;">{lbl}</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<span style="font-size:.82rem;color:{color_p};">'
                    f'{"--" if not is_active else f"{p.proj_points:.2f}pts"}</span>',
                    unsafe_allow_html=True,
                )

        # Rating override toggle button
        with c9:
            rating_key = f"show_ratings_{team_id}_{pid}"
            if rating_key not in st.session_state:
                st.session_state[rating_key] = False
            if is_active:
                btn_label = "⚡ Edit" if has_ov else "Edit"
                if st.button(btn_label, key=f"rbtn_{team_id}_{pid}",
                             use_container_width=True):
                    st.session_state[rating_key] = not st.session_state[rating_key]

        # -- Rating override panel (shown inline when toggled) ---------------
        if is_active and st.session_state.get(f"show_ratings_{team_id}_{pid}", False):
            rating_overrides = existing.get("rating_overrides", {})
            pos = eff_pos  # use effective (possibly overridden) position for rating filtering

            pos_label = POS_LABELS.get(eff_pos, eff_pos)
            pos_note = (f" · playing as {pos_label}" if eff_pos != p.position else "")

            with st.container():
                st.markdown(
                    f'<div style="background:rgba(30,58,95,.25);border-left:3px solid #0891b2;'
                    f'border-radius:0 6px 6px 0;padding:8px 12px;margin:2px 0 6px;">'
                    f'<span style="font-size:.75rem;color:#7dd3fc;font-weight:700;">'
                    f'Rating overrides — {nm}{pos_note}</span></div>',
                    unsafe_allow_html=True,
                )
                ratings_shown = False
                for key, meta in PLAYER_RATING_DEFS.items():
                    if pos not in meta.get("positions", []):
                        continue

                    model_val = _model_val_for(pid, key, p)
                    wgt_key   = f"pr_num_{team_id}_{pid}_{key}"

                    # Seed the widget's session state from saved rating_overrides
                    # the first time this panel opens, or after a reset.
                    # We only write to st.session_state[wgt_key] when it doesn't
                    # exist yet so we never overwrite a value the user just typed.
                    if wgt_key not in st.session_state:
                        seed_val = rating_overrides.get(key, model_val)
                        st.session_state[wgt_key] = float(
                            min(max(float(seed_val), meta["min"]), meta["max"])
                        )

                    def _on_change(t=team_id, p_=pid, k=key, wk=wgt_key, mn=meta["min"], mx=meta["max"], mv=model_val, stp=meta["step"]):
                        raw = st.session_state.get(wk, mv)
                        val = float(min(max(float(raw), mn), mx))
                        if abs(val - mv) > stp * 0.5:
                            set_player_rating(t, p_, k, val)
                        else:
                            dc_ = get_depth_chart(t)
                            if p_ in dc_ and k in dc_[p_].get("rating_overrides", {}):
                                del st.session_state.depth_charts[t][p_]["rating_overrides"][k]

                    rc1, rc2 = st.columns([3, 1])
                    with rc1:
                        st.number_input(
                            meta["label"],
                            min_value=meta["min"], max_value=meta["max"],
                            step=meta["step"],
                            help=meta["help"],
                            key=wgt_key,
                            on_change=_on_change,
                        )
                    with rc2:
                        current_val = float(st.session_state.get(wgt_key, model_val))
                        changed = abs(current_val - model_val) > meta["step"] * 0.5
                        color   = "#fbbf24" if changed else "#64748b"
                        label   = ("→ " + meta["fmt"].format(current_val)) if changed else ("model: " + meta["fmt"].format(model_val))
                        st.markdown(
                            f'<div style="padding-top:28px;">'
                            f'<span style="font-size:.72rem;color:{color};">{label}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    ratings_shown = True

                if not ratings_shown:
                    st.caption(f"No adjustable ratings for {POS_LABELS.get(pos, pos)}.")

                col_rst, col_close = st.columns(2)
                with col_rst:
                    if st.button(f"Reset ratings", key=f"rst_p_{team_id}_{pid}"):
                        if pid in st.session_state.depth_charts.get(team_id, {}):
                            st.session_state.depth_charts[team_id][pid].pop(
                                "rating_overrides", None
                            )
                        # Clear widget session state so inputs reseed from model values
                        for k2 in PLAYER_RATING_DEFS:
                            wk2 = f"pr_num_{team_id}_{pid}_{k2}"
                            if wk2 in st.session_state:
                                del st.session_state[wk2]
                        st.session_state[f"show_ratings_{team_id}_{pid}"] = False
                        st.rerun()
                with col_close:
                    if st.button("Close", key=f"close_r_{team_id}_{pid}"):
                        st.session_state[f"show_ratings_{team_id}_{pid}"] = False
                        st.rerun()

    st.markdown("")


# -- Render teams ------------------------------------------------------------
tab_away, tab_home = st.tabs([f"📋 {away_nm}", f"📋 {home_nm}"])

with tab_away:
    _render_team(away_id, away_nm, result.away_players)

with tab_home:
    _render_team(home_id, home_nm, result.home_players)

st.markdown("---")
st.markdown(
    '<span class="note-text">'
    'Active/usage changes apply on next 🔄 Update Projection. '
    'Pos dropdown overrides a player\'s position (e.g. Attack → Midfield) — changes projections on next update. '
    'Edit button opens inline rating overrides per player. '
    '⚡ indicates a player has active overrides.'
    '</span>',
    unsafe_allow_html=True,
)
