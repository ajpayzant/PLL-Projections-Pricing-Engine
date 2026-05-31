"""Page 3 — Depth Charts"""
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
    SHARED_CSS, PLAYER_RATING_DEFS, pos_badge,
    get_engine, init_session,
    team_name,
    get_depth_chart, set_player_override,
    set_player_rating,
)

st.set_page_config(page_title="Depth Charts · PLL", page_icon="🥍", layout="wide")
init_session()
st.markdown(SHARED_CSS, unsafe_allow_html=True)

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

st.title("📋 Depth Charts & Player Ratings")
st.markdown(f"**{away_nm} @ {home_nm}** · Game {game.get('game_number','—')}")

st.info(
    "**How this works:** Adjust rosters and individual player ratings below. "
    "Go back to **Projections** and click **▶ Run Projection** to apply all changes. "
    "Only players marked as active appear in projections. "
    "The model already filtered to each team's current-season roster — "
    "if a player looks wrong, use the Active toggle to remove them."
)

st.markdown("---")

# ── Official roster filter status ──────────────────────────────────────────
with st.expander("Official roster filter status", expanded=False):
    status = getattr(engine, "current_rosters_status", {}) or {}
    details = getattr(getattr(engine, "player_model", None), "last_roster_filter_details", {}) or {}

    available = bool(status.get("available", False))
    source = status.get("source", "unknown")
    path = status.get("path", "")
    reason = status.get("reason", "")

    if available:
        st.success(f"Official roster cache loaded — source: {source}")
    else:
        st.warning(f"Official roster cache unavailable — {reason or source}")

    if path:
        st.code(str(path))

    teams = status.get("teams", {})
    if teams:
        team_rows = pd.DataFrame([
            {"Team ID": k, "Official roster players": v}
            for k, v in sorted(teams.items())
        ])
        st.dataframe(team_rows, use_container_width=True, hide_index=True)

    if details:
        detail_rows = []
        for tid, d in details.items():
            detail_rows.append({
                "Team ID": tid,
                "Applied": d.get("applied"),
                "Reason": d.get("reason"),
                "Official Count": d.get("official_roster_count"),
                "Historical Candidates": d.get("historical_candidate_count"),
                "Matched": d.get("matched_count"),
                "Position Corrections": d.get("official_position_corrections"),
                "Name Corrections": d.get("official_name_corrections"),
                "Synthetic Added": d.get("synthetic_current_roster_added"),
                "Final Count": d.get("final_projection_roster_count"),
            })
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Run a projection once to populate per-team roster filter details.")

POS_ORDER = {"A": 0, "M": 1, "FO": 2, "D": 3, "SSDM": 4, "LSM": 5, "G": 6}


def _render_team(team_id: str, team_nm: str, players):
    st.markdown(f"## {team_nm}")
    dc = get_depth_chart(team_id)

    # Filter to active players only for display (inactive still shown but greyed)
    sorted_players = sorted(
        players,
        key=lambda p: (POS_ORDER.get(p.position, 9), -p.proj_points)
    )

    goalies = [p for p in sorted_players if p.position == "G"]
    current_starter = next(
        (p.player_id for p in goalies if dc.get(p.player_id, {}).get("is_starter", False)),
        max(goalies, key=lambda p: p.proj_save_pct).player_id if goalies else None,
    )

    # ── Column headers ─────────────────────────────────────────────────────
    hdr = st.columns([3, 1, 1, 1, 2])
    for col, lbl in zip(hdr, ["Player", "Pos", "Active", "Starter (G)", "Usage"]):
        col.markdown(f"**{lbl}**")
    st.markdown(
        '<hr style="margin:4px 0 8px;border-color:rgba(148,163,184,.15);">',
        unsafe_allow_html=True,
    )

    for p in sorted_players:
        pid       = p.player_id
        existing  = dc.get(pid, {})
        is_active = existing.get("active", True)
        usage_val = float(existing.get("usage_multiplier", 1.0))
        is_goalie = p.position == "G"
        nm        = p.full_name or pid

        c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 2])

        with c1:
            style = "" if is_active else "color:#64748b;text-decoration:line-through;"
            st.markdown(f'<span style="{style}">{nm}</span>', unsafe_allow_html=True)
        with c2:
            st.markdown(pos_badge(p.position), unsafe_allow_html=True)
        with c3:
            new_active = st.checkbox(
                "", value=is_active,
                key=f"act_{team_id}_{pid}",
                label_visibility="collapsed",
            )
            if new_active != is_active:
                set_player_override(team_id, pid, "active", new_active)
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
            else:
                st.write("")
        with c5:
            new_usage = st.number_input(
                "", min_value=0.0, max_value=2.5, step=0.05,
                value=usage_val,
                key=f"use_{team_id}_{pid}",
                label_visibility="collapsed",
                disabled=not is_active,
                help=(
                    "Usage multiplier applies proportionally to all of this player's stats. "
                    "1.0 = normal. 1.3 = ~30% more involvement (e.g. star is carrying the offense). "
                    "0.7 = limited role (playing through injury, coming off the bench). "
                    "0.0 = effectively inactive."
                ),
            )
            if abs(new_usage - usage_val) > 0.001:
                set_player_override(team_id, pid, "usage_multiplier", new_usage)

        # ── Individual rating adjustments (expandable) ────────────────────
        if is_active:
            rating_overrides = existing.get("rating_overrides", {})
            has_overrides = bool(rating_overrides)

            with st.expander(
                f"⚙ Adjust {nm}'s ratings" + (" ⚡ modified" if has_overrides else ""),
                expanded=False,
            ):
                st.markdown(
                    "Adjust the model's input ratings for this specific player. "
                    "**Each slider shows the model's current estimate** — move it to reflect "
                    "information that isn't yet in the stats "
                    "(hot streak, injury recovery, matchup advantage, etc.)."
                )

                pos = p.position
                ratings_shown = False

                for key, meta in PLAYER_RATING_DEFS.items():
                    # Only show ratings relevant to this position
                    if pos not in meta.get("positions", []):
                        continue

                    # Get the model's current value for this player
                    # Look it up from the player model's rating store
                    pm = engine.player_model
                    model_val: float = 0.0
                    if pm is not None and not pm.pr.empty:
                        rows = pm.pr[pm.pr["player_id"] == pid]
                        if not rows.empty and key in rows.columns:
                            model_val = float(rows[key].iloc[-1])

                    if model_val == 0.0 and key not in ("bayes_fo_pct", "bayes_save_pct"):
                        # Fall back to the projection values for display
                        proj_map = {
                            "share_goals_ewm": p.proj_goals / max(result.home_proj.proj_goals
                                               if team_id == home_id
                                               else result.away_proj.proj_goals, 1.0),
                            "share_assists_ewm": p.proj_assists / max(result.home_proj.proj_assists
                                                  if team_id == home_id
                                                  else result.away_proj.proj_assists, 1.0),
                            "two_pt_rate_ewm": p.proj_2pt_goals / max(p.proj_goals, 0.01),
                        }
                        model_val = proj_map.get(key, 0.0)

                    current_ov = rating_overrides.get(key, model_val)
                    # Clamp to slider range
                    clamped = min(max(float(current_ov), meta["min"]), meta["max"])

                    new_val = st.slider(
                        label=meta["label"],
                        min_value=meta["min"],
                        max_value=meta["max"],
                        value=clamped,
                        step=meta["step"],
                        help=meta["help"],
                        key=f"pr_{team_id}_{pid}_{key}",
                    )

                    model_str = meta["fmt"].format(model_val)
                    changed   = abs(new_val - model_val) > meta["step"] * 0.5
                    if changed:
                        st.markdown(
                            f'<span class="rating-changed note-text">'
                            f'Model: {model_str} → You: {meta["fmt"].format(new_val)}'
                            f'</span>',
                            unsafe_allow_html=True,
                        )
                        set_player_rating(team_id, pid, key, new_val)
                    else:
                        st.markdown(
                            f'<span class="note-text">Model value: {model_str}</span>',
                            unsafe_allow_html=True,
                        )
                        # Remove override if user slid back to model value
                        if key in (dc.get(pid, {}).get("rating_overrides", {})):
                            del st.session_state.depth_charts[team_id][pid]["rating_overrides"][key]

                    ratings_shown = True

                if not ratings_shown:
                    st.markdown(
                        f'<span class="note-text">No adjustable ratings for position {pos}.</span>',
                        unsafe_allow_html=True,
                    )

                if st.button(f"Reset {nm}'s ratings", key=f"rst_p_{team_id}_{pid}"):
                    if pid in st.session_state.depth_charts.get(team_id, {}):
                        st.session_state.depth_charts[team_id][pid].pop("rating_overrides", None)
                    st.rerun()

    st.markdown("")

    # ── Bulk actions ───────────────────────────────────────────────────────
    with st.expander(f"Bulk actions — {team_nm}"):
        ca, cb, cc = st.columns(3)
        with ca:
            if st.button(f"Activate all", key=f"act_all_{team_id}"):
                for pl in sorted_players:
                    set_player_override(team_id, pl.player_id, "active", True)
                st.rerun()
        with cb:
            if st.button(f"Reset all usage", key=f"rst_use_{team_id}"):
                for pl in sorted_players:
                    set_player_override(team_id, pl.player_id, "usage_multiplier", 1.0)
                st.rerun()
        with cc:
            if st.button(f"Clear all overrides", key=f"clr_{team_id}"):
                st.session_state.depth_charts[team_id] = {}
                st.rerun()


# ── Usage guide ───────────────────────────────────────────────────────────
with st.expander("📖 How to use this page", expanded=False):
    st.markdown("""
**Active toggle** — Uncheck to scratch a player entirely (DNP, injured, suspended).
Their projected stats go to zero and the team total is redistributed to active players.

**Starter (G)** — Select exactly one goalie per team as the starting goalie.
The model already picks the most likely starter by save%, but you can override it.

**Usage multiplier** — Scales all of a player's projections up or down proportionally.
- `1.0` = model's default
- `1.3` = player is playing an elevated role (star carrying the offense)
- `0.7` = player is limited (playing through injury, time-sharing)
- `0.0` = same as marking inactive

**Individual rating adjustments** — Click the ⚙ icon next to any active player.
These sliders let you override the specific model inputs for that player:

| Rating | What it controls |
|---|---|
| Goal share | What % of team goals this player scores |
| Assist share | What % of team assists this player earns |
| Shooting % | How often their shots become goals |
| 2PT goal rate | Fraction of goals that are 2-pointers (worth 2 pts) |
| Save % | Goalie's save rate vs opponent shots (goalies only) |
| FO win % | Faceoff specialist's win rate (FO only) |

All changes take effect when you return to **Projections** and click **▶ Run Projection**.
    """)

# ── Render teams ──────────────────────────────────────────────────────────
tab_away, tab_home = st.tabs([f"📋 {away_nm}", f"📋 {home_nm}"])

with tab_away:
    _render_team(away_id, away_nm, result.away_players)

with tab_home:
    _render_team(home_id, home_nm, result.home_players)

st.markdown("---")
st.markdown(
    '<span class="note-text">Changes take effect when you click ▶ Run Projection on the Projections page.</span>',
    unsafe_allow_html=True,
)
