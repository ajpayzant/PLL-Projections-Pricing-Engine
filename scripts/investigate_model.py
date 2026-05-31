"""
Deep-dive model investigation:
- Walk-forward projections for the last 8 games of the 2025 regular season
- Projected vs actual: game totals, team goals, team scores
- Top player projections vs actual for each game
- Full feature audit: what every feature is and why it's there
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import duckdb
import warnings
warnings.filterwarnings("ignore")

from projection_engine_v2 import (
    DataLoader, FeatureBuilder, TeamModel, PlayerModel,
    GameSimulator, PricingEngine, Calibrator,
    _nan, _season_weight,
)

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "analytics_database" / "pll_warehouse.duckdb"
con = duckdb.connect(str(DB_PATH), read_only=True)

loader = DataLoader()
all_tg = loader.load_team_games()
all_pg = loader.load_player_games()

SEP = "=" * 80
SEP2 = "-" * 80

# ============================================================
# PART 1 — FEATURE AUDIT
# ============================================================

print(SEP)
print("  PART 1: FEATURE AUDIT — What every feature is and why it's in the model")
print(SEP)

feature_docs = [
    # ── EWM Offensive ──────────────────────────────────────────────────────────
    ("ewm_goals",
     "EWM (half-life=5 games) of team goals per game, shifted to exclude current game. "
     "Primary offensive strength signal. EWM chosen over simple rolling avg because "
     "recent games matter more — a team on a hot streak should show it."),

    ("ewm_shots",
     "EWM of shots per game. Shots are more stable game-to-game than goals (less luck), "
     "so shots give a more reliable offensive volume signal than goals alone. "
     "Also used in the multiplicative log5 matchup model."),

    ("ewm_sog",
     "EWM of shots on goal. Captures shot quality (what fraction of shots test the goalie). "
     "Used to project goalie save opportunities and player SOG lines."),

    ("ewm_shot_pct",
     "EWM of goals/shots (team shooting %). Separates volume from efficiency. "
     "A team with 40 shots but 25% shot% projects differently than one with 40 shots at 30%."),

    ("ewm_sog_rate",
     "EWM of shots_on_goal/shots. Measures how many shots actually challenge the goalie. "
     "Teams with high SOG rate force more saves even when shot totals are equal."),

    ("ewm_turnovers",
     "EWM of turnovers per game. Possession disruptions directly reduce offensive sequences. "
     "High-turnover teams get fewer quality shot opportunities."),

    ("ewm_caused_turnovers",
     "EWM of caused turnovers (defensive forced TOs). Separate from own TOs — "
     "forced TOs are an active defensive weapon that creates fast-break opportunities."),

    ("ewm_ground_balls",
     "EWM of ground balls. A proxy for 50/50 possession battles. Teams that win GBs "
     "get extra offensive possessions and disrupt opponent rhythm."),

    ("ewm_touches",
     "EWM of ball touches per game. Captures possession style — "
     "high-touch teams run slower offenses with more ball movement."),

    ("ewm_2pt_rate",
     "EWM of 2pt_goals / goals (fraction of goals that were 2-pointers). "
     "Critical because a 2pt goal is worth 2 score units vs 1. "
     "Teams/players with high 2pt rates score more points per goal."),

    # ── EWM Defensive ──────────────────────────────────────────────────────────
    ("ewm_goals_against",
     "EWM of goals allowed per game. Core defensive quality signal used in the "
     "log5 opponent-adjusted projection: proj = lg_avg * (team_off/lg) * (opp_def/lg). "
     "Lower ewm_goals_against = better defense = opponent scores less."),

    ("ewm_saves",
     "EWM of saves per game. Directly reflects goalie workload and output. "
     "Used in player-level goalie projections."),

    ("ewm_sog_against",
     "EWM of shots on goal allowed. Defensive shot suppression quality — "
     "a team that allows few SOG is harder to score against regardless of SOG-to-goal conversion."),

    ("ewm_save_pct",
     "EWM of save percentage (saves / shots_on_goal_against). "
     "Key goalie quality metric. Bayesian version (bayes_save_pct) is preferred "
     "for small samples but EWM captures recent form."),

    # ── FO / Possession ────────────────────────────────────────────────────────
    ("ewm_fo_pct",
     "EWM of faceoff win% (using won+lost as denominator, not the raw 'faceoffs' column "
     "which has ±1 noise in 17% of games). Faceoffs are the single most controllable "
     "possession driver in lacrosse — winning FOs = more offensive possessions = more goals. "
     "Used in the FO possession-edge model: goal_adj = +/- 10% at FO extremes."),

    # ── Bayesian Rates ─────────────────────────────────────────────────────────
    ("bayes_fo_pct",
     "Bayesian Beta posterior mean of career FO win% (prior: 2 wins, 2 losses = 50%). "
     "More stable than EWM for faceoff specialists — FO data is noisy per game. "
     "Beta(alpha + cumulative_wins, beta + cumulative_losses) shrinks toward 50% "
     "until enough data accumulates. PREFERRED over ewm_fo_pct for the FO edge model."),

    ("bayes_shot_pct",
     "Bayesian Beta posterior of career shooting% (prior: 4 goals, 10 misses = ~29%). "
     "Prior is set at league avg shot%. Shrinks small-sample outliers toward league avg — "
     "prevents a team that shot 40% over 3 early games from being projected at 40%."),

    ("bayes_save_pct",
     "Bayesian Beta posterior of career save% (prior: 3 saves, 3 goals = 50%). "
     "Used for goalie projections. Prevents an early-season hot streak from over-inflating "
     "goalie projections."),

    # ── Rolling averages / context ─────────────────────────────────────────────
    ("season_goals_pg",
     "Current-season cumulative average goals per game (resets each season). "
     "Captures within-season trajectory separate from EWM. "
     "A team that was great last year but struggling this year will have a low season_goals_pg "
     "even if ewm_goals (which spans seasons) is still elevated."),

    ("career_goals_pg",
     "All-time (career/franchise) cumulative average goals per game. "
     "Long-run baseline that anchors projections when a team has few games in the current season. "
     "Regression to mean blends career_goals_pg + ewm_goals."),

    # ── Home/Away ──────────────────────────────────────────────────────────────
    ("home_advantage",
     "Binary flag (1=home, 0=away). Home teams avg +0.39 goals/game vs away (measured from data). "
     "Applied as an additive adjustment AFTER the multiplicative log5 projection. "
     "Home/away splits on ewm_goals_home / ewm_goals_away also built but used for context."),

    ("ewm_goals_home / ewm_goals_away",
     "Split EWM goals for home games vs away games separately. "
     "Some teams are dramatically better at home (crowd, travel schedule). "
     "Helps the Ridge correction learn home/away performance patterns."),

    # ── Season decay weight ─────────────────────────────────────────────────────
    ("season_weight",
     "Exponential decay weight applied during Ridge model fitting: "
     "w = 0.5^((current_season - season) / 1.5). "
     "2026 games weight 1.0, 2025 weight ~0.63, 2024 weight ~0.40, 2023 weight ~0.25, 2022 weight ~0.16. "
     "Prevents stale 4-year-old games from having equal influence to last week's game. "
     "Half-life of 1.5 seasons chosen to balance sample size vs recency."),

    # ── Games played ───────────────────────────────────────────────────────────
    ("games_played",
     "Number of games played by this team in the current season BEFORE this game. "
     "Used for confidence scoring (projection confidence scales with data availability) "
     "and controls how much the model regresses to mean in early season."),
]

print(f"\n  {'Feature':<32}  Description")
print(f"  {'-'*32}  {'-'*44}")
for name, desc in feature_docs:
    lines = []
    words = desc.split()
    line = ""
    for w in words:
        if len(line) + len(w) + 1 > 90:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    lines.append(line)
    print(f"\n  {name:<32}")
    for ln in lines:
        print(f"  {'':32}  {ln}")

print(f"\n  Total features built: 121 columns (team feature matrix)")
print(f"  Features used in Ridge regression: 16 (subset that are pre-game and numeric)")
print(f"  All features are leakage-safe: computed via shift(1) so row i uses only rows 0..i-1")


# ============================================================
# PART 2 — LAST 8 GAMES OF 2025 SEASON
# ============================================================

print(f"\n\n{SEP}")
print("  PART 2: LAST 8 GAMES OF 2025 — Walk-forward projections vs actual")
print(SEP)

# Get 2025 game list sorted
games_2025 = con.execute("""
    SELECT DISTINCT
        g.game_id,
        g.game_number,
        g.game_date_utc,
        g.team_id      AS home_team_id,
        g.team_name    AS home_team_name,
        g.goals        AS home_goals,
        g.scores       AS home_scores,
        g.two_point_goals AS home_2pt,
        g.shots        AS home_shots,
        g.shots_on_goal AS home_sog,
        g.faceoffs_won  AS home_fo_wins,
        CAST(g.faceoffs_won AS DOUBLE) / NULLIF(g.faceoffs_won + g.faceoffs_lost, 0) AS home_fo_pct,
        g.saves        AS home_saves,
        g.turnovers    AS home_to,
        a.team_id      AS away_team_id,
        a.team_name    AS away_team_name,
        a.goals        AS away_goals,
        a.scores       AS away_scores,
        a.two_point_goals AS away_2pt,
        a.shots        AS away_shots,
        a.shots_on_goal AS away_sog,
        a.faceoffs_won  AS away_fo_wins,
        CAST(a.faceoffs_won AS DOUBLE) / NULLIF(a.faceoffs_won + a.faceoffs_lost, 0) AS away_fo_pct,
        a.saves        AS away_saves,
        a.turnovers    AS away_to
    FROM clean.team_game_stats g
    JOIN clean.team_game_stats a ON g.game_id = a.game_id AND g.team_id != a.team_id
    WHERE g.season = 2025
      AND g.is_home = 1
    ORDER BY g.game_number DESC
    LIMIT 8
""").df()

print(f"\n  Found {len(games_2025)} games to evaluate (games {games_2025['game_number'].min()}–{games_2025['game_number'].max()})")

sim = GameSimulator(n_sims=10_000, seed=42)
pricing = PricingEngine(hold_pct=0.045)

all_game_rows = []
all_player_rows = []

for _, grow in games_2025.iterrows():
    game_id = grow["game_id"]
    game_num = int(grow["game_number"])
    home_id = grow["home_team_id"]
    away_id = grow["away_team_id"]
    home_name = grow["home_team_name"]
    away_name = grow["away_team_name"]

    # Train strictly on games BEFORE this one
    train_tg = all_tg[
        (all_tg["season"] < 2025) |
        ((all_tg["season"] == 2025) & (all_tg["game_number"] < game_num))
    ].copy()

    train_pg = all_pg[
        (all_pg["season"] < 2025) |
        ((all_pg["season"] == 2025) & (all_pg["game_number"] < game_num))
    ].copy()

    if len(train_tg) < 40:
        continue

    # Build features & fit
    fb = FeatureBuilder(train_tg, train_pg)
    fb.build_team_features()
    fb.build_player_features()

    tm = TeamModel()
    tm.fit(fb._tf)

    current_team_map = loader.resolve_current_team(train_pg)
    pm = PlayerModel(fb._pf if fb._pf is not None else pd.DataFrame(), current_team_map=current_team_map)

    h_feat = fb.get_team_features_current(home_id)
    a_feat = fb.get_team_features_current(away_id)
    if not h_feat or not a_feat:
        continue

    h_feat["is_home"] = 1
    h_feat["home_advantage"] = 1
    a_feat["is_home"] = 0
    a_feat["home_advantage"] = 0

    h_proj = tm.predict(h_feat, a_feat, is_home=True)
    a_proj = tm.predict(a_feat, h_feat, is_home=False)
    h_proj.proj_saves = a_proj.proj_sog * 0.55
    a_proj.proj_saves = h_proj.proj_sog * 0.55

    h_players = pm.project_roster(home_id, h_proj)
    a_players = pm.project_roster(away_id, a_proj)

    gs = sim.simulate_game(h_proj, a_proj)
    gm = pricing.price_game(gs)

    # Actual values
    act_home_goals = float(grow["home_goals"])
    act_away_goals = float(grow["away_goals"])
    act_home_scores = float(grow["home_scores"])
    act_away_scores = float(grow["away_scores"])
    act_total_goals = act_home_goals + act_away_goals
    act_total_scores = act_home_scores + act_away_scores
    act_home_win = 1 if act_home_scores > act_away_scores else 0

    proj_total_goals = h_proj.proj_goals + a_proj.proj_goals
    proj_total_scores = h_proj.proj_scores + a_proj.proj_scores

    game_date = str(grow["game_date_utc"])[:10]

    all_game_rows.append({
        "game_num": game_num,
        "date": game_date,
        "matchup": f"{home_name} vs {away_name}",
        "home": home_name,
        "away": away_name,
        # Projections
        "proj_h_goals": round(h_proj.proj_goals, 1),
        "proj_a_goals": round(a_proj.proj_goals, 1),
        "proj_h_score": round(h_proj.proj_scores, 1),
        "proj_a_score": round(a_proj.proj_scores, 1),
        "proj_total_goals": round(proj_total_goals, 1),
        "proj_total_scores": round(proj_total_scores, 1),
        "proj_h_win_pct": round(gs.home_win_prob * 100, 1),
        "proj_spread": round(gs.spread_home, 1),
        "proj_total_line": gm.total_line,
        "home_ml": gm.home_ml,
        "away_ml": gm.away_ml,
        # Actuals
        "act_h_goals": act_home_goals,
        "act_a_goals": act_away_goals,
        "act_h_score": act_home_scores,
        "act_a_score": act_away_scores,
        "act_total_goals": act_total_goals,
        "act_total_scores": act_total_scores,
        "act_winner": home_name if act_home_win else away_name,
        "proj_winner_correct": (gs.home_win_prob > 0.5) == bool(act_home_win),
        # Errors
        "err_h_goals": round(h_proj.proj_goals - act_home_goals, 1),
        "err_a_goals": round(a_proj.proj_goals - act_away_goals, 1),
        "err_total_goals": round(proj_total_goals - act_total_goals, 1),
        "err_total_scores": round(proj_total_scores - act_total_scores, 1),
        # Other projected stats
        "proj_h_fo_pct": round(h_proj.proj_faceoff_pct, 3),
        "proj_a_fo_pct": round(a_proj.proj_faceoff_pct, 3),
        "act_h_fo_pct": round(float(grow["home_fo_pct"]), 3) if grow["home_fo_pct"] else None,
        "act_a_fo_pct": round(float(grow["away_fo_pct"]), 3) if grow["away_fo_pct"] else None,
        "proj_h_shots": round(h_proj.proj_shots, 1),
        "proj_a_shots": round(a_proj.proj_shots, 1),
        "act_h_shots": float(grow["home_shots"]),
        "act_a_shots": float(grow["away_shots"]),
        "h_proj": h_proj,
        "a_proj": a_proj,
        "h_players": h_players,
        "a_players": a_players,
    })

    # Gather top player projections vs actual for this game
    act_player_rows = con.execute(f"""
        SELECT player_id, full_name, position, team_id,
               goals, one_point_goals, two_point_goals, assists, points,
               shots, shots_on_goal, saves, faceoffs_won,
               CAST(faceoffs_won AS DOUBLE) / NULLIF(faceoffs_won + faceoffs_lost, 0) AS fo_pct_act
        FROM clean.player_game_stats
        WHERE game_id = '{game_id}'
        ORDER BY points DESC NULLS LAST
    """).df()

    for pp in h_players + a_players:
        act_row = act_player_rows[act_player_rows["player_id"].astype(str) == str(pp.player_id)]
        if act_row.empty:
            continue
        act = act_row.iloc[0]
        all_player_rows.append({
            "game_num": game_num,
            "date": game_date,
            "matchup": f"{home_name} vs {away_name}",
            "team": pp.team_id,
            "player": pp.full_name,
            "pos": pp.position,
            "proj_goals": round(pp.proj_goals, 2),
            "act_goals": float(act["goals"]) if pd.notna(act["goals"]) else 0,
            "proj_assists": round(pp.proj_assists, 2),
            "act_assists": float(act["assists"]) if pd.notna(act["assists"]) else 0,
            "proj_points": round(pp.proj_points, 2),
            "act_points": float(act["points"]) if pd.notna(act["points"]) else 0,
            "proj_shots": round(pp.proj_shots, 2),
            "act_shots": float(act["shots"]) if pd.notna(act["shots"]) else 0,
            "proj_sog": round(pp.proj_sog, 2),
            "act_sog": float(act["shots_on_goal"]) if pd.notna(act["shots_on_goal"]) else 0,
            "proj_2pt": round(pp.proj_2pt_goals, 2),
            "act_2pt": float(act["two_point_goals"]) if pd.notna(act["two_point_goals"]) else 0,
            "proj_saves": round(pp.proj_saves, 2),
            "act_saves": float(act["saves"]) if pd.notna(act["saves"]) else 0,
            "err_goals": round(pp.proj_goals - (float(act["goals"]) if pd.notna(act["goals"]) else 0), 2),
            "err_points": round(pp.proj_points - (float(act["points"]) if pd.notna(act["points"]) else 0), 2),
        })


# ============================================================
# PRINT GAME-BY-GAME RESULTS
# ============================================================

print()
for row in sorted(all_game_rows, key=lambda r: r["game_num"]):
    print(f"\n  {SEP2}")
    print(f"  GAME {row['game_num']}  |  {row['date']}  |  {row['matchup']}")
    print(f"  {SEP2}")

    print(f"\n  {'':20} {'HOME':>10}  {'AWAY':>10}   {'NOTES'}")
    print(f"  {'':20} {row['home']:>10}  {row['away']:>10}")
    print(f"  {'GOALS Projected':20} {row['proj_h_goals']:>10.1f}  {row['proj_a_goals']:>10.1f}")
    print(f"  {'GOALS Actual':20} {row['act_h_goals']:>10.0f}  {row['act_a_goals']:>10.0f}   {'<-- ERROR: ' + str(row['err_h_goals']) + ' / ' + str(row['err_a_goals'])}")
    print(f"  {'SCORE Projected':20} {row['proj_h_score']:>10.1f}  {row['proj_a_score']:>10.1f}")
    print(f"  {'SCORE Actual':20} {row['act_h_score']:>10.0f}  {row['act_a_score']:>10.0f}   {'Winner: ' + row['act_winner']}  ({'CORRECT' if row['proj_winner_correct'] else 'WRONG'})")
    print(f"  {'FO% Projected':20} {row['proj_h_fo_pct']:>10.3f}  {row['proj_a_fo_pct']:>10.3f}")
    actual_h_fo = f"{row['act_h_fo_pct']:.3f}" if row['act_h_fo_pct'] else 'N/A'
    actual_a_fo = f"{row['act_a_fo_pct']:.3f}" if row['act_a_fo_pct'] else 'N/A'
    print(f"  {'FO% Actual':20} {actual_h_fo:>10}  {actual_a_fo:>10}")
    print(f"  {'Shots Projected':20} {row['proj_h_shots']:>10.1f}  {row['proj_a_shots']:>10.1f}")
    print(f"  {'Shots Actual':20} {row['act_h_shots']:>10.0f}  {row['act_a_shots']:>10.0f}")

    print(f"\n  {'Total Goals Projected':30} {row['proj_total_goals']:>6.1f}  |  Actual: {row['act_total_goals']:>5.0f}  |  Error: {row['err_total_goals']:>+5.1f}")
    print(f"  {'Total Score Projected':30} {row['proj_total_scores']:>6.1f}  |  Actual: {row['act_total_scores']:>5.0f}  |  Error: {row['err_total_scores']:>+5.1f}")
    print(f"  {'Home Win Prob':30} {row['proj_h_win_pct']:>5.1f}%  |  ML: {row['home_ml']:>6} / {row['away_ml']:>6}  |  Spread: {row['proj_spread']:>+5.1f}")
    print(f"  {'Total Line (market)':30} {row['proj_total_line']:>6.1f}")

    # Top player projections for this game
    game_players = [p for p in all_player_rows if p["game_num"] == row["game_num"]]
    game_players_sorted = sorted(game_players, key=lambda x: x["proj_points"], reverse=True)

    if game_players_sorted:
        print(f"\n  {'TOP PLAYER PROJECTIONS vs ACTUAL':}")
        print(f"  {'Player':<26} {'Team':<4} {'Pos':<4} {'ProjG':>6} {'ActG':>5} {'ProjA':>6} {'ActA':>5} {'ProjPts':>8} {'ActPts':>7} {'ProjSh':>7} {'ActSh':>6} {'Proj2pt':>8} {'Act2pt':>7}")
        print(f"  {'-'*26} {'-'*4} {'-'*4} {'-'*6} {'-'*5} {'-'*6} {'-'*5} {'-'*8} {'-'*7} {'-'*7} {'-'*6} {'-'*8} {'-'*7}")
        shown = 0
        for p in game_players_sorted:
            if shown >= 12:
                break
            # Skip truly zero players
            if p["proj_points"] < 0.3 and p["act_points"] == 0:
                continue
            err_pts = p["err_points"]
            err_flag = " <" if abs(err_pts) >= 2.0 else ""
            print(
                f"  {p['player']:<26} {p['team']:<4} {p['pos']:<4} "
                f"{p['proj_goals']:>6.2f} {p['act_goals']:>5.0f} "
                f"{p['proj_assists']:>6.2f} {p['act_assists']:>5.0f} "
                f"{p['proj_points']:>8.2f} {p['act_points']:>7.0f} "
                f"{p['proj_shots']:>7.2f} {p['act_shots']:>6.0f} "
                f"{p['proj_2pt']:>8.2f} {p['act_2pt']:>7.0f}"
                f"{err_flag}"
            )
            shown += 1

        # Goalie section
        goalies = [p for p in game_players if p["pos"] == "G" and (p["proj_saves"] > 0 or p["act_saves"] > 0)]
        if goalies:
            print(f"\n  {'GOALIE':}")
            print(f"  {'Player':<26} {'Team':<4} {'ProjSv':>8} {'ActSv':>7} {'Err':>6}")
            for g in goalies[:4]:
                err = round(g["proj_saves"] - g["act_saves"], 1)
                print(f"  {g['player']:<26} {g['team']:<4} {g['proj_saves']:>8.1f} {g['act_saves']:>7.0f} {err:>+6.1f}")


# ============================================================
# PART 3 — AGGREGATE ACCURACY SUMMARY
# ============================================================

print(f"\n\n{SEP}")
print("  PART 3: AGGREGATE ACCURACY — All 8 games")
print(SEP)

if all_game_rows:
    df_games = pd.DataFrame(all_game_rows)

    print(f"\n  Games evaluated:         {len(df_games)}")
    correct_dir = df_games["proj_winner_correct"].sum()
    print(f"  Correct winner:          {correct_dir}/{len(df_games)}  ({100*correct_dir/len(df_games):.0f}%)")

    # Goals errors
    all_side_errs = list(df_games["err_h_goals"]) + list(df_games["err_a_goals"])
    print(f"\n  --- Goals accuracy ---")
    print(f"  MAE per team goals:      {np.mean(np.abs(all_side_errs)):.2f}")
    print(f"  Bias per team goals:     {np.mean(all_side_errs):+.2f}  (+ = overprojects, - = underprojects)")
    print(f"  MAE total goals:         {np.mean(np.abs(df_games['err_total_goals'])):.2f}")
    print(f"  Bias total goals:        {np.mean(df_games['err_total_goals']):+.2f}")
    print(f"  MAE total scores:        {np.mean(np.abs(df_games['err_total_scores'])):.2f}")
    print(f"  Bias total scores:       {np.mean(df_games['err_total_scores']):+.2f}")

    print(f"\n  --- Spread accuracy ---")
    act_spreads = df_games["act_h_score"] - df_games["act_a_score"]
    proj_spreads = df_games["proj_spread"]
    spread_errs = proj_spreads - act_spreads
    print(f"  MAE spread:              {np.mean(np.abs(spread_errs)):.2f} points")
    print(f"  Bias spread:             {np.mean(spread_errs):+.2f}")

    # Player accuracy
    if all_player_rows:
        df_players = pd.DataFrame(all_player_rows)
        field = df_players[df_players["pos"] != "G"]
        print(f"\n  --- Player accuracy (field players, {len(field)} player-game observations) ---")
        print(f"  MAE goals:               {np.mean(np.abs(field['err_goals'])):.3f}")
        print(f"  Bias goals:              {np.mean(field['err_goals']):+.3f}")
        print(f"  MAE points:              {np.mean(np.abs(field['err_points'])):.3f}")
        print(f"  Bias points:             {np.mean(field['err_points']):+.3f}")

        print(f"\n  --- Point error distribution (field players) ---")
        for threshold in [0.5, 1.0, 1.5, 2.0, 3.0]:
            within = (np.abs(field['err_points']) <= threshold).mean()
            print(f"  Within ±{threshold:.1f} points:       {within:.0%} of player-game projections")

        print(f"\n  --- Goals error distribution (field players) ---")
        for threshold in [0, 1, 2]:
            exact = (np.abs(field["err_goals"]) <= threshold).mean()
            print(f"  Within ±{threshold} goals:          {exact:.0%} of player-game projections")


# ============================================================
# PART 4 — FEATURE IMPORTANCE FROM RIDGE WEIGHTS
# ============================================================

print(f"\n\n{SEP}")
print("  PART 4: RIDGE MODEL — Feature weights for GOALS (trained on full history)")
print(SEP)

# Re-fit on all data to show final feature weights
fb_full = FeatureBuilder(all_tg, all_pg)
fb_full.build_team_features()
tm_full = TeamModel()
tm_full.fit(fb_full._tf)

feat_cols = [
    "ewm_goals", "ewm_goals_against", "ewm_shots", "ewm_sog",
    "ewm_shot_pct", "ewm_sog_rate", "ewm_save_pct",
    "ewm_fo_pct", "ewm_turnovers", "ewm_caused_turnovers",
    "ewm_2pt_rate", "home_advantage", "season_goals_pg",
    "bayes_fo_pct", "bayes_shot_pct", "bayes_save_pct",
]

if "goals" in tm_full._models:
    model = tm_full._models["goals"]
    scaler = tm_full._scalers["goals"]
    n_feats = scaler.n_features_in_
    avail = [c for c in feat_cols if c in fb_full._tf.columns][:n_feats]
    coefs = model.coef_
    print(f"\n  Ridge alpha selected: {model.alpha_}")
    print(f"  n_features: {n_feats}")
    print(f"\n  {'Feature':<30} {'Ridge coef':>12}  {'Interpretation'}")
    print(f"  {'-'*30} {'-'*12}  {'-'*40}")
    sorted_feats = sorted(zip(avail, coefs), key=lambda x: abs(x[1]), reverse=True)
    for fname, coef in sorted_feats:
        direction = "more goals" if coef > 0 else "fewer goals"
        print(f"  {fname:<30} {coef:>+12.4f}  higher = {direction}")
else:
    print("  Ridge not fitted for goals (insufficient data in this context).")

print(f"\n  NOTE: Ridge coefficients are on STANDARDIZED features (mean=0, std=1).")
print(f"  Absolute magnitude = relative importance. Sign = direction of effect.")
print(f"  These coefficients correct the log5 multiplicative base projection,")
print(f"  blended 70% log5 / 30% ridge.")

con.close()
print(f"\n{SEP}")
print("  Investigation complete.")
print(SEP)
