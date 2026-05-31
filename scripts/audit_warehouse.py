"""
Warehouse audit script — run locally to verify the DuckDB is clean and complete.
Prints a structured report covering schema, row counts, column coverage, nulls,
season coverage, data quality checks, and projection-readiness.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "analytics_database" / "pll_warehouse.duckdb"

try:
    import duckdb
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

if not DB_PATH.exists():
    print(f"ERROR: warehouse not found at {DB_PATH}")
    sys.exit(1)

con = duckdb.connect(str(DB_PATH), read_only=True)

SEP = "=" * 72


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def q(sql, params=None):
    if params:
        return con.execute(sql, params).df()
    return con.execute(sql).df()


# ── 1. Schema inventory ────────────────────────────────────────────────────
section("1. SCHEMA INVENTORY")

tables = q("""
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_schema IN ('clean','marts','qc')
    ORDER BY table_schema, table_name
""")

for schema in ["clean", "marts", "qc"]:
    names = tables[tables["table_schema"] == schema]["table_name"].tolist()
    print(f"\n  {schema} ({len(names)} tables):")
    for name in names:
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {schema}.{name}").fetchone()[0]
            print(f"    {name:<45} {n:>8,} rows")
        except Exception as e:
            print(f"    {name:<45} ERROR: {e}")


# ── 2. Season coverage ─────────────────────────────────────────────────────
section("2. SEASON COVERAGE — game_manifest")

try:
    cov = q("""
        SELECT season,
               COUNT(DISTINCT game_id) AS games,
               MIN(game_date_utc)      AS first_game,
               MAX(game_date_utc)      AS last_game
        FROM clean.game_manifest
        GROUP BY season
        ORDER BY season
    """)
    print(cov.to_string(index=False))
except Exception as e:
    print(f"  ERROR: {e}")


section("2b. SEASON COVERAGE — team_game_stats")
try:
    tgs = q("""
        SELECT season,
               COUNT(DISTINCT game_id) AS games,
               COUNT(*) AS team_rows,
               COUNT(DISTINCT team_id) AS teams
        FROM clean.team_game_stats
        GROUP BY season ORDER BY season
    """)
    print(tgs.to_string(index=False))
except Exception as e:
    print(f"  ERROR: {e}")


section("2c. SEASON COVERAGE — player_game_stats")
try:
    pgs = q("""
        SELECT season,
               COUNT(DISTINCT game_id)   AS games,
               COUNT(*)                  AS player_rows,
               COUNT(DISTINCT player_id) AS players,
               COUNT(DISTINCT team_id)   AS teams
        FROM clean.player_game_stats
        GROUP BY season ORDER BY season
    """)
    print(pgs.to_string(index=False))
except Exception as e:
    print(f"  ERROR: {e}")


# ── 3. Schedule / upcoming games ──────────────────────────────────────────
section("3. SCHEDULE — upcoming games")

try:
    sched = q("""
        SELECT season, event_status_label,
               COUNT(*) AS games,
               MIN(game_date_guess) AS earliest,
               MAX(game_date_guess) AS latest
        FROM clean.game_schedule_all
        GROUP BY season, event_status_label
        ORDER BY season, event_status_label
    """)
    print(sched.to_string(index=False))
except Exception as e:
    print(f"  ERROR: {e}")

try:
    upcoming = q("""
        SELECT season, game_number, game_date_guess,
               away_team_id, home_team_id,
               away_score, home_score, event_status_label
        FROM clean.game_schedule_all
        WHERE event_status_label NOT IN ('final','completed')
           OR event_status_label IS NULL
        ORDER BY game_date_guess NULLS LAST, game_number
        LIMIT 20
    """)
    print(f"\n  Upcoming (first 20):")
    print(upcoming.to_string(index=False))
except Exception as e:
    print(f"  Upcoming query error: {e}")


# ── 4. team_game_stats column coverage ───────────────────────────────────
section("4. team_game_stats — COLUMN NULL RATES (2026 games)")

TEAM_STAT_COLS = [
    "goals", "one_point_goals", "two_point_goals", "assists",
    "shots", "shots_on_goal", "shot_pct",
    "faceoffs", "faceoffs_won", "faceoff_pct",
    "saves", "save_pct",
    "ground_balls", "turnovers", "caused_turnovers",
    "touches", "total_passes", "time_in_possession", "time_in_possession_pct",
    "total_possessions", "offensive_sequence_proxy",
    "clears", "clear_pct", "num_penalties",
    "two_point_goals_against", "scores", "scores_against",
]

try:
    cols_in_db = q("SELECT column_name FROM information_schema.columns WHERE table_schema='clean' AND table_name='team_game_stats'")["column_name"].tolist()

    rows_2026 = con.execute("SELECT COUNT(*) FROM clean.team_game_stats WHERE season=2026").fetchone()[0]
    rows_all = con.execute("SELECT COUNT(*) FROM clean.team_game_stats").fetchone()[0]
    print(f"\n  Total rows: {rows_all:,}   |   2026 rows: {rows_2026:,}")
    print(f"\n  {'column':<35} {'in_db':>6} {'null%_all':>10} {'null%_2026':>11} {'mean_all':>10}")
    print(f"  {'-'*35} {'-'*6} {'-'*10} {'-'*11} {'-'*10}")

    for col in TEAM_STAT_COLS:
        if col not in cols_in_db:
            print(f"  {col:<35} {'MISSING':>6}")
            continue
        try:
            r = con.execute(f"""
                SELECT
                    ROUND(100.0 * SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) AS null_pct_all,
                    ROUND(100.0 * SUM(CASE WHEN {col} IS NULL AND season=2026 THEN 1 ELSE 0 END) / NULLIF(SUM(CASE WHEN season=2026 THEN 1 ELSE 0 END),0), 1) AS null_pct_2026,
                    ROUND(AVG(TRY_CAST({col} AS DOUBLE)), 2) AS mean_all
                FROM clean.team_game_stats
            """).fetchone()
            print(f"  {col:<35} {'yes':>6} {str(r[0])+'%':>10} {str(r[1])+'%':>11} {str(r[2]):>10}")
        except Exception as e:
            print(f"  {col:<35} {'yes':>6} ERROR: {e}")
except Exception as e:
    print(f"  ERROR: {e}")


# ── 5. player_game_stats column coverage ─────────────────────────────────
section("5. player_game_stats — COLUMN NULL RATES")

PLAYER_STAT_COLS = [
    "goals", "one_point_goals", "two_point_goals", "assists", "points", "scoring_points",
    "shots", "shots_on_goal", "shot_pct",
    "saves", "save_pct", "scores_against",
    "faceoffs", "faceoffs_won", "faceoff_pct",
    "ground_balls", "turnovers", "caused_turnovers",
    "touches", "total_passes",
    "num_penalties", "two_point_goals_against",
    "position", "team_id", "player_id",
]

try:
    pcols_in_db = q("SELECT column_name FROM information_schema.columns WHERE table_schema='clean' AND table_name='player_game_stats'")["column_name"].tolist()
    p_rows = con.execute("SELECT COUNT(*) FROM clean.player_game_stats").fetchone()[0]
    print(f"\n  Total player-game rows: {p_rows:,}")
    print(f"\n  {'column':<35} {'in_db':>6} {'null%':>8} {'mean':>10}")
    print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*10}")

    for col in PLAYER_STAT_COLS:
        if col not in pcols_in_db:
            print(f"  {col:<35} {'MISSING':>6}")
            continue
        try:
            r = con.execute(f"""
                SELECT
                    ROUND(100.0 * SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1),
                    ROUND(AVG(TRY_CAST({col} AS DOUBLE)), 3)
                FROM clean.player_game_stats
            """).fetchone()
            print(f"  {col:<35} {'yes':>6} {str(r[0])+'%':>8} {str(r[1]):>10}")
        except Exception as e:
            print(f"  {col:<35} {'yes':>6} ERROR: {e}")
except Exception as e:
    print(f"  ERROR: {e}")


# ── 6. Scoring integrity checks ───────────────────────────────────────────
section("6. SCORING INTEGRITY CHECKS")

checks = [
    ("team: scores = 1pt_goals + 2*2pt_goals",
     "SELECT COUNT(*) FROM clean.team_game_stats WHERE ABS(scores - (one_point_goals + 2*two_point_goals)) > 0.5"),
    ("team: goals = 1pt + 2pt",
     "SELECT COUNT(*) FROM clean.team_game_stats WHERE ABS(goals - (one_point_goals + two_point_goals)) > 0.5"),
    ("team: faceoffs = won + lost",
     "SELECT COUNT(*) FROM clean.team_game_stats WHERE faceoffs IS NOT NULL AND faceoffs_won IS NOT NULL AND faceoffs_lost IS NOT NULL AND ABS(faceoffs - (faceoffs_won + faceoffs_lost)) > 0.5"),
    ("player: goals = 1pt + 2pt",
     "SELECT COUNT(*) FROM clean.player_game_stats WHERE ABS(goals - (one_point_goals + two_point_goals)) > 0.5"),
    ("player: points = scoring_pts + assists",
     "SELECT COUNT(*) FROM clean.player_game_stats WHERE points IS NOT NULL AND scoring_points IS NOT NULL AND assists IS NOT NULL AND ABS(points - (scoring_points + assists)) > 0.5"),
    ("team rows per game = 2",
     "SELECT COUNT(*) FROM (SELECT game_id, COUNT(*) AS n FROM clean.team_game_stats GROUP BY game_id HAVING n != 2)"),
    ("duplicate team-game keys",
     "SELECT COUNT(*) FROM (SELECT game_id, team_id, COUNT(*) AS n FROM clean.team_game_stats GROUP BY game_id, team_id HAVING n > 1)"),
    ("duplicate player-game keys",
     "SELECT COUNT(*) FROM (SELECT game_id, player_id, team_id, COUNT(*) AS n FROM clean.player_game_stats GROUP BY game_id, player_id, team_id HAVING n > 1)"),
]

for label, sql in checks:
    try:
        n = con.execute(sql).fetchone()[0]
        status = "PASS" if n == 0 else f"FAIL ({n:,} rows)"
        print(f"  {label:<50} {status}")
    except Exception as e:
        print(f"  {label:<50} ERROR: {e}")


# ── 7. Team scoring by season (sanity) ────────────────────────────────────
section("7. TEAM SCORING AVERAGES BY SEASON")

try:
    scoring = q("""
        SELECT season,
               ROUND(AVG(goals),2)           AS avg_goals,
               ROUND(AVG(scores),2)          AS avg_scores,
               ROUND(AVG(two_point_goals),2) AS avg_2pt,
               ROUND(AVG(shots),2)           AS avg_shots,
               ROUND(AVG(shots_on_goal),2)   AS avg_sog,
               ROUND(AVG(faceoffs_won),2)    AS avg_fo_wins,
               ROUND(AVG(turnovers),2)       AS avg_to,
               ROUND(AVG(ground_balls),2)    AS avg_gb,
               ROUND(AVG(saves),2)           AS avg_saves
        FROM clean.team_game_stats
        GROUP BY season ORDER BY season
    """)
    print(scoring.to_string(index=False))
except Exception as e:
    print(f"  ERROR: {e}")


# ── 8. Possession data availability ──────────────────────────────────────
section("8. POSSESSION DATA AVAILABILITY BY SEASON")

try:
    poss = q("""
        SELECT season,
               COUNT(*) AS team_rows,
               SUM(CASE WHEN time_in_possession IS NOT NULL AND time_in_possession > 0 THEN 1 ELSE 0 END) AS has_top,
               SUM(CASE WHEN touches IS NOT NULL AND touches > 0 THEN 1 ELSE 0 END) AS has_touches,
               SUM(CASE WHEN total_passes IS NOT NULL AND total_passes > 0 THEN 1 ELSE 0 END) AS has_passes,
               SUM(CASE WHEN total_possessions IS NOT NULL AND total_possessions > 0 THEN 1 ELSE 0 END) AS has_official_poss,
               ROUND(AVG(CASE WHEN time_in_possession > 0 THEN time_in_possession END), 1) AS avg_top_sec,
               ROUND(AVG(CASE WHEN touches > 0 THEN touches END), 1) AS avg_touches
        FROM clean.team_game_stats
        GROUP BY season ORDER BY season
    """)
    print(poss.to_string(index=False))
except Exception as e:
    print(f"  ERROR: {e}")


# ── 9. Player stat distributions (top scorers) ───────────────────────────
section("9. PLAYER STAT DISTRIBUTIONS — 2026 season (top 15 by points)")

try:
    top = q("""
        SELECT p.full_name, p.position, p.team_id,
               SUM(pg.goals)       AS goals,
               SUM(pg.assists)     AS assists,
               SUM(pg.points)      AS points,
               SUM(pg.two_point_goals) AS twopt,
               SUM(pg.shots)       AS shots,
               SUM(pg.shots_on_goal) AS sog,
               SUM(pg.ground_balls) AS gb,
               COUNT(DISTINCT pg.game_id) AS games
        FROM clean.player_game_stats pg
        JOIN clean.player_directory p ON pg.player_id = p.player_id
        WHERE pg.season = 2026
        GROUP BY p.full_name, p.position, p.team_id
        ORDER BY points DESC NULLS LAST
        LIMIT 15
    """)
    print(top.to_string(index=False))
except Exception as e:
    print(f"  ERROR: {e}")


# ── 10. Faceoff specialist check ─────────────────────────────────────────
section("10. FACEOFF SPECIALIST DATA (position=FO, 2026)")

try:
    fo = q("""
        SELECT pg.full_name, pg.team_id,
               SUM(pg.faceoffs_won) AS fo_wins,
               SUM(pg.faceoffs)     AS fo_total,
               ROUND(100.0*SUM(pg.faceoffs_won)/NULLIF(SUM(pg.faceoffs),0),1) AS fo_pct,
               COUNT(DISTINCT pg.game_id) AS games
        FROM clean.player_game_stats pg
        WHERE pg.season = 2026
          AND pg.position = 'FO'
          AND pg.faceoffs_won IS NOT NULL
        GROUP BY pg.full_name, pg.team_id
        ORDER BY fo_wins DESC NULLS LAST
        LIMIT 20
    """)
    print(fo.to_string(index=False))
except Exception as e:
    print(f"  ERROR: {e}")


# ── 11. Defense / opponent mart check ─────────────────────────────────────
section("11. DEFENSIVE MART — team_defense_season_stats")

try:
    def_s = q("""
        SELECT season, team_name,
               ROUND(scores_allowed_per_game,2) AS sag,
               ROUND(goals_allowed_per_game,2)  AS gag,
               ROUND(opponent_shots_per_game,2) AS opp_shots_pg,
               ROUND(save_pct_proxy,3)           AS sv_pct
        FROM marts.team_defense_season_stats
        ORDER BY season, sag NULLS LAST
    """)
    print(def_s.to_string(index=False))
except Exception as e:
    print(f"  ERROR: {e}")


# ── 12. Projection-readiness summary ──────────────────────────────────────
section("12. PROJECTION-READINESS SUMMARY")

checks_proj = [
    ("clean.team_game_stats has >=200 rows",
     "SELECT COUNT(*) >= 200 FROM clean.team_game_stats"),
    ("clean.player_game_stats has >=1000 rows",
     "SELECT COUNT(*) >= 1000 FROM clean.player_game_stats"),
    ("team_game_stats has goals column",
     "SELECT COUNT(*) > 0 FROM information_schema.columns WHERE table_schema='clean' AND table_name='team_game_stats' AND column_name='goals'"),
    ("team_game_stats has faceoffs_won",
     "SELECT COUNT(*) > 0 FROM information_schema.columns WHERE table_schema='clean' AND table_name='team_game_stats' AND column_name='faceoffs_won'"),
    ("team_game_stats has time_in_possession",
     "SELECT COUNT(*) > 0 FROM information_schema.columns WHERE table_schema='clean' AND table_name='team_game_stats' AND column_name='time_in_possession'"),
    ("player_game_stats has two_point_goals",
     "SELECT COUNT(*) > 0 FROM information_schema.columns WHERE table_schema='clean' AND table_name='player_game_stats' AND column_name='two_point_goals'"),
    ("game_schedule_all has upcoming games",
     "SELECT COUNT(*) > 0 FROM clean.game_schedule_all WHERE event_status_label NOT IN ('final','completed') OR event_status_label IS NULL"),
    ("player_directory has >=200 players",
     "SELECT COUNT(*) >= 200 FROM clean.player_directory"),
    ("marts.team_defense_season_stats populated",
     "SELECT COUNT(*) > 0 FROM marts.team_defense_season_stats"),
    ("marts.player_ranking_profiles populated",
     "SELECT COUNT(*) > 0 FROM marts.player_ranking_profiles"),
]

all_pass = True
for label, sql in checks_proj:
    try:
        ok = con.execute(sql).fetchone()[0]
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    except Exception as e:
        print(f"  ERROR  {label}: {e}")
        all_pass = False

print()
if all_pass:
    print("  >>> ALL PROJECTION-READINESS CHECKS PASSED")
else:
    print("  >>> SOME CHECKS FAILED — see above for gaps")

con.close()
print(f"\n{SEP}")
print("  Audit complete.")
print(SEP)
