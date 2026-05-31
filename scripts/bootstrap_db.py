"""
bootstrap_db.py
---------------
Rebuilds the DuckDB warehouse from committed parquet files.

Called automatically by the Streamlit app at startup when the .duckdb file
is absent (e.g. fresh Streamlit Cloud deploy, new clone, CI environment).

The .duckdb file is excluded from git (.gitignore) because it can be 50-200MB.
The parquet files under data/curated_data/all_requested_seasons/ are committed
and contain all the same data — rebuilding the DB from them takes ~5-10 seconds.

Usage:
    python scripts/bootstrap_db.py                  # rebuild if missing
    python scripts/bootstrap_db.py --force          # always rebuild
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bootstrap")

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "analytics_database" / "pll_warehouse.duckdb"
PARQUET_DIR = REPO_ROOT / "data" / "curated_data" / "all_requested_seasons"

# Schema mapping: parquet file name -> (duckdb schema, table name)
CLEAN_TABLES = [
    "game_manifest",
    "team_game_stats",
    "player_game_stats",
    "team_alias_mapping",
    "team_directory",
    "player_directory",
    "game_schedule_all",
    "game_schedule_2026",
]

MART_TABLES = [
    "player_season_stats_by_team",
    "player_season_stats",
    "player_career_stats",
    "player_vs_opponent_stats",
    "player_last5_stats",
    "player_last10_stats",
    "player_season_last5_stats",
    "player_season_last10_stats",
    "player_ranking_profiles",
    "team_season_stats",
    "team_career_stats",
    "team_vs_opponent_stats",
    "team_last5_stats",
    "team_last10_stats",
    "team_season_last5_stats",
    "team_season_last10_stats",
    "team_style_profiles",
    "team_game_possession_quality",
    "team_game_opponent_context",
    "team_defense_season_stats",
    "team_defense_career_stats",
]

QC_TABLES = [
    "season_schedule_inventory",
    "stat_slug_inventory",
    "quality_summary",
    "skipped_games",
]


def parquet_exists(name: str) -> bool:
    return (PARQUET_DIR / f"{name}.parquet").exists()


def bootstrap(force: bool = False) -> bool:
    """
    Rebuild the DuckDB warehouse from parquet files.

    Returns True if the DB was (re)built, False if it already existed.
    """
    if DB_PATH.exists() and not force:
        logger.info("DuckDB already exists at %s — skipping bootstrap", DB_PATH)
        return False

    if not PARQUET_DIR.exists():
        logger.error(
            "Parquet directory not found: %s\n"
            "Run the GitHub Action (update-pll-data) or build_warehouse.py locally first.",
            PARQUET_DIR,
        )
        sys.exit(1)

    # Check that at minimum the core tables are present
    core_required = ["team_game_stats", "player_game_stats", "game_schedule_all"]
    missing = [t for t in core_required if not parquet_exists(t)]
    if missing:
        logger.error(
            "Required parquet files missing: %s\n"
            "The GitHub Action has not run yet, or data/ was not committed.",
            missing,
        )
        sys.exit(1)

    try:
        import duckdb
    except ImportError:
        logger.error("duckdb not installed. Run: pip install duckdb")
        sys.exit(1)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        DB_PATH.unlink()

    t0 = time.time()
    logger.info("Bootstrapping DuckDB from parquet files in %s", PARQUET_DIR)

    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE SCHEMA IF NOT EXISTS clean;")
    con.execute("CREATE SCHEMA IF NOT EXISTS marts;")
    con.execute("CREATE SCHEMA IF NOT EXISTS qc;")

    loaded, skipped = 0, 0

    for name in CLEAN_TABLES:
        fp = PARQUET_DIR / f"{name}.parquet"
        if fp.exists():
            con.execute(f"""
                CREATE OR REPLACE TABLE clean.{name} AS
                SELECT * FROM read_parquet('{fp.as_posix()}')
            """)
            loaded += 1
            logger.debug("  clean.%s loaded", name)
        else:
            logger.warning("  clean.%s — parquet not found, skipping", name)
            skipped += 1

    for name in MART_TABLES:
        fp = PARQUET_DIR / f"{name}.parquet"
        if fp.exists():
            con.execute(f"""
                CREATE OR REPLACE TABLE marts.{name} AS
                SELECT * FROM read_parquet('{fp.as_posix()}')
            """)
            loaded += 1
            logger.debug("  marts.%s loaded", name)
        else:
            logger.warning("  marts.%s — parquet not found, skipping", name)
            skipped += 1

    for name in QC_TABLES:
        fp = PARQUET_DIR / f"{name}.parquet"
        if fp.exists():
            con.execute(f"""
                CREATE OR REPLACE TABLE qc.{name} AS
                SELECT * FROM read_parquet('{fp.as_posix()}')
            """)
            loaded += 1
        else:
            skipped += 1

    # Quick validation
    try:
        n_tg = con.execute("SELECT COUNT(*) FROM clean.team_game_stats").fetchone()[0]
        n_pg = con.execute("SELECT COUNT(*) FROM clean.player_game_stats").fetchone()[0]
        n_sc = con.execute("SELECT COUNT(*) FROM clean.game_schedule_all").fetchone()[0]
        logger.info(
            "Validation: team_game_stats=%d, player_game_stats=%d, schedule=%d",
            n_tg, n_pg, n_sc,
        )
        if n_tg < 100:
            logger.warning("team_game_stats has only %d rows — data may be incomplete", n_tg)
    except Exception as e:
        logger.warning("Validation query failed: %s", e)

    con.close()

    elapsed = time.time() - t0
    logger.info(
        "Bootstrap complete in %.1fs — %d tables loaded, %d skipped",
        elapsed, loaded, skipped,
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap DuckDB from parquet files")
    parser.add_argument("--force", action="store_true", help="Rebuild even if DB already exists")
    args = parser.parse_args()
    rebuilt = bootstrap(force=args.force)
    sys.exit(0)
