"""
PLL Projection Engine v3
========================
Complete rewrite addressing all known issues from v2.

Key changes from v2:
- No hardcoded home advantage (not statistically significant, t=1.62)
- No forced regression to league mean (let ratings speak)
- FO% treated as possession driver, not direct goal driver
- Possession chain model: FO -> TOP/OSP -> shots -> goals
- Team goals: truncated Normal (var/mean=0.854, underdispersed)
- Player goals/assists: zero-inflated NegBin (61-78% zero rate)
- Player shots: NegBin (var/mean=3.0, overdispersed)
- Rich rating system per stat: mean, median, EWM, std, IQR (all leakage-safe)
- EWM half-lives tuned to autocorrelation per stat
- SSDM/LSM mapped to their own correct position profiles
- Goalie save% uses saves/(saves+goals_against) as denominator
- Points sim includes 2pt scoring premium
- Calibrator trained on raw game predictions (not binned)
- Starter designation for goalies
- Team offensive/defensive multiplier overrides
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
import warnings
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import duckdb
except ImportError as exc:
    raise ImportError("duckdb is required: pip install duckdb") from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pll.v3")

# ---------------------------------------------------------------------------
# League-level constants (used as priors, not hard overrides)
# ---------------------------------------------------------------------------
LG_GOALS: float = 11.25
LG_SHOTS: float = 41.0
LG_SOG: float = 26.0
LG_SOG_RATE: float = 0.633          # SOG / shots
LG_SHOT_PCT: float = 0.274          # goals / shots
LG_SAVES: float = 13.1
LG_SHOTS_FACED: float = 24.5        # saves + goals_against per team per game
LG_SAVE_PCT: float = 0.537          # saves / shots_faced
LG_FO_PCT: float = 0.500
LG_FOS_PER_GAME: int = 26
LG_TO: float = 17.5
LG_GB: float = 31.8
LG_TOUCHES: float = 263.0
LG_PASSES: float = 205.0
LG_TOP_SEC: float = 1344.0          # time in possession seconds
LG_OSP: float = 61.8                # offensive sequence proxy
LG_2PT_RATE: float = 0.074          # two_point_goals / goals

# EWM half-lives tuned to autocorrelation per stat
# (from audit: fo_pct=0.33, shots=0.124, to=0.135, goals=0.079, TOP=0.068)
HL_FO: int = 8       # most persistent
HL_TO: int = 5
HL_SHOTS: int = 5
HL_GOALS: int = 4    # least persistent
HL_POSS: int = 4

SEASON_HALFLIFE: float = 1.5
N_SIMS: int = 20_000

# Team goal std dev (measured: ~3.1/game per team)
TEAM_GOAL_SIGMA_BASE: float = 3.1

# Player zero-inflation rates — empirically measured from all seasons.
# These are the PRIOR defaults; the engine also computes per-player
# zero_rate_goals from career history via expanding mean (overrides these
# for players with sufficient data). Audit verified actual rates:
#   A=0.202, M=0.388, FO=0.781, SSDM=0.840, D=0.948, LSM~0.87
# Assists have higher zero rates than goals across all positions.
ZERO_RATE: Dict[str, float] = {
    "A_goals":    0.20, "M_goals":    0.39, "FO_goals":   0.78,
    "SSDM_goals": 0.84, "LSM_goals":  0.87, "D_goals":    0.95,
    "A_assists":  0.55, "M_assists":  0.72, "FO_assists":  0.94,
    "SSDM_assists": 0.90, "LSM_assists": 0.90, "D_assists": 0.96,
}

# NegBin phi for player stats (var/mean from data)
PHI_PLAYER: Dict[str, float] = {
    "goals": 1.8,      # var/mean=1.58
    "assists": 2.0,    # var/mean=1.72
    "shots": 1.5,      # var/mean=3.03 -> lower phi = more overdispersion
    "sog": 2.0,
    "2pt": 1.2,
    "gb": 1.8,
    "saves": 3.5,      # goalies are consistent
    "fo_wins": 5.0,    # FO specialists are very consistent
    "default": 2.0,
}

# Position profiles: defaults when player has sparse data
# Keys: goals_share, assists_share, shots_share, gb_pg, cto_pg, to_pg, touches_pg
POS_DEFAULTS: Dict[str, Dict[str, float]] = {
    "A":    {"goals_share": 0.200, "assists_share": 0.240, "shots_share": 0.180,
             "gb_pg": 1.34, "cto_pg": 0.16, "to_pg": 1.61, "touches_pg": 28.1},
    "M":    {"goals_share": 0.130, "assists_share": 0.150, "shots_share": 0.140,
             "gb_pg": 1.00, "cto_pg": 0.13, "to_pg": 1.03, "touches_pg": 18.8},
    "SSDM": {"goals_share": 0.020, "assists_share": 0.025, "shots_share": 0.025,
             "gb_pg": 1.27, "cto_pg": 0.43, "to_pg": 0.35, "touches_pg": 7.3},
    "LSM":  {"goals_share": 0.015, "assists_share": 0.020, "shots_share": 0.020,
             "gb_pg": 2.40, "cto_pg": 0.74, "to_pg": 0.37, "touches_pg": 5.7},
    "D":    {"goals_share": 0.005, "assists_share": 0.008, "shots_share": 0.008,
             "gb_pg": 1.67, "cto_pg": 0.87, "to_pg": 0.30, "touches_pg": 6.6},
    "FO":   {"goals_share": 0.030, "assists_share": 0.030, "shots_share": 0.030,
             "gb_pg": 7.03, "cto_pg": 0.18, "to_pg": 0.75, "touches_pg": 7.7},
    "G":    {"goals_share": 0.000, "assists_share": 0.000, "shots_share": 0.002,
             "gb_pg": 1.85, "cto_pg": 0.25, "to_pg": 0.39, "touches_pg": 14.1},
}

# Position caps — set above observed 99th percentile to block physically
# impossible projections (e.g. a goalie projected at 4 goals) without
# cutting legitimate tail events. Observed maxima from audit:
#   G: goals=1, shots=2 | FO: goals=3 | D: goals=2, shots=4
#   SSDM: goals=3, shots=5 | LSM: goals=2, shots=4
POS_CAPS: Dict[str, Dict[str, float]] = {
    "G":    {"goals": 0.10, "assists": 0.20, "shots": 2.5},
    "FO":   {"goals": 3.5},
    "D":    {"goals": 2.5, "shots": 5.0},
    "SSDM": {"goals": 3.5, "shots": 6.0},
    "LSM":  {"goals": 2.5, "shots": 5.0},
}


# ---------------------------------------------------------------------------
# Position normalisation
# ---------------------------------------------------------------------------

def _norm_pos(raw: str) -> str:
    """Normalise raw position string to one of: A M SSDM LSM D FO G."""
    if not raw:
        return "M"
    r = str(raw).strip().upper()
    if r in ("A", "ATT", "ATTACK"):
        return "A"
    if r in ("M", "MID", "MIDFIELD"):
        return "M"
    if r in ("SSDM", "SS", "SSD"):
        return "SSDM"
    if r in ("LSM", "LS", "LSM1"):
        return "LSM"
    if r in ("D", "DEF", "DEFENSE", "DEF/LSM"):
        return "D"
    if r in ("FO", "FACEOFF", "FOG", "FO/MF"):
        return "FO"
    if r in ("G", "GK", "GOAL", "GOALIE", "GOALKEEPER"):
        return "G"
    return "M"



# ---------------------------------------------------------------------------
# Current official roster filtering
# ---------------------------------------------------------------------------

PUBLIC_TEAM_TO_ENGINE_TEAM: Dict[str, str] = {
    "BOS": "CAN",   # Boston Cannons
    "CAL": "RED",   # California Redwoods
    "CAR": "CHA",   # Carolina Chaos
    "DEN": "OUT",   # Denver Outlaws
    "MD":  "WHP",   # Maryland Whipsnakes
    "NY":  "ATL",   # New York Atlas
    "PHI": "WAT",   # Philadelphia Waterdogs
    "UTA": "ARC",   # Utah Archers
}

ENGINE_TEAM_IDS = {"CAN", "RED", "CHA", "OUT", "WHP", "ATL", "WAT", "ARC"}


def _norm_person_name(value: str) -> str:
    """Normalize player names for safe matching between roster cache and stat rows."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    # Ignore common suffixes and punctuation differences.
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _find_current_rosters_csv(db_path: Optional[Path] = None) -> Optional[Path]:
    """Find current_rosters.csv in repo/Streamlit/GitHub Action layouts."""
    candidates: List[Path] = []

    env_path = os.environ.get("PLL_CURRENT_ROSTERS_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path))

    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()

    # Common app/repo layouts.
    candidates.extend([
        here / "data" / "reference_tables" / "current_rosters.csv",
        here.parent / "data" / "reference_tables" / "current_rosters.csv",
        cwd / "data" / "reference_tables" / "current_rosters.csv",
        cwd.parent / "data" / "reference_tables" / "current_rosters.csv",
    ])

    if db_path is not None:
        dbp = Path(db_path).resolve()
        # DB usually lives at data/analytics_database/pll_warehouse.duckdb.
        candidates.append(dbp.parent.parent / "reference_tables" / "current_rosters.csv")
        candidates.append(dbp.parent.parent / "curated_data" / "all_requested_seasons" / "current_rosters.csv")

    seen = set()
    for c in candidates:
        try:
            c = c.resolve()
        except Exception:
            pass
        if str(c) in seen:
            continue
        seen.add(str(c))
        if c.exists():
            return c
    return None


def load_current_rosters_cache(db_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load the official current roster cache created by pll_current_roster_cache.py."""
    path = _find_current_rosters_csv(db_path)
    if path is None:
        return pd.DataFrame(), {
            "available": False,
            "source": "missing_csv",
            "path": "",
            "reason": "current_rosters.csv was not found in data/reference_tables/ or PLL_CURRENT_ROSTERS_PATH.",
        }

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return pd.DataFrame(), {
            "available": False,
            "source": "read_error",
            "path": str(path),
            "reason": f"Could not read current_rosters.csv: {exc}",
        }

    return df, {
        "available": not df.empty,
        "source": "current_rosters_csv",
        "path": str(path),
        "rows": int(len(df)),
        "reason": "loaded" if not df.empty else "current_rosters.csv is empty.",
    }


class CurrentRosterFilter:
    """
    Applies official current PLL roster cache to projection depth charts.

    The roster cache may contain both:
      - Team_Code: public codes from PLL roster pages (BOS, NY, UTA, etc.)
      - Team_ID: warehouse/model codes (CAN, ATL, ARC, etc.)

    Team_ID is preferred when present because it already matches the projection
    warehouse. Team_Code is mapped as a fallback.
    """

    def __init__(self, roster_df: Optional[pd.DataFrame] = None):
        self.raw = roster_df.copy() if roster_df is not None and not roster_df.empty else pd.DataFrame()
        self.df = pd.DataFrame()
        self.available = False
        self.status: Dict[str, object] = {
            "available": False,
            "reason": "No roster dataframe supplied.",
            "total_players": 0,
            "teams": {},
        }
        self._name_keys_by_team: Dict[str, set] = {}
        self._prepare()

    def _prepare(self) -> None:
        if self.raw.empty:
            return

        df = self.raw.copy()
        if "Player" not in df.columns:
            self.status["reason"] = "Roster cache is missing Player column."
            return

        # Prefer Team_ID because your generated CSV already has engine IDs.
        if "Team_ID" in df.columns:
            df["engine_team_id"] = df["Team_ID"].astype(str).str.strip().str.upper()
        elif "team_id" in df.columns:
            df["engine_team_id"] = df["team_id"].astype(str).str.strip().str.upper()
        elif "Team_Code" in df.columns:
            df["engine_team_id"] = (
                df["Team_Code"].astype(str).str.strip().str.upper().map(PUBLIC_TEAM_TO_ENGINE_TEAM)
            )
        else:
            self.status["reason"] = "Roster cache is missing Team_ID/team_id/Team_Code."
            return

        # Fill any unresolved Team_ID rows using public Team_Code mapping.
        if "Team_Code" in df.columns:
            mapped = df["Team_Code"].astype(str).str.strip().str.upper().map(PUBLIC_TEAM_TO_ENGINE_TEAM)
            df["engine_team_id"] = df["engine_team_id"].where(
                df["engine_team_id"].isin(ENGINE_TEAM_IDS),
                mapped,
            )

        df["name_key"] = df["Player"].map(_norm_person_name)
        df = df[(df["engine_team_id"].isin(ENGINE_TEAM_IDS)) & (df["name_key"] != "")].copy()

        counts = df.groupby("engine_team_id")["name_key"].nunique().to_dict() if not df.empty else {}
        self._name_keys_by_team = {
            str(tid): set(g["name_key"].dropna().astype(str))
            for tid, g in df.groupby("engine_team_id")
        }
        self.df = df
        self.available = bool(len(df) >= 100 and len(self._name_keys_by_team) >= 8)
        self.status = {
            "available": self.available,
            "reason": "loaded" if self.available else "Roster cache loaded but failed minimum validation.",
            "total_players": int(len(df)),
            "teams": {k: int(v) for k, v in counts.items()},
        }

    def team_name_keys(self, team_id: str) -> set:
        return self._name_keys_by_team.get(str(team_id).upper(), set())

    def team_roster_frame(self, team_id: str) -> pd.DataFrame:
        """Return official current roster rows for one engine team id."""
        if self.df.empty:
            return pd.DataFrame()
        team_id = str(team_id).upper()
        return self.df[self.df["engine_team_id"].astype(str).str.upper() == team_id].copy()

    def build_placeholder_rows(self, team_id: str, existing_rows: pd.DataFrame) -> pd.DataFrame:
        """
        Build low-usage placeholder projection rows for official-roster players
        who do not yet have historical PLL stat rows in the warehouse.

        This keeps the depth chart aligned to the current official roster while
        avoiding aggressive projections for rookies/new additions with no PLL
        sample. Users can manually raise usage/rating sliders when news implies
        a real role.
        """
        official = self.team_roster_frame(team_id)
        if official.empty:
            return pd.DataFrame()

        existing_keys = set()
        if existing_rows is not None and not existing_rows.empty and "full_name" in existing_rows.columns:
            existing_keys = set(existing_rows["full_name"].map(_norm_person_name).dropna().astype(str))

        rows: List[Dict[str, object]] = []
        for _, r in official.iterrows():
            player_name = str(r.get("Player", "")).strip()
            name_key = _norm_person_name(player_name)
            if not player_name or not name_key or name_key in existing_keys:
                continue

            pos = _norm_pos(str(r.get("Position", "M")))
            pos_defaults = POS_DEFAULTS.get(pos, POS_DEFAULTS["M"])
            safe_key = re.sub(r"[^a-z0-9]+", "_", name_key).strip("_")

            # Low default usage prevents unknown/new current-roster players from
            # stealing too much projection volume before the user sets a role.
            default_usage = 0.30 if pos in {"A", "M", "FO", "G"} else 0.20

            rows.append({
                "player_id": f"current_{team_id}_{safe_key}",
                "full_name": player_name,
                "first_name": str(r.get("First_Name", "")),
                "last_name": str(r.get("Last_Name", "")),
                "team_id": str(team_id).upper(),
                "team_name": str(r.get("Team", "")),
                "position": pos,
                "position_norm": pos,
                "pos_norm": pos,
                "games_played": 0,
                "season": date.today().year,
                "game_number": 0,
                "game_date_utc": pd.Timestamp.now(tz="UTC"),
                "usage_multiplier": default_usage,
                "synthetic_current_roster": 1,
                "share_goals_ewm": pos_defaults.get("goals_share", 0.03) * 0.35,
                "share_assists_ewm": pos_defaults.get("assists_share", 0.03) * 0.35,
                "share_shots_ewm": pos_defaults.get("shots_share", 0.03) * 0.35,
                "career_goals_pg": 0.0,
                "career_assists_pg": 0.0,
                "career_shots_pg": 0.0,
                "sog_rate_ewm": LG_SOG_RATE,
                "shot_pct_ewm": LG_SHOT_PCT,
                "two_pt_rate_ewm": LG_2PT_RATE,
                "ground_balls_ewm": pos_defaults.get("gb_pg", 1.0),
                "turnovers_ewm": pos_defaults.get("to_pg", 0.5),
                "caused_turnovers_ewm": pos_defaults.get("cto_pg", 0.2),
                "zero_rate_goals": ZERO_RATE.get(f"{pos}_goals", 0.75),
                "zero_rate_assists": ZERO_RATE.get(f"{pos}_assists", 0.80),
                "bayes_fo_pct": LG_FO_PCT,
                "bayes_save_pct": LG_SAVE_PCT,
                "saves_ewm": LG_SAVES if pos == "G" else 0.0,
                "fo_wins_ewm": LG_FOS_PER_GAME * 0.5 if pos == "FO" else 0.0,
            })

        return pd.DataFrame(rows)

    def filter_player_rows(self, team_id: str, roster_latest: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Dict[str, object]]:
        team_id = str(team_id).upper()
        official_keys = self.team_name_keys(team_id)
        detail: Dict[str, object] = {
            "team_id": team_id,
            "official_filter_available": self.available,
            "official_roster_count": int(len(official_keys)),
            "historical_candidate_count": int(len(roster_latest)),
            "matched_count": 0,
            "applied": False,
            "reason": "not_attempted",
        }

        if not self.available:
            detail["reason"] = self.status.get("reason", "official roster filter unavailable")
            return None, detail

        if not official_keys:
            detail["reason"] = "No official roster rows for this engine team_id."
            return None, detail

        if roster_latest.empty or "full_name" not in roster_latest.columns:
            detail["reason"] = "No historical projection candidates or missing full_name column."
            return None, detail

        tmp = roster_latest.copy()
        tmp["_name_key"] = tmp["full_name"].map(_norm_person_name)
        matched = tmp[tmp["_name_key"].isin(official_keys)].drop(columns=["_name_key"], errors="ignore").copy()

        detail["matched_count"] = int(len(matched))
        # Use a deliberately low threshold because rookies/new adds may not have historical PLL stats yet.
        min_needed = min(8, max(4, int(len(official_keys) * 0.25)))
        if len(matched) >= min_needed:
            detail["applied"] = True
            detail["reason"] = "official_current_roster_csv"
            return matched, detail

        detail["reason"] = f"Only matched {len(matched)} current roster players; minimum needed is {min_needed}."
        return None, detail

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _nan(x, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _safe_div(n: float, d: float, fill: float = 0.0) -> float:
    if not d or not np.isfinite(d):
        return fill
    return float(n / d) if np.isfinite(n) else fill


def _season_w(season: int, current: int) -> float:
    return float(0.5 ** ((current - season) / SEASON_HALFLIFE))


def _ewm_shift(series: pd.Series, halflife: int) -> pd.Series:
    """Leakage-safe EWM: shift before smoothing."""
    return series.shift(1).ewm(halflife=halflife, min_periods=1).mean()


def _rolling_stat(series: pd.Series, window: int = 10) -> pd.Series:
    """Shift-safe expanding-window rolling stat."""
    return series.shift(1).expanding(min_periods=1)


def _safe_div_s(num: pd.Series, den: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, num / den, 0.0)
    return pd.Series(out, index=num.index).fillna(0.0)


def _nearest_half(v: float) -> float:
    return round(v * 2.0) / 2.0


def _bayesian_rate(s: float, n: float, a: float = 2.0, b: float = 2.0) -> float:
    denom = n + a + b
    if denom <= 0:
        return a / (a + b)
    return float((s + a) / denom)


def _negbinom_params(mu: float, phi: float) -> Tuple[int, float]:
    mu = max(mu, 0.01)
    phi = max(phi, 0.1)
    n = max(int(round(phi)), 1)
    p = phi / (mu + phi)
    return n, min(max(p, 1e-6), 1 - 1e-6)


def _trunc_normal_sample(rng: np.random.Generator, mu: float, sigma: float, n: int,
                          lo: float = 0.0) -> np.ndarray:
    """Sample from Normal(mu, sigma) truncated below at lo."""
    raw = rng.normal(mu, max(sigma, 0.5), n)
    raw = np.maximum(raw, lo)
    return raw


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TeamProjection:
    team_id: str
    team_name: str
    proj_goals: float
    proj_scores: float
    proj_shots: float
    proj_sog: float
    proj_faceoff_wins: float
    proj_faceoff_pct: float
    proj_saves: float
    proj_shots_faced: float
    proj_save_pct: float
    proj_turnovers: float
    proj_caused_turnovers: float
    proj_ground_balls: float
    proj_assists: float
    proj_2pt_goals: float
    proj_1pt_goals: float
    proj_osp: float
    proj_top_sec: float
    confidence: float
    model_used: str = "v3"


@dataclass
class PlayerProjection:
    player_id: str
    full_name: str
    team_id: str
    position: str
    proj_goals: float
    proj_1pt_goals: float
    proj_2pt_goals: float
    proj_assists: float
    proj_points: float
    proj_shots: float
    proj_sog: float
    proj_ground_balls: float
    proj_turnovers: float
    proj_caused_turnovers: float
    proj_saves: float
    proj_save_pct: float
    proj_faceoff_wins: float
    proj_faceoff_pct: float
    zero_prob_goals: float = 0.0
    zero_prob_assists: float = 0.0
    confidence: float = 0.5
    active: bool = True
    usage_multiplier: float = 1.0
    is_starter: bool = False


@dataclass
class GameSimulation:
    n_sims: int
    home_goals: np.ndarray
    away_goals: np.ndarray
    home_scores: np.ndarray
    away_scores: np.ndarray
    home_win_prob: float
    away_win_prob: float
    tie_prob: float
    expected_total: float
    spread_home: float
    total_distribution: np.ndarray
    margin_distribution: np.ndarray


@dataclass
class PlayerSimulation:
    player_id: str
    full_name: str
    stat_distributions: Dict[str, np.ndarray]
    proj_values: Dict[str, float]
    prop_lines: Dict[str, float]


@dataclass
class MarketLine:
    stat: str
    line: float
    fair_over_prob: float
    fair_under_prob: float
    over_odds: str
    under_odds: str
    juice: float


@dataclass
class GameMarket:
    home_ml: str
    away_ml: str
    home_win_prob: float
    away_win_prob: float
    spread_home: float
    spread_home_odds: str
    spread_away_odds: str
    total_line: float
    over_odds: str
    under_odds: str


@dataclass
class BacktestResult:
    n_games: int
    mae_home_goals: float
    mae_away_goals: float
    mae_total_goals: float
    rmse_total_goals: float
    brier_score: float
    correct_winner_pct: float
    mae_total_scores: float
    bias_total_goals: float
    calibration_table: pd.DataFrame
    raw_rows: List[Dict]


@dataclass
class ProjectionResult:
    game_id: str
    home_team: str
    away_team: str
    home_proj: TeamProjection
    away_proj: TeamProjection
    home_players: List[PlayerProjection]
    away_players: List[PlayerProjection]
    game_sim: GameSimulation
    home_player_sims: List[PlayerSimulation]
    away_player_sims: List[PlayerSimulation]
    game_market: GameMarket
    player_markets: Dict[str, Dict]
    generated_at: str


# ---------------------------------------------------------------------------
# Class 1: DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    _DEFAULT_RELATIVE = Path("data") / "analytics_database" / "pll_warehouse.duckdb"

    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            self.db_path = Path(db_path)
        elif os.environ.get("PLL_DB_PATH"):
            self.db_path = Path(os.environ["PLL_DB_PATH"])
        else:
            self.db_path = Path(__file__).parent.resolve() / self._DEFAULT_RELATIVE
        if not self.db_path.exists():
            raise FileNotFoundError(f"DuckDB not found: {self.db_path}")
        logger.info("DataLoader: %s", self.db_path)

    def _conn(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path), read_only=True)

    def load_team_games(self) -> pd.DataFrame:
        con = self._conn()
        try:
            df = con.execute(
                "SELECT * FROM clean.team_game_stats ORDER BY season, game_number, team_id"
            ).df()
            # Add clean faceoff denominator and shots_faced
            df["fo_denom"] = (df["faceoffs_won"].fillna(0) + df["faceoffs_lost"].fillna(0)).clip(lower=1)
            df["fo_pct_clean"] = df["faceoffs_won"].fillna(0) / df["fo_denom"]
            df["shots_faced"] = df["saves"].fillna(0) + df["goals_against"].fillna(0)
            df["save_pct_clean"] = _safe_div_s(
                df["saves"].fillna(0),
                df["shots_faced"].clip(lower=0.1)
            )
            logger.info("Loaded %d team-game rows", len(df))
            return df
        finally:
            con.close()

    def load_player_games(self) -> pd.DataFrame:
        con = self._conn()
        try:
            df = con.execute(
                "SELECT * FROM clean.player_game_stats ORDER BY season, game_number, team_id, player_id"
            ).df()
            df["position_norm"] = df["position"].fillna("M").apply(_norm_pos)
            df["fo_denom_p"] = (df["faceoffs_won"].fillna(0) + df["faceoffs_lost"].fillna(0)).clip(lower=1)
            df["shots_faced_p"] = df["saves"].fillna(0) + df["goals_against"].fillna(0)
            logger.info("Loaded %d player-game rows", len(df))
            return df
        finally:
            con.close()

    def load_schedule(self, include_completed: bool = False) -> pd.DataFrame:
        con = self._conn()
        try:
            if include_completed:
                df = con.execute("SELECT * FROM clean.game_schedule_all").df()
            else:
                df = con.execute("""
                    SELECT * FROM clean.game_schedule_all
                    WHERE LOWER(COALESCE(event_status_label,'')) NOT IN ('final','completed')
                      AND COALESCE(CAST(event_status AS VARCHAR),'') NOT IN ('3')
                """).df()
            logger.info("Loaded %d schedule rows", len(df))
            return df
        finally:
            con.close()

    def load_player_directory(self) -> pd.DataFrame:
        con = self._conn()
        try:
            return con.execute("SELECT * FROM clean.player_directory").df()
        finally:
            con.close()

    def load_current_rosters(self) -> Tuple[pd.DataFrame, Dict[str, object]]:
        return load_current_rosters_cache(self.db_path)

    def resolve_current_team(self, player_games: pd.DataFrame) -> Dict[str, str]:
        """
        Return {player_id: team_id} based on the most recent season that has data,
        preferring the current/latest season over all-time last game.

        This prevents players who appeared for Team A in 2023 but switched to Team B
        in 2025 from still showing up on Team A's 2026 roster.
        """
        if player_games.empty:
            return {}

        # Use the most recent season with data for each player
        sorted_pg = player_games.sort_values(["season", "game_number"])

        # First try: current season (max season in data)
        max_season = int(sorted_pg["season"].max())
        current_pg = sorted_pg[sorted_pg["season"] == max_season]

        # For each player in current season, get their latest team
        current_map: Dict[str, str] = {}
        if not current_pg.empty:
            latest_current = (
                current_pg.groupby("player_id")[["team_id"]].last().reset_index()
            )
            current_map = dict(zip(
                latest_current["player_id"].astype(str),
                latest_current["team_id"].astype(str),
            ))

        # For players not in current season, fall back to most recent prior season
        prior_pg = sorted_pg[sorted_pg["season"] < max_season]
        if not prior_pg.empty:
            latest_prior = (
                prior_pg.groupby("player_id")[["team_id"]].last().reset_index()
            )
            prior_map = dict(zip(
                latest_prior["player_id"].astype(str),
                latest_prior["team_id"].astype(str),
            ))
            # Only add players NOT already resolved from current season
            for pid, tid in prior_map.items():
                if pid not in current_map:
                    current_map[pid] = tid

        return current_map


# ---------------------------------------------------------------------------
# Class 2: RatingBuilder
# ---------------------------------------------------------------------------

class RatingBuilder:
    """
    Builds rich, leakage-safe per-(team|player, game) ratings.

    For each stat: mean, median, EWM (half-life tuned per stat), std, IQR.
    All computed via shift(1) so game N uses only games 0..N-1.

    Team ratings also include possession chain metrics:
        fo_pct -> top_sec -> osp -> shots -> goals
    """

    def __init__(self, team_games: pd.DataFrame, player_games: pd.DataFrame):
        self.tg = team_games.copy()
        self.pg = player_games.copy()
        self._current_season = int(team_games["season"].max()) if len(team_games) else 2026
        self._tr: Optional[pd.DataFrame] = None   # team ratings
        self._pr: Optional[pd.DataFrame] = None   # player ratings

        # League means from data (used as Bayesian priors only, never forced)
        self._lg = {
            "goals": float(self.tg["goals"].mean()) if len(self.tg) else LG_GOALS,
            "shots": float(self.tg["shots"].mean()) if len(self.tg) else LG_SHOTS,
            "sog": float(self.tg["shots_on_goal"].mean()) if len(self.tg) else LG_SOG,
            "saves": float(self.tg["saves"].mean()) if len(self.tg) else LG_SAVES,
            "shots_faced": float(self.tg["shots_faced"].mean()) if len(self.tg) else LG_SHOTS_FACED,
            "fo_pct": LG_FO_PCT,
            "to": float(self.tg["turnovers"].mean()) if len(self.tg) else LG_TO,
            "gb": float(self.tg["ground_balls"].mean()) if len(self.tg) else LG_GB,
            "top": LG_TOP_SEC,
            "osp": LG_OSP,
            "touches": LG_TOUCHES,
        }

    # ── Team ratings ──────────────────────────────────────────────────────

    def build_team_ratings(self) -> pd.DataFrame:
        chunks = []
        for tid, df in self.tg.groupby("team_id", sort=False):
            df = df.sort_values(["season", "game_number"]).reset_index(drop=True)
            chunks.append(self._team_chunk(df))
        if not chunks:
            self._tr = pd.DataFrame()
            return self._tr
        result = pd.concat(chunks, ignore_index=True).sort_values(
            ["season", "game_number", "team_id"]
        ).reset_index(drop=True)
        self._tr = result
        logger.info("Built team ratings: %d rows x %d cols", result.shape[0], result.shape[1])
        return self._tr

    def _team_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        lg = self._lg

        def _stat_block(series: pd.Series, fill: float, hl: int, name: str) -> pd.DataFrame:
            """Build mean/median/ewm/std/iqr for one stat, all leakage-safe."""
            s = series.fillna(fill)
            exp = s.shift(1).expanding(min_periods=1)
            ewm_s = _ewm_shift(s, hl)
            return pd.DataFrame({
                f"{name}_mean":   exp.mean().fillna(fill),
                f"{name}_median": exp.median().fillna(fill),
                f"{name}_ewm":    ewm_s.fillna(fill),
                f"{name}_std":    exp.std().fillna(1.0).clip(lower=0.1),
                f"{name}_iqr":    (exp.quantile(0.75) - exp.quantile(0.25)).fillna(1.0).clip(lower=0.1),
                f"{name}_ewm_std": s.shift(1).ewm(halflife=hl, min_periods=2).std().fillna(1.0).clip(lower=0.1),
            }, index=df.index)

        blocks = []

        # ── Offensive stats
        blocks.append(_stat_block(df["goals"],         lg["goals"], HL_GOALS, "goals"))
        blocks.append(_stat_block(df["shots"],         lg["shots"], HL_SHOTS, "shots"))
        blocks.append(_stat_block(df["shots_on_goal"], lg["sog"],   HL_SHOTS, "sog"))
        blocks.append(_stat_block(df["assists"],       lg["goals"] * 0.65, HL_GOALS, "assists"))

        twoptr = _safe_div_s(
            df.get("two_point_goals", pd.Series(0, index=df.index)).fillna(0),
            df["goals"].clip(lower=1)
        )
        blocks.append(_stat_block(twoptr, LG_2PT_RATE, HL_GOALS, "two_pt_rate"))

        shot_pct = _safe_div_s(df["goals"].fillna(0), df["shots"].clip(lower=1))
        blocks.append(_stat_block(shot_pct, LG_SHOT_PCT, HL_SHOTS, "shot_pct"))

        sog_rate = _safe_div_s(df["shots_on_goal"].fillna(0), df["shots"].clip(lower=1))
        blocks.append(_stat_block(sog_rate, LG_SOG_RATE, HL_SHOTS, "sog_rate"))

        # ── Defensive stats
        blocks.append(_stat_block(df["goals_against"].fillna(lg["goals"]), lg["goals"], HL_GOALS, "goals_against"))
        blocks.append(_stat_block(df["shots_faced"],   lg["shots_faced"], HL_SHOTS, "shots_faced"))

        sp_clean = df["save_pct_clean"].fillna(LG_SAVE_PCT)
        blocks.append(_stat_block(sp_clean, LG_SAVE_PCT, HL_SHOTS, "save_pct"))
        blocks.append(_stat_block(df["saves"].fillna(lg["saves"]), lg["saves"], HL_SHOTS, "saves"))

        # ── Possession chain
        top = df["time_in_possession"].fillna(0)
        top_valid = top.where(top > 0).ffill().fillna(lg["top"])
        blocks.append(_stat_block(top_valid, lg["top"], HL_POSS, "top_sec"))

        blocks.append(_stat_block(df["touches"].fillna(lg["touches"]), lg["touches"], HL_POSS, "touches"))
        blocks.append(_stat_block(df["total_passes"].fillna(LG_PASSES), LG_PASSES, HL_POSS, "passes"))

        osp = df["offensive_sequence_proxy"].fillna(lg["osp"])
        blocks.append(_stat_block(osp, lg["osp"], HL_POSS, "osp"))

        blocks.append(_stat_block(df["turnovers"].fillna(lg["to"]), lg["to"], HL_TO, "turnovers"))
        blocks.append(_stat_block(df["caused_turnovers"].fillna(5.0), 5.0, HL_TO, "caused_turnovers"))
        blocks.append(_stat_block(df["ground_balls"].fillna(lg["gb"]), lg["gb"], HL_SHOTS, "ground_balls"))

        # ── FO (highest autocorrelation = longest half-life)
        blocks.append(_stat_block(df["fo_pct_clean"], LG_FO_PCT, HL_FO, "fo_pct"))
        blocks.append(_stat_block(df["faceoffs_won"].fillna(LG_FOS_PER_GAME * 0.5),
                                  LG_FOS_PER_GAME * 0.5, HL_FO, "fo_wins"))

        # ── Bayesian career rates (cumulative, most conservative estimate)
        fo_w = df["faceoffs_won"].fillna(0).shift(1).cumsum().fillna(0)
        fo_t = df["fo_denom"].shift(1).cumsum().fillna(0)
        df["bayes_fo_pct"] = [_bayesian_rate(w, t, 2, 2) for w, t in zip(fo_w, fo_t)]

        sv_n = df["saves"].fillna(0).shift(1).cumsum().fillna(0)
        sv_d = df["shots_faced"].shift(1).cumsum().fillna(0)
        df["bayes_save_pct"] = [_bayesian_rate(s, n, 3, 3) for s, n in zip(sv_n, sv_d)]

        g_n = df["goals"].fillna(0).shift(1).cumsum().fillna(0)
        g_d = df["shots"].fillna(0).shift(1).cumsum().fillna(0)
        df["bayes_shot_pct"] = [_bayesian_rate(g, s, 4, 10) for g, s in zip(g_n, g_d)]

        # ── Context
        df["games_played"] = df.groupby("season").cumcount()
        df["season_weight"] = df["season"].apply(lambda s: _season_w(int(s), self._current_season))

        # Merge all blocks efficiently
        extra = pd.concat(blocks, axis=1)
        result = pd.concat([df.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        return result

    # ── Player ratings ────────────────────────────────────────────────────

    def build_player_ratings(self) -> pd.DataFrame:
        if self.pg.empty:
            self._pr = pd.DataFrame()
            return self._pr

        pg = self.pg.copy()
        pg["game_date_utc"] = pd.to_datetime(pg["game_date_utc"], utc=True, errors="coerce")

        # Merge team totals for share computation
        tg_totals = self.tg[["game_id", "team_id"] + [
            c for c in ("goals", "assists", "shots") if c in self.tg.columns
        ]].rename(columns={"goals": "team_goals", "assists": "team_assists", "shots": "team_shots"})
        pg = pg.merge(tg_totals, on=["game_id", "team_id"], how="left")

        chunks = []
        for pid, df in pg.groupby("player_id", sort=False):
            df = df.sort_values(["season", "game_date_utc", "game_number"],
                                na_position="last").reset_index(drop=True)
            chunks.append(self._player_chunk(df))

        if not chunks:
            self._pr = pd.DataFrame()
            return self._pr

        result = pd.concat(chunks, ignore_index=True)
        num_cols = result.select_dtypes(include=[np.number]).columns
        result[num_cols] = result[num_cols].fillna(0)
        self._pr = result
        logger.info("Built player ratings: %d rows x %d cols", result.shape[0], result.shape[1])
        return self._pr

    def _player_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        pos_raw = str(
            df["position_norm"].dropna().iloc[0]
            if "position_norm" in df.columns and df["position_norm"].notna().any()
            else "M"
        )
        pos = _norm_pos(pos_raw)
        df["pos_norm"] = pos

        def _pstat(series: pd.Series, fill: float, hl: int, name: str):
            s = series.fillna(fill)
            exp = s.shift(1).expanding(min_periods=1)
            return pd.DataFrame({
                f"{name}_mean": exp.mean().fillna(fill),
                f"{name}_median": exp.median().fillna(fill),
                f"{name}_ewm": _ewm_shift(s, hl).fillna(fill),
                f"{name}_std": exp.std().fillna(0.5).clip(lower=0.05),
            }, index=df.index)

        blocks = []
        blocks.append(_pstat(df["goals"].fillna(0), 0, HL_GOALS, "goals"))
        blocks.append(_pstat(df["assists"].fillna(0), 0, HL_GOALS, "assists"))
        blocks.append(_pstat(df["shots"].fillna(0), 0, HL_SHOTS, "shots"))
        blocks.append(_pstat(df["shots_on_goal"].fillna(0), 0, HL_SHOTS, "sog"))
        blocks.append(_pstat(df["ground_balls"].fillna(0), 0, HL_SHOTS, "ground_balls"))
        blocks.append(_pstat(df["turnovers"].fillna(0), 0, HL_TO, "turnovers"))
        blocks.append(_pstat(df["caused_turnovers"].fillna(0), 0, HL_TO, "caused_turnovers"))

        two_pt = df.get("two_point_goals", pd.Series(0, index=df.index)).fillna(0)
        two_rate = _safe_div_s(two_pt, df["goals"].clip(lower=1))
        blocks.append(_pstat(two_rate, LG_2PT_RATE, HL_GOALS, "two_pt_rate"))

        sp = _safe_div_s(df["goals"].fillna(0), df["shots"].clip(lower=1))
        blocks.append(_pstat(sp, LG_SHOT_PCT, HL_SHOTS, "shot_pct"))

        sogr = _safe_div_s(df["shots_on_goal"].fillna(0), df["shots"].clip(lower=1))
        blocks.append(_pstat(sogr, LG_SOG_RATE, HL_SHOTS, "sog_rate"))

        # Share features (player's share of team totals)
        for stat, fill_team in [("goals", LG_GOALS), ("assists", LG_GOALS * 0.65), ("shots", LG_SHOTS)]:
            if stat not in df.columns:
                df[f"share_{stat}_ewm"] = POS_DEFAULTS.get(pos, {}).get(f"{stat}_share", 0.05)
                continue
            tcol = f"team_{stat}"
            denom = df[tcol].fillna(fill_team).clip(lower=1) if tcol in df.columns else pd.Series(fill_team, index=df.index)
            share = _safe_div_s(df[stat].fillna(0), denom)
            df[f"share_{stat}_ewm"] = _ewm_shift(share, HL_GOALS).fillna(
                POS_DEFAULTS.get(pos, {}).get(f"{stat}_share", 0.05)
            )
            df[f"career_{stat}_pg"] = df[stat].shift(1).expanding(min_periods=1).mean().fillna(0)

        # Zero-inflation: fraction of games player scored 0
        for stat in ["goals", "assists"]:
            if stat in df.columns:
                zero_series = (df[stat].fillna(0) == 0).astype(float)
                df[f"zero_rate_{stat}"] = zero_series.shift(1).expanding(min_periods=1).mean().fillna(
                    ZERO_RATE.get(f"{pos}_{stat}", 0.6)
                )
            else:
                df[f"zero_rate_{stat}"] = ZERO_RATE.get(f"{pos}_{stat}", 0.6)

        # Bayesian rates
        fo_w = df.get("faceoffs_won", pd.Series(0, index=df.index)).fillna(0).shift(1).cumsum().fillna(0)
        fo_t = df["fo_denom_p"].shift(1).cumsum().fillna(0)
        df["bayes_fo_pct"] = [_bayesian_rate(w, t, 2, 2) for w, t in zip(fo_w, fo_t)]

        sv_n = df.get("saves", pd.Series(0, index=df.index)).fillna(0).shift(1).cumsum().fillna(0)
        sv_d = df.get("shots_faced_p", pd.Series(0, index=df.index)).fillna(0).shift(1).cumsum().fillna(0)
        df["bayes_save_pct"] = [_bayesian_rate(s, n, 3, 3) for s, n in zip(sv_n, sv_d)]

        df["games_played"] = df.groupby("season").cumcount()

        extra = pd.concat(blocks, axis=1)
        result = pd.concat([df.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        return result

    # ── Retrieval helpers ─────────────────────────────────────────────────

    def get_team_rating(self, team_id: str, as_of_date=None) -> Dict:
        if self._tr is None:
            self.build_team_ratings()
        df = self._tr
        if df is None or df.empty:
            return {}

        mask = df["team_id"].astype(str) == str(team_id)

        # Parquet/DuckDB can return game_date_utc as Arrow string dtype on
        # Streamlit Cloud/Python 3.14. Always coerce both sides to UTC datetime
        # before filtering, otherwise pandas raises:
        #   TypeError: Invalid comparison between dtype=string and Timestamp
        if as_of_date and "game_date_utc" in df.columns:
            cutoff = pd.to_datetime(as_of_date, utc=True, errors="coerce")
            if pd.notna(cutoff):
                game_dates = pd.to_datetime(df["game_date_utc"], utc=True, errors="coerce")
                mask &= game_dates.lt(cutoff).fillna(False)

        sub = df[mask].copy()
        if sub.empty:
            logger.warning("No ratings for team_id=%s", team_id)
            return {}

        if "game_date_utc" in sub.columns:
            sub["_sort_game_date_utc"] = pd.to_datetime(sub["game_date_utc"], utc=True, errors="coerce")
            sort_cols = [c for c in ["season", "_sort_game_date_utc", "game_number"] if c in sub.columns]
        else:
            sort_cols = [c for c in ["season", "game_number"] if c in sub.columns]

        row = sub.sort_values(sort_cols, na_position="last").iloc[-1]
        out = row.drop(labels=["_sort_game_date_utc"], errors="ignore").to_dict()
        return {k: _nan(v) if isinstance(v, float) else v for k, v in out.items()}

    def get_player_rating(self, player_id: str, as_of_date=None) -> Dict:
        if self._pr is None:
            self.build_player_ratings()
        df = self._pr
        if df is None or df.empty:
            return {}

        mask = df["player_id"].astype(str) == str(player_id)

        # Same defensive date coercion as get_team_rating(). This keeps
        # historical rating lookups safe regardless of whether the column
        # is stored as string, object, date, or datetime.
        if as_of_date and "game_date_utc" in df.columns:
            cutoff = pd.to_datetime(as_of_date, utc=True, errors="coerce")
            if pd.notna(cutoff):
                game_dates = pd.to_datetime(df["game_date_utc"], utc=True, errors="coerce")
                mask &= game_dates.lt(cutoff).fillna(False)

        sub = df[mask].copy()
        if sub.empty:
            return {}

        if "game_date_utc" in sub.columns:
            sub["_sort_game_date_utc"] = pd.to_datetime(sub["game_date_utc"], utc=True, errors="coerce")
            sort_cols = [c for c in ["season", "_sort_game_date_utc", "game_number"] if c in sub.columns]
        else:
            sort_cols = [c for c in ["season", "game_number"] if c in sub.columns]

        row = sub.sort_values(sort_cols, na_position="last").iloc[-1]
        out = row.drop(labels=["_sort_game_date_utc"], errors="ignore").to_dict()
        return {k: _nan(v) if isinstance(v, float) else v for k, v in out.items()}


# ---------------------------------------------------------------------------
# Class 3: TeamModel  (possession chain)
# ---------------------------------------------------------------------------

class TeamModel:
    """
    Possession-chain projection model.

    Stage 1 — FO rating  →  projected FO win%
    Stage 2 — FO% + team possession style  →  proj offensive sequences (OSP)
    Stage 3 — OSP + shot volume rating  →  proj shots, SOG
    Stage 4 — shots + shot quality rating  →  proj goals
    Stage 5 — goals + 2pt rate + assist rate  →  scores, assists
    Stage 6 — opponent defensive adjustment at shots stage (not goals)

    No forced regression to league mean.
    No hardcoded home advantage.
    Opponent quality adjusts the shot-to-goal conversion stage.
    """

    # Ridge input features — strictly pre-game ratings
    _RIDGE_FEATS = [
        "goals_ewm", "goals_mean", "shot_pct_ewm", "sog_rate_ewm",
        "shots_ewm", "sog_ewm", "osp_ewm", "fo_pct_ewm",
        "turnovers_ewm", "ground_balls_ewm", "top_sec_ewm",
        "goals_against_ewm", "save_pct_ewm",
        "bayes_fo_pct", "bayes_shot_pct",
    ]

    def __init__(self):
        self._ridge: Dict[str, RidgeCV] = {}
        self._scalers: Dict[str, StandardScaler] = {}
        self._fitted = False
        self._lg: Dict[str, float] = {}

    def fit(self, ratings_df: pd.DataFrame) -> None:
        if ratings_df.empty:
            logger.warning("TeamModel.fit: empty frame")
            return

        # Compute league averages from the training data
        for col, key in [
            ("goals", "goals"), ("shots", "shots"), ("shots_on_goal", "sog"),
            ("goals_against", "goals_against"), ("saves", "saves"),
            ("turnovers", "to"), ("ground_balls", "gb"),
            ("osp_ewm", "osp"), ("fo_pct_clean", "fo_pct"),
            ("shots_faced", "shots_faced"),
        ]:
            src = col if col in ratings_df.columns else (key if key in ratings_df.columns else None)
            if src and src in ratings_df.columns:
                self._lg[key] = float(ratings_df[src].mean())
            else:
                self._lg[key] = getattr(
                    __builtins__ if hasattr(__builtins__, "globals") else None,
                    f"LG_{key.upper()}", 0
                ) or LG_GOALS

        avail = [c for c in self._RIDGE_FEATS if c in ratings_df.columns]
        if len(avail) < 3:
            logger.warning("Not enough Ridge features available")
            return

        X_raw = (
            ratings_df[avail]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0).replace([np.inf, -np.inf], 0)
        )
        weights = ratings_df.get("season_weight", pd.Series(1.0, index=ratings_df.index)).fillna(1.0)

        targets = {
            "goals":    ratings_df.get("goals",         pd.Series()),
            "shots":    ratings_df.get("shots",         pd.Series()),
            "sog":      ratings_df.get("shots_on_goal", pd.Series()),
            "turnovers": ratings_df.get("turnovers",    pd.Series()),
            "gb":       ratings_df.get("ground_balls",  pd.Series()),
            "assists":  ratings_df.get("assists",       pd.Series()),
        }

        for stat, y_full in targets.items():
            if y_full is None or len(y_full) == 0:
                continue
            y = y_full.reindex(ratings_df.index).fillna(
                self._lg.get(stat, LG_GOALS)
            )
            valid = y.notna() & X_raw.notna().all(axis=1)
            if valid.sum() < 20:
                continue
            try:
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X_raw[valid].values)
                mdl = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
                mdl.fit(Xs, y[valid].values, sample_weight=weights[valid].values)
                self._ridge[stat] = mdl
                self._scalers[stat] = scaler
            except Exception as exc:
                logger.debug("Ridge %s failed: %s", stat, exc)

        self._fitted = True
        logger.info("TeamModel fitted: %s", sorted(self._ridge.keys()))

    def predict(
        self,
        team_r: Dict,
        opp_r: Dict,
        team_adj: Optional[Dict] = None,
    ) -> TeamProjection:
        """
        Project stats for one team.

        Parameters
        ----------
        team_r   : rating dict for the projecting team
        opp_r    : rating dict for the opponent
        team_adj : optional {"off_mult": float, "def_mult": float} overrides
        """
        adj = team_adj or {}
        off_mult = float(adj.get("off_mult", 1.0))
        def_mult_opp = float(adj.get("def_mult_opp", 1.0))  # applied to opponent defense

        lg_goals = self._lg.get("goals", LG_GOALS)
        lg_shots = self._lg.get("shots", LG_SHOTS)
        lg_sog = self._lg.get("sog", LG_SOG)

        # ── Stage 1: FO projection ──────────────────────────────────────
        fo_pct = _nan(float(team_r.get("bayes_fo_pct", team_r.get("fo_pct_ewm", LG_FO_PCT))), LG_FO_PCT)
        fo_pct = min(max(fo_pct, 0.25), 0.75)
        proj_fo_wins = fo_pct * LG_FOS_PER_GAME

        # ── Stage 2: Possession chain ────────────────────────────────────
        # Possession time driven by FO% (r_fo_top=0.504) + team possession style
        team_top_base = _nan(float(team_r.get("top_sec_ewm", LG_TOP_SEC)), LG_TOP_SEC)
        # FO effect on possession: modest — r=0.504 means strong but not total driver
        fo_possession_delta = (fo_pct - LG_FO_PCT) * LG_TOP_SEC * 0.40
        proj_top = max(team_top_base + fo_possession_delta, 500.0)

        # OSP: use team's EWM directly (more reliable than deriving from TOP)
        # TOP just adds context to OSP via the FO adjustment
        team_osp_ewm = _nan(float(team_r.get("osp_ewm", LG_OSP)), LG_OSP)
        osp_top_adj = (proj_top / max(team_top_base, 500.0))  # ratio of projected vs historical TOP
        osp_top_adj = min(max(osp_top_adj, 0.85), 1.15)       # cap at ±15%
        proj_osp = team_osp_ewm * osp_top_adj

        # ── Stage 3: Shots ────────────────────────────────────────────────
        # Use team's shots EWM directly, adjusted by FO-driven possession edge.
        # Chaining through OSP introduced noise; shots EWM (autocorr 0.124) is
        # a more reliable direct signal than deriving shots from OSP.
        team_shots_ewm = _nan(float(team_r.get("shots_ewm", lg_shots)), lg_shots)
        team_shots_mean = _nan(float(team_r.get("shots_mean", lg_shots)), lg_shots)
        # Blend EWM (recent form) 60% + career mean 40% for stability
        blended_shots_base = 0.60 * team_shots_ewm + 0.40 * team_shots_mean

        # FO drives possession which drives shot volume: FO 0.30 autocorr
        # A team at 60% FO (vs 50% avg) gets ~10% more possession, ~5-7% more shots
        fo_shots_adj = 1.0 + (fo_pct - LG_FO_PCT) * 0.50
        fo_shots_adj = min(max(fo_shots_adj, 0.85), 1.15)
        proj_shots = max(blended_shots_base * fo_shots_adj * off_mult, 5.0)

        # SOG from shots × team's SOG rate (quality of shot selection)
        sog_rate = _nan(float(team_r.get("sog_rate_ewm", LG_SOG_RATE)), LG_SOG_RATE)
        sog_rate = min(max(sog_rate, 0.40), 0.85)
        proj_sog = proj_shots * sog_rate

        # ── Stage 4: Goals — two paths that should agree ──────────────────
        #
        # Path A: shots × shot_pct (goals/shots = 0.274 at league avg)
        #   This is CORRECT — shot% is defined on all shots, not just SOG.
        #   BUG IN PRIOR VERSION: applied shot_pct to SOG, giving ~7 goals.
        #   Correct: 41 shots × 0.274 = 11.2 goals ✓
        #
        # Path B: SOG × goals_per_SOG (goals/SOG = 0.436 at league avg)
        #   Equivalent numerically: 26 SOG × 0.436 = 11.3 goals ✓
        #
        # Blend both paths 50/50 — they are equivalent numerically at league
        # average but diverge at extremes, providing mutual correction.

        bayes_sp = _nan(float(team_r.get("bayes_shot_pct", LG_SHOT_PCT)), LG_SHOT_PCT)
        ewm_sp   = _nan(float(team_r.get("shot_pct_ewm",   LG_SHOT_PCT)), LG_SHOT_PCT)
        blended_sp = 0.55 * bayes_sp + 0.45 * ewm_sp
        blended_sp = min(max(blended_sp, 0.12), 0.55)

        # True goals-per-SOG rate (not shot_pct; different denominator)
        lg_goals_per_sog = lg_goals / max(self._lg.get("sog", LG_SOG), 1.0)

        goals_path_A = proj_shots * blended_sp                # shots × shot%
        goals_path_B = proj_sog   * lg_goals_per_sog          # SOG × goals/SOG

        goals_quality = 0.50 * goals_path_A + 0.50 * goals_path_B

        # Opponent defensive adjustment — applied here at the goals stage (not shots)
        # log5: proj = lg × (team_off/lg) × (opp_def/lg)
        lg_goals = self._lg.get("goals", LG_GOALS)
        team_off_idx = _nan(float(team_r.get("goals_ewm", lg_goals)), lg_goals) / max(lg_goals, 1.0)
        opp_ga       = _nan(float(opp_r.get("goals_against_ewm", lg_goals)), lg_goals)
        opp_def_idx  = (opp_ga * def_mult_opp) / max(lg_goals, 1.0)
        goals_log5   = lg_goals * team_off_idx * opp_def_idx

        # Final goal projection: 60% quality chain + 40% log5 opponent adjustment
        proj_goals_raw = 0.60 * goals_quality + 0.40 * goals_log5

        # ── Ridge correction (25% blend) ─────────────────────────────────
        # Ridge learns residual patterns the chain misses.
        # Kept at 25% to avoid overfitting on 336 training rows.
        if self._fitted:
            combined = {**opp_r, **team_r}
            fv = np.array([_nan(float(combined.get(c, 0))) for c in self._RIDGE_FEATS], dtype=float)
            if "goals" in self._ridge:
                try:
                    sc = self._scalers["goals"]
                    n = sc.n_features_in_
                    ridge_pred = float(self._ridge["goals"].predict(
                        sc.transform(fv[:n].reshape(1, -1))
                    )[0])
                    proj_goals_raw = 0.75 * proj_goals_raw + 0.25 * max(ridge_pred, 0.5)
                except Exception:
                    pass

        proj_goals = max(proj_goals_raw, 0.5) * off_mult

        # ── Stage 5: Derived stats ────────────────────────────────────────
        two_pt_rate = _nan(float(team_r.get("two_pt_rate_ewm", LG_2PT_RATE)), LG_2PT_RATE)
        two_pt_rate = min(max(two_pt_rate, 0.0), 0.40)
        proj_2pt = proj_goals * two_pt_rate
        proj_1pt = max(proj_goals - proj_2pt, 0.0)
        proj_scores = proj_1pt + 2.0 * proj_2pt

        assist_rate = _nan(float(team_r.get("assists_ewm", lg_goals * 0.65)), lg_goals * 0.65) / max(
            _nan(float(team_r.get("goals_ewm", lg_goals)), lg_goals), 0.5
        )
        assist_rate = min(max(assist_rate, 0.30), 1.20)
        proj_assists = proj_goals * assist_rate

        proj_to = max(_nan(float(team_r.get("turnovers_ewm", LG_TO)), LG_TO), 1.0)
        proj_cto = max(_nan(float(team_r.get("caused_turnovers_ewm", 5.0)), 5.0), 0.0)
        proj_gb = max(_nan(float(team_r.get("ground_balls_ewm", LG_GB)), LG_GB), 1.0)

        # Goalie stats (filled by orchestrator using opp projections)
        opp_sog = max(proj_sog * 0.95, 1.0)   # placeholder; replaced later
        opp_save_pct_proj = _nan(float(team_r.get("bayes_save_pct", LG_SAVE_PCT)), LG_SAVE_PCT)
        proj_shots_faced = opp_sog
        proj_saves = opp_sog * opp_save_pct_proj
        proj_save_pct = opp_save_pct_proj

        gp = int(_nan(float(team_r.get("games_played", 0))))
        confidence = round(min(0.45 + 0.03 * gp, 0.90), 3)

        return TeamProjection(
            team_id=str(team_r.get("team_id", "")),
            team_name=str(team_r.get("team_name", "")),
            proj_goals=round(proj_goals, 3),
            proj_scores=round(proj_scores, 3),
            proj_shots=round(proj_shots, 3),
            proj_sog=round(proj_sog, 3),
            proj_faceoff_wins=round(proj_fo_wins, 3),
            proj_faceoff_pct=round(fo_pct, 4),
            proj_saves=round(proj_saves, 3),
            proj_shots_faced=round(proj_shots_faced, 3),
            proj_save_pct=round(proj_save_pct, 4),
            proj_turnovers=round(proj_to, 3),
            proj_caused_turnovers=round(proj_cto, 3),
            proj_ground_balls=round(proj_gb, 3),
            proj_assists=round(proj_assists, 3),
            proj_2pt_goals=round(proj_2pt, 3),
            proj_1pt_goals=round(proj_1pt, 3),
            proj_osp=round(proj_osp, 3),
            proj_top_sec=round(proj_top, 1),
            confidence=confidence,
            model_used="v3_chain" + ("+ridge" if self._fitted else ""),
        )


# ---------------------------------------------------------------------------
# Class 4: PlayerModel
# ---------------------------------------------------------------------------

class PlayerModel:
    """
    Share-based player projection with zero-inflation awareness.
    Blends EWM share + career share + position default.
    """

    def __init__(self, player_ratings: pd.DataFrame,
                 current_team_map: Optional[Dict[str, str]] = None,
                 current_roster_filter: Optional[CurrentRosterFilter] = None):
        self.pr = player_ratings.copy() if player_ratings is not None and not player_ratings.empty else pd.DataFrame()
        self._current_team = current_team_map or {}
        self.current_roster_filter = current_roster_filter
        self.last_roster_filter_details: Dict[str, Dict[str, object]] = {}

    def project_roster(
        self,
        team_id: str,
        team_proj: TeamProjection,
        overrides: Optional[Dict] = None,
        starter_goalie: Optional[str] = None,
        use_current_roster_filter: bool = True,
    ) -> List[PlayerProjection]:
        if self.pr.empty:
            return []
        roster = self.pr[self.pr["team_id"] == team_id]
        if roster.empty:
            return []

        sort_cols = [c for c in ("season", "game_date_utc", "game_number") if c in roster.columns]
        if sort_cols:
            roster = roster.sort_values(sort_cols)
        roster_latest = roster.groupby("player_id").last().reset_index()

        # First preference: official current roster cache. This prevents old team
        # players from appearing in 2026 depth charts just because they have
        # historical rows for this team.
        official_applied = False
        if use_current_roster_filter and self.current_roster_filter is not None:
            filtered, detail = self.current_roster_filter.filter_player_rows(team_id, roster_latest)
            self.last_roster_filter_details[str(team_id)] = detail
            if filtered is not None:
                roster_latest = filtered
                official_applied = True

                # Add official-roster players who have no historical PLL stat row.
                # These show in depth charts with conservative default projections.
                placeholders = self.current_roster_filter.build_placeholder_rows(team_id, roster_latest)
                if placeholders is not None and not placeholders.empty:
                    roster_latest = pd.concat([roster_latest, placeholders], ignore_index=True, sort=False)
                    detail["synthetic_current_roster_added"] = int(len(placeholders))
                    detail["final_projection_roster_count"] = int(len(roster_latest))
                else:
                    detail["synthetic_current_roster_added"] = 0
                    detail["final_projection_roster_count"] = int(len(roster_latest))
                self.last_roster_filter_details[str(team_id)] = detail
        elif not use_current_roster_filter:
            self.last_roster_filter_details[str(team_id)] = {
                "team_id": str(team_id),
                "official_filter_available": bool(self.current_roster_filter and self.current_roster_filter.available),
                "applied": False,
                "reason": "current_roster_filter_disabled_for_historical_game_date",
                "historical_candidate_count": int(len(roster_latest)),
            }

        # Fallback: latest historical team map. This is only used if no official
        # roster cache exists or if name matching fails badly.
        if (not official_applied) and self._current_team:
            roster_latest = roster_latest[
                roster_latest["player_id"].astype(str).map(
                    lambda pid: self._current_team.get(pid, team_id) == team_id
                )
            ].copy()
            prior_detail = self.last_roster_filter_details.get(str(team_id), {})
            reason = prior_detail.get("reason") or "fallback_latest_historical_team_map"
            if reason == "not_attempted":
                reason = "fallback_latest_historical_team_map"
            self.last_roster_filter_details[str(team_id)] = {
                **prior_detail,
                "team_id": str(team_id),
                "official_filter_available": bool(self.current_roster_filter and self.current_roster_filter.available),
                "applied": False,
                "fallback_applied": True,
                "reason": reason if not use_current_roster_filter else "fallback_latest_historical_team_map",
                "historical_candidate_count": int(len(roster_latest)),
                "final_projection_roster_count": int(len(roster_latest)),
            }

        # Share sums across multi-season rosters are typically 1.5-2.5x due to
        # historical players no longer active. The _rescale() call in _reconcile()
        # corrects this after projection — no normalization needed here.

        overrides = overrides or {}
        projections = []

        for _, row in roster_latest.iterrows():
            pid = str(row.get("player_id", ""))
            po = overrides.get(pid, {})
            active = bool(po.get("active", True))
            usage = float(po.get("usage_multiplier", row.get("usage_multiplier", 1.0)))
            is_starter = bool(po.get("is_starter", starter_goalie == pid))

            feats = row.to_dict()
            feats["usage_multiplier"] = usage
            for k, v in po.items():
                if k not in ("active", "usage_multiplier", "is_starter"):
                    feats[k] = v

            proj = self._project_player(feats, team_proj)
            proj.active = active
            proj.usage_multiplier = usage
            proj.is_starter = is_starter

            if not active:
                proj = self._zero(proj)
            projections.append(proj)

        return self._reconcile(projections, team_proj)

    def _project_player(self, f: Dict, tp: TeamProjection) -> PlayerProjection:
        pos = _norm_pos(str(f.get("pos_norm", f.get("position_norm", f.get("position", "M")))))
        pid = str(f.get("player_id", ""))
        name = str(f.get("full_name", pid))
        tid = str(f.get("team_id", ""))
        usage = _nan(float(f.get("usage_multiplier", 1.0)), 1.0)
        gp = int(_nan(float(f.get("games_played", 0))))
        pos_def = POS_DEFAULTS.get(pos, POS_DEFAULTS["M"])

        def _share(stat: str, team_total: float) -> float:
            team_total = max(team_total, 1.0)
            ewm_s = _nan(float(f.get(f"share_{stat}_ewm", 0.0)))
            career_v = _nan(float(f.get(f"career_{stat}_pg", 0.0)))
            career_s = career_v / team_total
            pos_s = pos_def.get(f"{stat}_share", 0.05)
            # Weight blend shifts toward EWM as more data accumulates
            w_ewm = min(0.30 + 0.04 * gp, 0.65)
            w_career = min(0.20 + 0.02 * gp, 0.35)
            w_pos = max(1.0 - w_ewm - w_career, 0.05)
            return max(w_ewm * ewm_s + w_career * career_s + w_pos * pos_s, 0.0)

        # Goals
        if pos == "G":
            proj_goals = 0.0
        else:
            proj_goals = tp.proj_goals * _share("goals", tp.proj_goals) * usage

        # Assists
        if pos == "G":
            proj_assists = 0.0
        else:
            proj_assists = tp.proj_assists * _share("assists", tp.proj_assists) * usage

        # Shots
        if pos == "G":
            proj_shots = max(_nan(float(f.get("shots_ewm", 0.2)), 0.2) * usage, 0.0)
        else:
            proj_shots = tp.proj_shots * _share("shots", tp.proj_shots) * usage

        sog_rate = _nan(float(f.get("sog_rate_ewm", LG_SOG_RATE)), LG_SOG_RATE)
        sog_rate = min(max(sog_rate, 0.20), 1.0)
        proj_sog = proj_shots * sog_rate

        # Ground balls, TOs
        proj_gb = max(_nan(float(f.get("ground_balls_ewm", pos_def.get("gb_pg", 1.0))), pos_def.get("gb_pg", 1.0)) * usage, 0.0)
        proj_to = max(_nan(float(f.get("turnovers_ewm", pos_def.get("to_pg", 0.5))), pos_def.get("to_pg", 0.5)) * usage, 0.0)
        proj_cto = max(_nan(float(f.get("caused_turnovers_ewm", pos_def.get("cto_pg", 0.2))), pos_def.get("cto_pg", 0.2)) * usage, 0.0)

        # 2pt split
        two_rate = _nan(float(f.get("two_pt_rate_ewm", LG_2PT_RATE)), LG_2PT_RATE)
        two_rate = min(max(two_rate, 0.0), 0.45)
        proj_2pt = proj_goals * two_rate
        proj_1pt = max(proj_goals - proj_2pt, 0.0)
        # Correct points formula: 1pt_goals + 2*2pt_goals + assists
        proj_points = proj_1pt + 2.0 * proj_2pt + proj_assists

        # Saves (goalie only) — using correct denominator
        if pos == "G":
            sv_pct = _nan(float(f.get("bayes_save_pct", LG_SAVE_PCT)), LG_SAVE_PCT)
            sv_pct = min(max(sv_pct, 0.30), 0.80)
            proj_saves = tp.proj_sog * sv_pct   # opp SOG assigned by orchestrator
            proj_sv_pct = sv_pct
        else:
            proj_saves = 0.0
            proj_sv_pct = 0.0

        # FO
        if pos == "FO":
            fo_pct = _nan(float(f.get("bayes_fo_pct", LG_FO_PCT)), LG_FO_PCT)
            proj_fo = max(_nan(float(f.get("fo_wins_ewm", LG_FOS_PER_GAME * fo_pct)), 0.0) * usage, 0.0)
            proj_fo_pct = fo_pct
        else:
            proj_fo = 0.0
            proj_fo_pct = 0.0

        # Zero-inflation probabilities
        z_goals = _nan(float(f.get("zero_rate_goals", ZERO_RATE.get(f"{pos}_goals", 0.6))), 0.6)
        z_assists = _nan(float(f.get("zero_rate_assists", ZERO_RATE.get(f"{pos}_assists", 0.75))), 0.75)

        confidence = min(0.40 + 0.025 * gp, 0.85)

        raw = PlayerProjection(
            player_id=pid, full_name=name, team_id=tid, position=pos,
            proj_goals=max(proj_goals, 0.0),
            proj_1pt_goals=max(proj_1pt, 0.0),
            proj_2pt_goals=max(proj_2pt, 0.0),
            proj_assists=max(proj_assists, 0.0),
            proj_points=max(proj_points, 0.0),
            proj_shots=max(proj_shots, 0.0),
            proj_sog=max(proj_sog, 0.0),
            proj_ground_balls=max(proj_gb, 0.0),
            proj_turnovers=max(proj_to, 0.0),
            proj_caused_turnovers=max(proj_cto, 0.0),
            proj_saves=max(proj_saves, 0.0),
            proj_save_pct=max(proj_sv_pct, 0.0),
            proj_faceoff_wins=max(proj_fo, 0.0),
            proj_faceoff_pct=proj_fo_pct,
            zero_prob_goals=z_goals,
            zero_prob_assists=z_assists,
            confidence=confidence,
        )
        return self._apply_caps(raw)

    def _apply_caps(self, p: PlayerProjection) -> PlayerProjection:
        caps = POS_CAPS.get(p.position, {})
        for stat, cap in caps.items():
            attr = f"proj_{stat}"
            if hasattr(p, attr):
                setattr(p, attr, min(getattr(p, attr), cap))
        for attr in vars(p):
            if attr.startswith("proj_") and isinstance(getattr(p, attr), float):
                if getattr(p, attr) < 0:
                    setattr(p, attr, 0.0)
        p.proj_goals = p.proj_1pt_goals + p.proj_2pt_goals
        p.proj_points = p.proj_1pt_goals + 2.0 * p.proj_2pt_goals + p.proj_assists
        return p

    def _reconcile(self, projs: List[PlayerProjection], tp: TeamProjection) -> List[PlayerProjection]:
        """
        Rescale active player projections so their sums match the team projection.

        Uses full proportional rescaling (not 80/20 soft blend). This is necessary
        because the share sums across a multi-season roster are typically 2-3×
        the team total due to historical players no longer on the roster. Full
        rescaling ensures player prop lines are consistent with team-level totals.
        """
        active = [p for p in projs if p.active]
        if not active:
            return projs

        def _rescale(stat: str, team_total: float):
            """
            Proportionally rescale active player projections to match team total.
            Always applies — the share model produces inflated sums from multi-season
            rosters, and full proportional rescaling preserves relative player rankings
            while bringing totals in line.
            """
            s = sum(getattr(p, f"proj_{stat}", 0.0) for p in active)
            if s <= 0 or team_total <= 0:
                return
            scale = team_total / s
            for p in active:
                orig = getattr(p, f"proj_{stat}", 0.0)
                setattr(p, f"proj_{stat}", max(orig * scale, 0.0))

        _rescale("goals", tp.proj_goals)
        _rescale("assists", tp.proj_assists)
        _rescale("shots", tp.proj_shots)
        _rescale("ground_balls", tp.proj_ground_balls)
        _rescale("turnovers", tp.proj_turnovers)

        fo_players = [p for p in active if p.position == "FO"]
        fo_sum = sum(p.proj_faceoff_wins for p in fo_players)
        if fo_players and fo_sum > 0 and tp.proj_faceoff_wins > 0:
            scale = tp.proj_faceoff_wins / fo_sum
            for p in fo_players:
                p.proj_faceoff_wins = max(p.proj_faceoff_wins * scale, 0.0)

        for p in active:
            g = p.proj_goals
            r = p.proj_2pt_goals / g if g > 0 else 0.0
            p.proj_1pt_goals = g * (1.0 - r)
            p.proj_2pt_goals = g * r
            p.proj_points = p.proj_1pt_goals + 2.0 * p.proj_2pt_goals + p.proj_assists
        return projs

    def _zero(self, p: PlayerProjection) -> PlayerProjection:
        for attr in list(vars(p).keys()):
            if attr.startswith("proj_") and isinstance(getattr(p, attr), float):
                setattr(p, attr, 0.0)
        return p


# ---------------------------------------------------------------------------
# Class 5: GameSimulator  (corrected distributions)
# ---------------------------------------------------------------------------

class GameSimulator:
    """
    Monte Carlo simulation with distribution assumptions matched to data.

    Team goals: truncated Normal (var/mean = 0.854, underdispersed)
    Player goals/assists: zero-inflated NegBin (61-78% zero rate)
    Player shots: NegBin (var/mean = 3.03)
    FO wins: Normal (specialists are consistent)
    Saves: NegBin (moderate variance)
    """

    def __init__(self, n_sims: int = N_SIMS, seed: int = 42):
        self.n_sims = n_sims
        self.seed = seed

    def simulate_game(self, home_proj: TeamProjection, away_proj: TeamProjection) -> GameSimulation:
        rng = np.random.default_rng(self.seed)
        n = self.n_sims

        # Team goals: truncated Normal
        # Sigma from data (~3.1) blended with team's own std rating if available
        mu_h = max(float(home_proj.proj_goals), 0.5)
        mu_a = max(float(away_proj.proj_goals), 0.5)

        home_goals = _trunc_normal_sample(rng, mu_h, TEAM_GOAL_SIGMA_BASE, n, lo=0.0)
        away_goals = _trunc_normal_sample(rng, mu_a, TEAM_GOAL_SIGMA_BASE, n, lo=0.0)

        # Floor to non-negative integers for discrete scoring
        home_goals_int = np.floor(home_goals).astype(int).clip(min=0)
        away_goals_int = np.floor(away_goals).astype(int).clip(min=0)

        # 2pt goals: Binomial within each draw
        r_h = min(max(float(home_proj.proj_2pt_goals) / max(mu_h, 1.0), 0.01), 0.45)
        r_a = min(max(float(away_proj.proj_2pt_goals) / max(mu_a, 1.0), 0.01), 0.45)

        two_h = rng.binomial(home_goals_int, r_h)
        two_a = rng.binomial(away_goals_int, r_a)

        home_scores = (home_goals_int - two_h) + 2.0 * two_h
        away_scores = (away_goals_int - two_a) + 2.0 * two_a

        # PLL has overtime — ties are resolved, so true tie rate = 0.
        # Simulated ties (same integer score) are broken by a coin flip,
        # distributing them 50/50 to home and away.
        tied = home_scores == away_scores
        tie_coin = rng.random(n) < 0.5
        home_wins = (home_scores > away_scores) | (tied & tie_coin)
        away_wins = (away_scores > home_scores) | (tied & ~tie_coin)

        home_win_prob = float(np.mean(home_wins))
        away_win_prob = float(np.mean(away_wins))
        tie_prob = 0.0   # resolved by OT
        spread_home = float(np.median(home_scores - away_scores))
        expected_total = float(np.median(home_scores + away_scores))

        return GameSimulation(
            n_sims=n,
            home_goals=home_goals,
            away_goals=away_goals,
            home_scores=home_scores,
            away_scores=away_scores,
            home_win_prob=home_win_prob,
            away_win_prob=away_win_prob,
            tie_prob=tie_prob,
            expected_total=expected_total,
            spread_home=spread_home,
            total_distribution=home_scores + away_scores,
            margin_distribution=home_scores - away_scores,
        )

    def simulate_players(
        self,
        player_projs: List[PlayerProjection],
        team_goal_draws: np.ndarray,
        team_proj_goals: float,
    ) -> List[PlayerSimulation]:
        """
        Simulate player stats. Player goals conditioned on team goal draws.
        Uses zero-inflated NegBin for goals and assists.
        """
        rng = np.random.default_rng(self.seed + 1)
        n = self.n_sims
        active = [p for p in player_projs if p.active]
        results = []

        def _zinb(mu: float, phi_key: str, zero_prob: float) -> np.ndarray:
            """Zero-inflated NegBin draw."""
            mu = max(mu, 0.01)
            nb_n, nb_p = _negbinom_params(mu / max(1.0 - zero_prob, 0.01), PHI_PLAYER.get(phi_key, 2.0))
            is_zero = rng.random(n) < zero_prob
            counts = rng.negative_binomial(nb_n, nb_p, n).astype(float)
            return np.where(is_zero, 0.0, counts)

        def _nb(mu: float, phi_key: str) -> np.ndarray:
            mu = max(mu, 0.01)
            nb_n, nb_p = _negbinom_params(mu, PHI_PLAYER.get(phi_key, 2.0))
            return rng.negative_binomial(nb_n, nb_p, n).astype(float)

        # Collect raw goal draws for conditioning
        raw_goals: Dict[str, np.ndarray] = {}
        raw_assists: Dict[str, np.ndarray] = {}
        raw_shots: Dict[str, np.ndarray] = {}

        field = [p for p in active if p.position != "G"]
        goalies = [p for p in active if p.position == "G"]

        for pp in field:
            raw_goals[pp.player_id] = _zinb(pp.proj_goals, "goals", pp.zero_prob_goals)
            raw_assists[pp.player_id] = _zinb(pp.proj_assists, "assists", pp.zero_prob_assists)
            raw_shots[pp.player_id] = _nb(pp.proj_shots, "shots")

        # Condition field players' goals on team draw
        if field and len(raw_goals) > 0:
            sum_raw = sum(raw_goals[p.player_id] for p in field)
            sum_raw = np.maximum(sum_raw, 0.01)
            team_draw_rounded = np.round(team_goal_draws).clip(min=0)
            scale = team_draw_rounded / sum_raw
            for pp in field:
                raw_goals[pp.player_id] = np.round(
                    raw_goals[pp.player_id] * scale
                ).clip(min=0)

        # Build simulations
        for pp in field:
            pid = pp.player_id
            g = raw_goals.get(pid, np.zeros(n))
            two_rate = pp.proj_2pt_goals / max(pp.proj_goals, 0.01)
            two_rate = min(max(two_rate, 0.01), 0.45)
            t = rng.binomial(g.astype(int), two_rate).astype(float)
            one = np.maximum(g - t, 0)
            a = raw_assists.get(pid, np.zeros(n))
            sh = raw_shots.get(pid, np.zeros(n))
            sog_rate = pp.proj_sog / max(pp.proj_shots, 0.01)
            sog = np.minimum(_nb(pp.proj_sog, "sog"), sh)

            dists: Dict[str, np.ndarray] = {
                "goals": g,
                "one_pt_goals": one,
                "two_pt_goals": t,
                "assists": a,
                # CORRECT points: 1pt + 2*2pt + assists
                "points": one + 2.0 * t + a,
                "shots": sh,
                "shots_on_goal": sog,
                "ground_balls": _nb(pp.proj_ground_balls, "gb"),
            }

            proj_vals = {k: float(np.median(v)) for k, v in dists.items()}
            prop_lines = {k: _nearest_half(float(np.median(v))) for k, v in dists.items()}

            results.append(PlayerSimulation(
                player_id=pid, full_name=pp.full_name,
                stat_distributions=dists, proj_values=proj_vals, prop_lines=prop_lines,
            ))

        for pp in goalies:
            pid = pp.player_id
            sv = _nb(pp.proj_saves, "saves")
            shots_faced = max(pp.proj_saves / max(pp.proj_save_pct, 0.01), 1.0)
            dists = {
                "saves": sv,
                "save_pct": np.where(
                    shots_faced > 0,
                    np.minimum(sv / max(shots_faced, 1.0), 1.0),
                    pp.proj_save_pct,
                ),
            }
            proj_vals = {k: float(np.median(v)) for k, v in dists.items()}
            prop_lines = {"saves": _nearest_half(float(np.median(dists["saves"])))}
            results.append(PlayerSimulation(
                player_id=pid, full_name=pp.full_name,
                stat_distributions=dists, proj_values=proj_vals, prop_lines=prop_lines,
            ))

        for pp in [p for p in active if p.position == "FO"]:
            pid = pp.player_id
            existing = next((r for r in results if r.player_id == pid), None)
            if existing is not None:
                fo_draws = _nb(pp.proj_faceoff_wins, "fo_wins")
                existing.stat_distributions["faceoff_wins"] = fo_draws
                existing.proj_values["faceoff_wins"] = float(np.median(fo_draws))
                existing.prop_lines["faceoff_wins"] = _nearest_half(float(np.median(fo_draws)))

        return results


# ---------------------------------------------------------------------------
# Class 6: Calibrator
# ---------------------------------------------------------------------------

class Calibrator:
    def __init__(self):
        self._model: Optional[LogisticRegression] = None
        self._fitted = False

    def fit(self, raw_rows: List[Dict]) -> None:
        """Fit on list of {pred_prob, actual} dicts — raw game predictions."""
        if len(raw_rows) < 15:
            logger.warning("Calibrator: only %d samples, skipping", len(raw_rows))
            return
        preds = np.array([r["pred_prob"] for r in raw_rows]).reshape(-1, 1)
        actuals = np.array([r["actual"] for r in raw_rows])
        self._model = LogisticRegression(C=1e6)
        self._model.fit(preds, actuals)
        self._fitted = True
        logger.info("Calibrator fitted on %d samples", len(raw_rows))

    def calibrate(self, p: float) -> float:
        if not self._fitted or self._model is None:
            return p
        try:
            return float(min(max(self._model.predict_proba([[p]])[0, 1], 0.01), 0.99))
        except Exception:
            return p


# ---------------------------------------------------------------------------
# Class 7: PricingEngine
# ---------------------------------------------------------------------------

class PricingEngine:
    def __init__(self, hold_pct: float = 0.045):
        self.hold_pct = hold_pct

    def price_game(self, gs: GameSimulation, cal: Optional[Calibrator] = None) -> GameMarket:
        h = gs.home_win_prob
        a = gs.away_win_prob
        if cal is not None and cal._fitted:
            h = cal.calibrate(h)
            a = 1.0 - h
        # tie_prob is always 0 (OT resolves ties), but guard defensively
        tie = gs.tie_prob
        if tie > 1e-6:
            split = h / max(h + a, 1e-9)
            h += tie * split
            a = 1.0 - h
        h_adj, a_adj = self._hold(max(h, 1e-4), max(a, 1e-4))

        # Market lines should be x.5 only so integer PLL scoring outcomes cannot push.
        # Model projections can still be any decimal; only priced/displayed lines are snapped.
        total_line = self._opt_line(gs.total_distribution, allow_negative=False)
        ov_p = float(np.mean(gs.total_distribution > total_line))
        ov_adj, un_adj = self._hold(max(ov_p, 1e-4), max(1.0 - ov_p, 1e-4))

        spread_line = self._opt_line(gs.margin_distribution, allow_negative=True)
        h_cover = float(np.mean(gs.margin_distribution > spread_line))
        h_spd_adj, a_spd_adj = self._hold(max(h_cover, 1e-4), max(1.0 - h_cover, 1e-4))

        return GameMarket(
            home_ml=self._am(h_adj), away_ml=self._am(a_adj),
            home_win_prob=round(h, 4), away_win_prob=round(a, 4),
            spread_home=round(float(spread_line), 1),
            spread_home_odds=self._am(h_spd_adj), spread_away_odds=self._am(a_spd_adj),
            total_line=round(float(total_line), 1),
            over_odds=self._am(ov_adj), under_odds=self._am(un_adj),
        )

    def price_prop(self, ps: PlayerSimulation, stat: str, line: Optional[float] = None) -> MarketLine:
        if stat not in ps.stat_distributions:
            fb = ps.prop_lines.get(stat, 0.5)
            return MarketLine(stat, fb, 0.50, 0.50, "-110", "-110", round(self.hold_pct, 4))
        dist = ps.stat_distributions[stat]
        if line is None:
            line = self._opt_line(dist)
        else:
            line = self._force_half_only(line)
        ov = float(np.mean(dist > line))
        ov_adj, un_adj = self._hold(max(ov, 1e-4), max(1.0 - ov, 1e-4))
        return MarketLine(
            stat=stat, line=line,
            fair_over_prob=round(ov, 4), fair_under_prob=round(1.0 - ov, 4),
            over_odds=self._am(ov_adj), under_odds=self._am(un_adj),
            juice=round((ov_adj + un_adj) - 1.0, 4),
        )

    def price_milestones(self, ps: PlayerSimulation, stat: str,
                         milestones: List[float]) -> List[MarketLine]:
        out = []
        for m in milestones:
            ml = self.price_prop(ps, stat, line=m - 0.5)
            ml.stat = f"{stat}_{int(m)}+"
            out.append(ml)
        return out

    def _hold(self, p1: float, p2: float) -> Tuple[float, float]:
        total = p1 + p2
        if total <= 0:
            h = self.hold_pct / 2
            return 0.50 + h, 0.50 + h
        t = 1.0 + self.hold_pct
        return (p1 / total) * t, (p2 / total) * t

    def _am(self, prob: float) -> str:
        prob = min(max(prob, 0.001), 0.999)
        if prob >= 0.50:
            return str(int(-round((prob / (1.0 - prob)) * 100)))
        return "+" + str(int(round(((1.0 - prob) / prob) * 100)))

    def _opt_line(self, dist: np.ndarray, allow_negative: bool = False) -> float:
        """Return the most balanced no-push line, restricted to x.5 values.

        PLL totals, spreads, team totals, and player props are integer-scored
        outcomes, so a whole-number line can push.  Sportsbook-style display
        should therefore be 0.5, 1.5, 2.5, etc.  Spreads can be negative,
        but they still use .5 increments only.
        """
        if len(dist) == 0:
            return 0.5
        arr = np.asarray(dist, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.5

        lo = float(np.percentile(arr, 1)) - (3.0 if allow_negative else 0.0)
        hi = float(np.percentile(arr, 99)) + 5.0
        if not allow_negative:
            lo = max(0.5, lo)

        cands = self._half_only_candidates(lo, hi)
        if cands.size == 0:
            return 0.5

        best, best_d = float(cands[0]), float("inf")
        for c in cands:
            d = abs(float(np.mean(arr > c)) - 0.50)
            if d < best_d:
                best_d, best = d, float(c)
        return round(float(best), 1)

    @staticmethod
    def _half_only_candidates(lo: float, hi: float) -> np.ndarray:
        """Candidates whose decimal part is always .5; excludes whole numbers."""
        if not np.isfinite(lo) or not np.isfinite(hi):
            return np.array([0.5])
        if hi < lo:
            lo, hi = hi, lo
        start = np.floor(lo) + 0.5
        if start < lo - 1e-9:
            start += 1.0
        end = np.ceil(hi) + 0.5
        return np.round(np.arange(start, end + 1e-9, 1.0), 1)

    def price_distribution(self, stat: str, dist: np.ndarray, line: Optional[float] = None,
                           allow_negative: bool = False) -> MarketLine:
        """Generic pricing helper for totals/spreads/team totals outside PlayerSimulation."""
        if line is None:
            line = self._opt_line(dist, allow_negative=allow_negative)
        else:
            line = self._force_half_only(line)
        arr = np.asarray(dist, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return MarketLine(stat, float(line), 0.50, 0.50, "-110", "-110", round(self.hold_pct, 4))
        ov = float(np.mean(arr > line))
        ov_adj, un_adj = self._hold(max(ov, 1e-4), max(1.0 - ov, 1e-4))
        return MarketLine(
            stat=stat, line=round(float(line), 1),
            fair_over_prob=round(ov, 4), fair_under_prob=round(1.0 - ov, 4),
            over_odds=self._am(ov_adj), under_odds=self._am(un_adj),
            juice=round((ov_adj + un_adj) - 1.0, 4),
        )

    @staticmethod
    def _force_half_only(line: float) -> float:
        """Snap an arbitrary line to the nearest x.5 value, never x.0."""
        try:
            v = float(line)
        except Exception:
            return 0.5
        if not np.isfinite(v):
            return 0.5
        lower = np.floor(v) + 0.5
        upper = lower + (1.0 if lower < v else -1.0)
        candidates = [lower, upper]
        best = min(candidates, key=lambda c: abs(c - v))
        return round(float(best), 1)


# ---------------------------------------------------------------------------
# Class 8: Backtester
# ---------------------------------------------------------------------------

class Backtester:
    def __init__(self, loader: DataLoader, n_sims: int = 5_000):
        self.loader = loader
        self.n_sims = n_sims
        self.raw_rows: List[Dict] = []

    def run(self, val_seasons: Optional[List[int]] = None) -> BacktestResult:
        tg = self.loader.load_team_games()
        current_season = int(tg["season"].max())
        if val_seasons is None:
            val_seasons = [s for s in sorted(tg["season"].unique()) if s >= 2023]

        rows = []
        self.raw_rows = []

        for _, grow in tg[tg["season"].isin(val_seasons)].drop_duplicates("game_id").iterrows():
            gid = grow["game_id"]
            gsea = int(grow["season"])
            gnum = int(grow["game_number"])

            train = tg[
                (tg["season"] < gsea) |
                ((tg["season"] == gsea) & (tg["game_number"] < gnum))
            ].copy()
            if len(train) < 30:
                continue

            game_rows = tg[tg["game_id"] == gid]
            if len(game_rows) != 2:
                continue

            home_r = game_rows[game_rows["is_home"] == 1]
            away_r = game_rows[game_rows["is_home"] == 0]
            if home_r.empty or away_r.empty:
                home_r = game_rows.iloc[[0]]
                away_r = game_rows.iloc[[1]]

            home_team = str(home_r.iloc[0]["team_id"])
            away_team = str(away_r.iloc[0]["team_id"])
            act_hg = float(home_r.iloc[0]["goals"])
            act_ag = float(away_r.iloc[0]["goals"])
            act_hs = float(home_r.iloc[0]["scores"])
            act_as = float(away_r.iloc[0]["scores"])
            act_home_win = 1 if act_hs > act_as else 0

            try:
                rb = RatingBuilder(train, pd.DataFrame())
                rb.build_team_ratings()
                tm = TeamModel()
                tm.fit(rb._tr)

                hf = rb.get_team_rating(home_team)
                af = rb.get_team_rating(away_team)
                if not hf or not af:
                    continue

                hp = tm.predict(hf, af)
                ap = tm.predict(af, hf)
                _assign_goalie_saves_team(hp, ap)
                _assign_goalie_saves_team(ap, hp)

                sim = GameSimulator(n_sims=self.n_sims, seed=42)
                gs = sim.simulate_game(hp, ap)

                row = {
                    "game_id": gid, "season": gsea,
                    "home_team": home_team, "away_team": away_team,
                    "pred_h_goals": hp.proj_goals, "pred_a_goals": ap.proj_goals,
                    "pred_total": hp.proj_goals + ap.proj_goals,
                    "pred_h_score": hp.proj_scores, "pred_a_score": ap.proj_scores,
                    "pred_total_scores": hp.proj_scores + ap.proj_scores,
                    "act_h_goals": act_hg, "act_a_goals": act_ag,
                    "act_total": act_hg + act_ag,
                    "act_h_score": act_hs, "act_a_score": act_as,
                    "act_total_scores": act_hs + act_as,
                    "pred_prob": gs.home_win_prob,
                    "actual": act_home_win,
                }
                rows.append(row)
                self.raw_rows.append({"pred_prob": gs.home_win_prob, "actual": act_home_win})
            except Exception as exc:
                logger.debug("Backtest %s failed: %s", gid, exc)

        if not rows:
            logger.warning("No backtest rows collected")
            return BacktestResult(0, 999, 999, 999, 999, 999, 0.0, 999, 999,
                                  pd.DataFrame(), [])

        df = pd.DataFrame(rows)
        mae_h = float(np.mean(np.abs(df["pred_h_goals"] - df["act_h_goals"])))
        mae_a = float(np.mean(np.abs(df["pred_a_goals"] - df["act_a_goals"])))
        mae_t = float(np.mean(np.abs(df["pred_total"] - df["act_total"])))
        rmse_t = float(np.sqrt(np.mean((df["pred_total"] - df["act_total"]) ** 2)))
        bias_t = float(np.mean(df["pred_total"] - df["act_total"]))
        mae_ts = float(np.mean(np.abs(df["pred_total_scores"] - df["act_total_scores"])))
        brier = float(np.mean((df["pred_prob"] - df["actual"]) ** 2))
        correct = float(np.mean((df["pred_prob"] > 0.5) == df["actual"].astype(bool)))

        df["prob_bucket"] = pd.cut(df["pred_prob"], bins=np.arange(0, 1.1, 0.1), right=False)
        cal_table = (
            df.groupby("prob_bucket", observed=True)
            .agg(n=("pred_prob", "count"),
                 mean_pred=("pred_prob", "mean"),
                 actual_rate=("actual", "mean"))
            .reset_index()
        )

        logger.info("Backtest: n=%d, MAE_total=%.2f, bias=%.2f, Brier=%.4f, correct=%.1f%%",
                    len(df), mae_t, bias_t, brier, correct * 100)

        return BacktestResult(
            n_games=len(df),
            mae_home_goals=round(mae_h, 3),
            mae_away_goals=round(mae_a, 3),
            mae_total_goals=round(mae_t, 3),
            rmse_total_goals=round(rmse_t, 3),
            brier_score=round(brier, 4),
            correct_winner_pct=round(correct, 4),
            mae_total_scores=round(mae_ts, 3),
            bias_total_goals=round(bias_t, 3),
            calibration_table=cal_table,
            raw_rows=self.raw_rows,
        )


# ---------------------------------------------------------------------------
# Class 9: ProjectionEngine
# ---------------------------------------------------------------------------

class ProjectionEngine:
    """Main orchestrator."""

    def __init__(self, db_path: Optional[str] = None, hold_pct: float = 0.045):
        self.loader = DataLoader(db_path=db_path)
        self.team_games: pd.DataFrame = pd.DataFrame()
        self.player_games: pd.DataFrame = pd.DataFrame()
        self.schedule: pd.DataFrame = pd.DataFrame()
        self.current_rosters: pd.DataFrame = pd.DataFrame()
        self.current_rosters_status: Dict[str, object] = {}
        self.current_roster_filter: Optional[CurrentRosterFilter] = None
        self.rating_builder: Optional[RatingBuilder] = None
        self.team_model: Optional[TeamModel] = None
        self.player_model: Optional[PlayerModel] = None
        self.simulator = GameSimulator(n_sims=N_SIMS, seed=42)
        self.calibrator = Calibrator()
        self.pricing = PricingEngine(hold_pct=hold_pct)
        self._loaded = False
        self._fitted = False

    def load(self) -> None:
        logger.info("Loading warehouse data...")
        self.team_games = self.loader.load_team_games()
        self.player_games = self.loader.load_player_games()
        self.schedule = self.loader.load_schedule(include_completed=False)
        self.current_rosters, self.current_rosters_status = self.loader.load_current_rosters()
        self._loaded = True
        logger.info("Loaded: %d team-game rows | %d player-game rows | %d upcoming games",
                    len(self.team_games), len(self.player_games), len(self.schedule))
        logger.info("Current roster cache: %s", self.current_rosters_status)

    def fit(self, run_backtest: bool = False) -> Optional[BacktestResult]:
        if not self._loaded:
            self.load()
        logger.info("Building ratings...")
        self.rating_builder = RatingBuilder(self.team_games, self.player_games)
        self.rating_builder.build_team_ratings()
        self.rating_builder.build_player_ratings()

        logger.info("Fitting team model...")
        tr = self.rating_builder._tr
        self.team_model = TeamModel()
        self.team_model.fit(tr if tr is not None else pd.DataFrame())

        current_team_map = self.loader.resolve_current_team(self.player_games)
        self.current_roster_filter = CurrentRosterFilter(self.current_rosters)
        # Preserve CSV load metadata while adding validation/match status.
        self.current_rosters_status = {
            **(self.current_rosters_status or {}),
            **self.current_roster_filter.status,
        }
        pr = self.rating_builder._pr
        self.player_model = PlayerModel(
            pr if pr is not None else pd.DataFrame(),
            current_team_map=current_team_map,
            current_roster_filter=self.current_roster_filter,
        )
        self._fitted = True
        logger.info("Models fitted.")

        bt_result = None
        if run_backtest:
            logger.info("Running backtest for calibration...")
            bt = Backtester(self.loader, n_sims=5_000)
            bt_result = bt.run()
            if bt.raw_rows:
                self.calibrator.fit(bt.raw_rows)
        return bt_result

    @staticmethod
    def _should_use_current_rosters(game_date: Optional[str]) -> bool:
        """Use official current roster cache only for current/future game projections."""
        if not game_date:
            return True
        try:
            gd = pd.to_datetime(game_date, utc=True, errors="coerce")
            if pd.isna(gd):
                return True
            return gd.date() >= date.today()
        except Exception:
            return True

    def project(
        self,
        home_team_id: Optional[str] = None,
        away_team_id: Optional[str] = None,
        game_date: Optional[str] = None,
        player_overrides: Optional[Dict] = None,
        active_players: Optional[Dict] = None,
        starter_goalies: Optional[Dict[str, str]] = None,
        team_adjustments: Optional[Dict[str, Dict]] = None,
        team_rating_overrides: Optional[Dict[str, Dict]] = None,
    ) -> ProjectionResult:
        """
        Parameters
        ----------
        starter_goalies       : {team_id: player_id}
        team_adjustments      : {team_id: {"off_mult": float, "def_mult_opp": float}}
        team_rating_overrides : {team_id: {rating_key: value}} — injects values directly
                                into the team rating dict before prediction, e.g.
                                {"ATL": {"goals_ewm": 13.5, "bayes_fo_pct": 0.62}}
                                Supports any key that RatingBuilder puts in the rating dict.
        """
        if not self._fitted:
            self.fit()

        overrides: Dict = dict(player_overrides or {})
        for pid, is_active in (active_players or {}).items():
            overrides.setdefault(pid, {})["active"] = is_active

        if not home_team_id or not away_team_id:
            upcoming = self.upcoming_games()
            if not upcoming:
                raise ValueError("No upcoming games and no team IDs provided.")
            g = upcoming[0]
            home_team_id = home_team_id or g["home_team_id"]
            away_team_id = away_team_id or g["away_team_id"]

        logger.info("Projecting: %s vs %s", home_team_id, away_team_id)

        assert self.rating_builder is not None
        assert self.team_model is not None
        assert self.player_model is not None

        hf = self.rating_builder.get_team_rating(home_team_id, game_date)
        af = self.rating_builder.get_team_rating(away_team_id, game_date)

        # Apply direct rating overrides (user-adjusted values from UI)
        tro = team_rating_overrides or {}
        if home_team_id in tro:
            hf = {**hf, **tro[home_team_id]}
        if away_team_id in tro:
            af = {**af, **tro[away_team_id]}

        adj_map = team_adjustments or {}
        h_proj = self.team_model.predict(hf, af, adj_map.get(home_team_id))
        a_proj = self.team_model.predict(af, hf, adj_map.get(away_team_id))

        # Assign goalie stats using opponent's projected SOG
        _assign_goalie_saves_team(h_proj, a_proj)
        _assign_goalie_saves_team(a_proj, h_proj)

        starters = starter_goalies or {}
        use_current_rosters = self._should_use_current_rosters(game_date)

        def _team_ov(tid: str) -> Dict:
            pr = self.rating_builder._pr
            if pr is None or pr.empty:
                return overrides
            team_pids = set(pr[pr["team_id"] == tid]["player_id"].astype(str).tolist())
            return {pid: v for pid, v in overrides.items() if pid in team_pids}

        h_players = self.player_model.project_roster(
            home_team_id, h_proj, _team_ov(home_team_id),
            starter_goalie=starters.get(home_team_id),
            use_current_roster_filter=use_current_rosters,
        )
        a_players = self.player_model.project_roster(
            away_team_id, a_proj, _team_ov(away_team_id),
            starter_goalie=starters.get(away_team_id),
            use_current_roster_filter=use_current_rosters,
        )

        # Assign goalie saves using correct opponent SOG
        _assign_player_goalie_saves(h_players, a_proj.proj_sog)
        _assign_player_goalie_saves(a_players, h_proj.proj_sog)

        game_sim = self.simulator.simulate_game(h_proj, a_proj)
        h_psims = self.simulator.simulate_players(h_players, game_sim.home_goals, h_proj.proj_goals)
        a_psims = self.simulator.simulate_players(a_players, game_sim.away_goals, a_proj.proj_goals)

        game_market = self.pricing.price_game(
            game_sim, self.calibrator if self.calibrator._fitted else None
        )
        player_markets = _price_players(h_psims + a_psims, self.pricing)

        game_id = f"{home_team_id}_vs_{away_team_id}_{game_date or 'latest'}".replace(" ", "_")
        return ProjectionResult(
            game_id=game_id,
            home_team=str(hf.get("team_name", home_team_id)),
            away_team=str(af.get("team_name", away_team_id)),
            home_proj=h_proj, away_proj=a_proj,
            home_players=h_players, away_players=a_players,
            game_sim=game_sim,
            home_player_sims=h_psims, away_player_sims=a_psims,
            game_market=game_market, player_markets=player_markets,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def upcoming_games(self) -> List[Dict]:
        if self.schedule.empty:
            return []
        return [
            {
                "home_team_id": str(r.get("home_team_id", "")),
                "away_team_id": str(r.get("away_team_id", "")),
                "home_team_name": str(r.get("home_team_name", "")),
                "away_team_name": str(r.get("away_team_name", "")),
                "game_date": str(r.get("game_date_guess", "")),
                "game_number": str(r.get("game_number", "")),
            }
            for _, r in self.schedule.iterrows()
        ]

    def export(self, result: ProjectionResult, path: Optional[str] = None) -> str:
        if path is None:
            path = str(Path(__file__).parent / f"projection_{result.game_id}.json")

        def _pcts(arr: np.ndarray) -> Dict:
            return {k: float(v) for k, v in zip(
                ["p10", "p25", "p50", "p75", "p90", "mean"],
                [np.percentile(arr, p) for p in [10, 25, 50, 75, 90]] + [np.mean(arr)],
            )}

        def _default(obj):
            if isinstance(obj, np.ndarray):
                return _pcts(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            return str(obj)

        data = {
            "game_id": result.game_id,
            "home_team": result.home_team, "away_team": result.away_team,
            "generated_at": result.generated_at,
            "home_proj": asdict(result.home_proj), "away_proj": asdict(result.away_proj),
            "game_simulation": {
                "n_sims": result.game_sim.n_sims,
                "home_win_prob": result.game_sim.home_win_prob,
                "away_win_prob": result.game_sim.away_win_prob,
                "tie_prob": result.game_sim.tie_prob,
                "expected_total": result.game_sim.expected_total,
                "spread_home": result.game_sim.spread_home,
                "home_goals_dist": _pcts(result.game_sim.home_goals),
                "away_goals_dist": _pcts(result.game_sim.away_goals),
                "total_dist": _pcts(result.game_sim.total_distribution),
            },
            "game_market": asdict(result.game_market),
            "home_players": [asdict(p) for p in result.home_players],
            "away_players": [asdict(p) for p in result.away_players],
            "player_markets": result.player_markets,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=_default)
        logger.info("Exported: %s", path)
        return path

    def print_result(self, result: ProjectionResult) -> None:
        gs = result.game_sim
        gm = result.game_market
        hp, ap = result.home_proj, result.away_proj

        print("=" * 72)
        print(f"  {result.home_team} (home)  vs  {result.away_team} (away)")
        print("=" * 72)
        print(f"\n  {'TEAM PROJECTIONS':=<52}")
        print(f"  {'':14} {'Goals':>6} {'Score':>6} {'Shots':>7} {'SOG':>6} {'FO%':>6} {'2PT':>5} {'SavePct':>8}")
        print(f"  {'-'*62}")
        for nm, proj in [(result.home_team, hp), (result.away_team, ap)]:
            print(f"  {nm:<14} {proj.proj_goals:>6.1f} {proj.proj_scores:>6.1f} "
                  f"{proj.proj_shots:>7.1f} {proj.proj_sog:>6.1f} "
                  f"{proj.proj_faceoff_pct:>6.3f} {proj.proj_2pt_goals:>5.1f} "
                  f"{proj.proj_save_pct:>8.3f}")
        print(f"\n  {'SIMULATION':=<52}")
        print(f"  Home win: {gs.home_win_prob:.1%}  |  Away: {gs.away_win_prob:.1%}  |  Tie: {gs.tie_prob:.1%}")
        print(f"  Spread (home): {gs.spread_home:+.1f}  |  Expected total: {gs.expected_total:.1f}")
        print(f"\n  {'MARKET':=<52}")
        print(f"  ML  {result.home_team} {gm.home_ml}  /  {result.away_team} {gm.away_ml}")
        print(f"  Spread ({gs.spread_home:+.1f}) {gm.spread_home_odds}/{gm.spread_away_odds}  "
              f"|  Total {gm.total_line} O{gm.over_odds}/U{gm.under_odds}")

        for team_name, players in [(result.home_team, result.home_players),
                                    (result.away_team, result.away_players)]:
            active = [p for p in players if p.active]
            if not active:
                continue
            print(f"\n  {'PLAYERS -- ' + team_name:=<52}")
            print(f"  {'Name':<26} {'Pos':<5} {'G':>5} {'A':>5} {'Pts':>6} {'Sh':>5} {'SOG':>5} {'2PT':>5}")
            for pp in sorted(active, key=lambda x: x.proj_points, reverse=True)[:8]:
                print(f"  {(pp.full_name or pp.player_id):<26} {pp.position:<5} "
                      f"{pp.proj_goals:>5.2f} {pp.proj_assists:>5.2f} {pp.proj_points:>6.2f} "
                      f"{pp.proj_shots:>5.2f} {pp.proj_sog:>5.2f} {pp.proj_2pt_goals:>5.2f}")

        meaningful = sorted(
            [(pid, pm) for pid, pm in result.player_markets.items()
             if pm.get("proj_values", {}).get("points", 0) >= 0.5
             or pm.get("proj_values", {}).get("saves", 0) >= 3.0],
            key=lambda x: x[1].get("proj_values", {}).get("points", 0), reverse=True
        )[:6]
        if meaningful:
            print(f"\n  {'PROP MARKETS (sample)':=<52}")
            for pid, pm in meaningful:
                nm = pm.get("full_name", pid)
                for stat in ["goals", "assists", "points"]:
                    mkt = pm.get("markets", {}).get(stat)
                    if mkt:
                        print(f"  {nm:<26}  {stat} {mkt['line']}  "
                              f"O{mkt['over_odds']}/U{mkt['under_odds']}")
        print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assign_goalie_saves_team(team_proj: TeamProjection, opp_proj: TeamProjection) -> None:
    """Set team-level save projection using opponent's projected SOG.
    Uses team's Bayesian save_pct; falls back to league average if missing.
    """
    opp_sog = max(float(opp_proj.proj_sog), 1.0)
    sv_pct = float(team_proj.proj_save_pct) if team_proj.proj_save_pct > 0.20 else LG_SAVE_PCT
    sv_pct = min(max(sv_pct, 0.35), 0.72)
    team_proj.proj_shots_faced = opp_sog
    team_proj.proj_saves = opp_sog * sv_pct
    team_proj.proj_save_pct = sv_pct


def _assign_player_goalie_saves(
    player_projs: List[PlayerProjection],
    opp_sog: float,
    starter_id: Optional[str] = None,
) -> None:
    """
    Assign goalie saves using correct denominator: opponent's SOG.

    Starter selection priority:
    1. Explicitly designated starter_id
    2. Goalie with most career games (highest confidence rating)
    3. Any active goalie on the roster
    Falls back to any goalie (active or not) if all are inactive — ensures
    we never silently produce 0.0 saves for a team that has a goalie.
    """
    goalies = [p for p in player_projs if p.position == "G"]
    active_goalies = [g for g in goalies if g.active]
    pool = active_goalies if active_goalies else goalies
    if not pool:
        return

    # Pick starter
    if starter_id:
        named = [g for g in pool if g.player_id == starter_id]
        starter = named[0] if named else max(pool, key=lambda p: p.proj_save_pct)
    else:
        # Highest save_pct = best goalie = most likely starter.
        # confidence alone is misleading when a career-starter has 0 current-season games.
        starter = max(pool, key=lambda p: p.proj_save_pct)

    # Save% must be based on saves/(saves+goals_against); already correct in bayes_save_pct
    sv_pct = max(starter.proj_save_pct, 0.35)
    # Clamp to observed range: min ~0.46 (2022 WAT), max ~0.59 (2025 ARC)
    sv_pct = min(sv_pct, 0.72)
    starter.proj_saves = opp_sog * sv_pct
    starter.is_starter = True

    for g in pool:
        if g.player_id != starter.player_id:
            g.proj_saves = 0.0
            g.is_starter = False


def _price_players(sims: List[PlayerSimulation], pricing: PricingEngine) -> Dict[str, Dict]:
    markets: Dict[str, Dict] = {}
    for ps in sims:
        total_proj = sum(ps.proj_values.get(s, 0.0) for s in ["goals", "assists", "saves", "faceoff_wins"])
        if total_proj < 0.10:
            continue
        lines: Dict[str, Dict] = {}
        for stat in ["goals", "assists", "points", "shots", "shots_on_goal",
                     "two_pt_goals", "saves", "faceoff_wins"]:
            if stat in ps.stat_distributions:
                lines[stat] = asdict(pricing.price_prop(ps, stat))
        if "goals" in ps.stat_distributions:
            for ml in pricing.price_milestones(ps, "goals", [1.0, 2.0, 3.0]):
                lines[ml.stat] = asdict(ml)
        if "assists" in ps.stat_distributions:
            for ml in pricing.price_milestones(ps, "assists", [1.0, 2.0]):
                lines[ml.stat] = asdict(ml)
        markets[ps.player_id] = {
            "player_id": ps.player_id, "full_name": ps.full_name,
            "proj_values": ps.proj_values, "prop_lines": ps.prop_lines,
            "markets": lines,
        }
    return markets


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run_game(
    home_team_id: str,
    away_team_id: str,
    db_path: Optional[str] = None,
    game_date: Optional[str] = None,
    player_overrides: Optional[Dict] = None,
    active_players: Optional[Dict] = None,
    starter_goalies: Optional[Dict[str, str]] = None,
    team_adjustments: Optional[Dict[str, Dict]] = None,
    hold_pct: float = 0.045,
    print_output: bool = True,
    export: bool = False,
) -> ProjectionResult:
    engine = ProjectionEngine(db_path=db_path, hold_pct=hold_pct)
    engine.load()
    engine.fit(run_backtest=False)
    result = engine.project(
        home_team_id=home_team_id, away_team_id=away_team_id,
        game_date=game_date, player_overrides=player_overrides,
        active_players=active_players, starter_goalies=starter_goalies,
        team_adjustments=team_adjustments,
    )
    if print_output:
        engine.print_result(result)
    if export:
        engine.export(result)
    return result


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    db_arg = sys.argv[1] if len(sys.argv) > 1 else None
    print("PLL Projection Engine v3 -- Smoke Test")
    print("=" * 72)
    engine = ProjectionEngine(db_path=db_arg)
    print("[1/3] Loading...")
    engine.load()
    print(f"  Team-game rows:   {len(engine.team_games):,}")
    print(f"  Player-game rows: {len(engine.player_games):,}")
    upcoming = engine.upcoming_games()
    print(f"  Upcoming games:   {len(upcoming)}")
    print("[2/3] Fitting...")
    engine.fit(run_backtest=False)
    print("[3/3] Projecting...")
    if upcoming:
        g = upcoming[0]
        hid, aid = g["home_team_id"], g["away_team_id"]
    else:
        teams = engine.team_games["team_id"].dropna().unique().tolist()
        hid, aid = teams[0], teams[1]
    result = engine.project(hid, aid)
    engine.print_result(result)
    print("Done.")
