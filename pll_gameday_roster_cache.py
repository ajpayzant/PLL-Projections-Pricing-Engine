"""
PLL Gameday Roster Cache
========================
Python port of the Google Apps Script gameday roster tracker.

Hits: https://api.stats.premierlacrosseleague.com/api/v4/events/gameday-rosters
Saves: data/reference_tables/gameday_rosters/gameday_{year}_week{week:02d}.csv
       data/reference_tables/gameday_rosters/gameday_latest.csv  (always current week)

Usage:
    python pll_gameday_roster_cache.py               # auto-detect current week
    python pll_gameday_roster_cache.py --year 2026 --week 3
    python pll_gameday_roster_cache.py --all-weeks   # scrape all weeks so far

Auth:
    Set PLL_BEARER_TOKEN env var (same token used in build_warehouse.py / GitHub Actions).

Output columns (mapped from API response):
    player_id, player_name, first_name, last_name,
    team_id, team_name, team_code,
    position, position_group,
    jersey_number, status, is_active,
    game_id, game_slug, game_number,
    home_team_id, away_team_id,
    year, week,
    scraped_at

The app uses this file via GamedayRosterFilter in projection_engine_v3.py.
Priority: gameday roster > current_rosters.csv > historical fallback.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# ── Paths ─────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent
GAMEDAY_DIR  = REPO_ROOT / "data" / "reference_tables" / "gameday_rosters"
LATEST_PATH  = GAMEDAY_DIR / "gameday_latest.csv"

# ── API ───────────────────────────────────────────────────────────────────
BASE_URL     = "https://api.stats.premierlacrosseleague.com/api/v4/events/gameday-rosters"
CURRENT_YEAR = 2026
FIRST_WEEK   = 1
LAST_WEEK    = 14

# ── Team ID mapping (public API codes → engine warehouse codes) ───────────
PUBLIC_TO_ENGINE: Dict[str, str] = {
    "BOS": "CAN", "CAL": "RED", "CAR": "CHA", "DEN": "OUT",
    "MD":  "WHP", "NY":  "ATL", "PHI": "WAT", "UTA": "ARC",
    # Engine IDs may also appear directly
    "CAN": "CAN", "RED": "RED", "CHA": "CHA", "OUT": "OUT",
    "WHP": "WHP", "ATL": "ATL", "WAT": "WAT", "ARC": "ARC",
}

ENGINE_TEAM_NAMES: Dict[str, str] = {
    "CAN": "Boston Cannons",   "RED": "California Redwoods",
    "CHA": "Carolina Chaos",   "OUT": "Denver Outlaws",
    "WHP": "Maryland Whipsnakes", "ATL": "New York Atlas",
    "WAT": "Philadelphia Waterdogs", "ARC": "Utah Archers",
}

# ── Position normalisation ────────────────────────────────────────────────
_POS_ALIASES = {
    "ATTACK": "A", "ATT": "A",
    "MIDFIELD": "M", "MID": "M",
    "DEFENSE": "D", "DEF": "D",
    "FACEOFF": "FO", "FACE-OFF": "FO", "FO/MF": "FO",
    "GOALIE": "G", "GOALTENDER": "G", "GK": "G",
    "LONG STICK MIDFIELD": "LSM", "LSM": "LSM",
    "SHORT STICK DEFENSIVE MIDFIELD": "SSDM", "SSDM": "SSDM", "SS": "SSDM",
}

def _norm_pos(raw: str) -> str:
    if not raw:
        return "M"
    r = str(raw).strip().upper()
    return _POS_ALIASES.get(r, r if r in {"A","M","D","FO","G","LSM","SSDM"} else "M")

_POS_GROUP = {
    "A": "Attack", "M": "Midfield", "D": "Defense",
    "FO": "Faceoff", "G": "Goalie", "LSM": "Long Stick Midfield",
    "SSDM": "Short Stick Defensive Midfield",
}

# ── Output columns (in order) ────────────────────────────────────────────
GAMEDAY_COLUMNS = [
    "player_id", "player_name", "first_name", "last_name",
    "team_id", "team_name", "team_code",
    "position", "position_group",
    "jersey_number", "status", "is_active",
    "game_id", "game_slug", "game_number",
    "home_team_id", "away_team_id",
    "year", "week",
    "scraped_at",
]


# ── Auth ──────────────────────────────────────────────────────────────────
def _get_token() -> str:
    token = os.environ.get("PLL_BEARER_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "PLL_BEARER_TOKEN environment variable is not set.\n"
            "Set it before running:\n"
            "  Windows: set PLL_BEARER_TOKEN=your_token_here\n"
            "  bash:    export PLL_BEARER_TOKEN=your_token_here\n"
            "The token is stored in your GitHub Actions secrets."
        )
    return token


def _headers(token: str) -> Dict[str, str]:
    return {
        "accept":        "application/json",
        "content-type":  "application/json",
        "origin":        "https://premierlacrosseleague.com",
        "referer":       "https://premierlacrosseleague.com/",
        "authsource":    "web",
        "time-zone":     "America/Los_Angeles",
        "Authorization": f"Bearer {token}",
    }


# ── API fetch ─────────────────────────────────────────────────────────────
def fetch_gameday_rosters(year: int, week: int, token: str) -> Dict[str, Any]:
    """
    Hit the PLL gameday-rosters endpoint and return raw JSON.

    On first run this prints the top-level keys so you can verify the
    response structure. The parser below handles the most common shapes;
    if the API changes the field names, update _parse_player() accordingly.
    """
    url    = BASE_URL
    params = {"year": year, "week": week}
    resp   = requests.get(url, params=params, headers=_headers(token), timeout=30)

    if resp.status_code == 401:
        raise RuntimeError("API returned 401 Unauthorized. Check your PLL_BEARER_TOKEN.")
    if resp.status_code == 404:
        return {}   # week not yet available
    resp.raise_for_status()

    data = resp.json()
    print(f"  API keys: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}")
    return data


# ── Parser ────────────────────────────────────────────────────────────────
def _clean(v: Any) -> str:
    if v is None:
        return ""
    return re.sub(r"\s+", " ", str(v).replace("\xa0", " ")).strip()


def _team_engine_id(raw_id: str, raw_name: str = "") -> str:
    """Map any team identifier to an engine warehouse ID."""
    rid = str(raw_id).strip().upper()
    if rid in PUBLIC_TO_ENGINE:
        return PUBLIC_TO_ENGINE[rid]
    # Try matching by name fragment
    for eng, nm in ENGINE_TEAM_NAMES.items():
        if raw_name and raw_name.lower() in nm.lower():
            return eng
    return rid


def _parse_player(player: Dict, game_meta: Dict, year: int, week: int,
                  scraped_at: str) -> Optional[Dict]:
    """
    Extract one player row from an API player object.

    The PLL API uses camelCase. Known field variants handled here:
      playerName / fullName / name
      firstName / first_name
      lastName / last_name
      teamId / team_id / teamAbbreviation
      position / positionAbbreviation / posAbbr
      jerseyNumber / number / jersey
      rosterStatus / status / active / isActive
      playerId / id / officialId
    """
    def _get(*keys, default=""):
        for k in keys:
            v = player.get(k)
            if v is not None and str(v).strip() not in ("", "None", "null"):
                return v
        return default

    raw_name  = _clean(_get("playerName", "fullName", "name", "full_name"))
    first     = _clean(_get("firstName", "first_name", "givenName"))
    last      = _clean(_get("lastName", "last_name", "familyName"))

    if not raw_name and first:
        raw_name = f"{first} {last}".strip()
    if not raw_name:
        return None   # skip unnamed entries

    # Player ID — prefer officialId, fall back to id
    raw_pid   = _clean(_get("playerId", "id", "officialId", "player_id"))

    # Team
    raw_team  = _clean(_get("teamId", "team_id", "teamAbbreviation", "teamCode"))
    team_name_raw = _clean(_get("teamName", "team_name", "teamFullName"))
    team_id   = _team_engine_id(raw_team, team_name_raw)
    team_code = raw_team.upper() if raw_team else team_id

    # Position
    raw_pos   = _clean(_get("position", "positionAbbreviation", "posAbbr", "pos"))
    position  = _norm_pos(raw_pos)
    pos_group = _POS_GROUP.get(position, "Unknown")

    # Jersey
    jersey    = _clean(_get("jerseyNumber", "number", "jersey", "jerseyNum"))

    # Active / status
    # The API may return: rosterStatus string, active bool, isActive bool, status string
    status_raw = _clean(_get("rosterStatus", "status", "rosterStatusLabel"))
    is_active_raw = _get("active", "isActive", "is_active", default=None)

    if is_active_raw is not None:
        is_active = bool(is_active_raw)
    else:
        # Infer from status string: "Active" / "Inactive" / "Injured" / "Suspended"
        sl = status_raw.lower()
        is_active = sl in ("", "active", "1", "true", "yes") and "inactive" not in sl and "injur" not in sl

    status = status_raw if status_raw else ("Active" if is_active else "Inactive")

    return {
        "player_id":      raw_pid,
        "player_name":    raw_name,
        "first_name":     first,
        "last_name":      last,
        "team_id":        team_id,
        "team_name":      ENGINE_TEAM_NAMES.get(team_id, team_name_raw),
        "team_code":      team_code,
        "position":       position,
        "position_group": pos_group,
        "jersey_number":  jersey,
        "status":         status,
        "is_active":      is_active,
        "game_id":        game_meta.get("game_id", ""),
        "game_slug":      game_meta.get("slug", ""),
        "game_number":    game_meta.get("game_number", ""),
        "home_team_id":   game_meta.get("home_team_id", ""),
        "away_team_id":   game_meta.get("away_team_id", ""),
        "year":           year,
        "week":           week,
        "scraped_at":     scraped_at,
    }


def _extract_game_meta(game_obj: Dict) -> Dict:
    """Pull game-level fields from a game/event object in the response."""
    def _g(*keys, default=""):
        for k in keys:
            v = game_obj.get(k)
            if v is not None:
                return v
        return default

    raw_home = _clean(_g("homeTeamId", "home_team_id"))
    raw_away = _clean(_g("awayTeamId", "visitorTeamId", "away_team_id"))

    # homeTeam / awayTeam may be nested dicts (gamedayRoster shape and others)
    home_obj = _g("homeTeam")
    if isinstance(home_obj, dict):
        raw_home = _clean(
            home_obj.get("locationCode") or home_obj.get("abbreviation")
            or home_obj.get("teamId") or home_obj.get("officialId") or home_obj.get("id", "")
        )
    away_obj = _g("visitorTeam") or _g("awayTeam")
    if isinstance(away_obj, dict):
        raw_away = _clean(
            away_obj.get("locationCode") or away_obj.get("abbreviation")
            or away_obj.get("teamId") or away_obj.get("officialId") or away_obj.get("id", "")
        )

    return {
        "game_id":      _clean(_g("gameId", "eventId", "id", "game_id")),
        "slug":         _clean(_g("slug", "slugname", "eventSlug")),
        "game_number":  _clean(_g("gameNumber", "game_number", "week")),
        "home_team_id": _team_engine_id(raw_home),
        "away_team_id": _team_engine_id(raw_away),
    }


def _parse_gamedayRoster_player(player: Dict, team_obj: Dict, game_meta: Dict,
                                year: int, week: int, scraped_at: str) -> Optional[Dict]:
    """
    Parse a player from the gamedayRoster shape.

    The PLL /api/v4/events/gameday-rosters endpoint returns:
      data.items[i].homeTeam.gamedayRoster[j]  and
      data.items[i].awayTeam.gamedayRoster[j]

    Player fields: firstName, lastName, officialId, position, positionName,
                   rosterStatus, injuryStatus, jerseyNum, slug, age, etc.
    Team  fields:  locationCode, officialId, fullName, location
    """
    first = _clean(player.get("firstName", ""))
    last  = _clean(player.get("lastName", ""))
    raw_name = f"{first} {last}".strip()
    if not raw_name:
        return None

    raw_pid = _clean(player.get("officialId") or player.get("playerId") or player.get("id", ""))

    # Team identity from the parent team object
    raw_team = _clean(
        team_obj.get("locationCode") or team_obj.get("abbreviation")
        or team_obj.get("teamId") or team_obj.get("officialId", "")
    )
    team_name_raw = _clean(
        (team_obj.get("location") or "") + " " + (team_obj.get("fullName") or "")
    ).strip()
    team_id  = _team_engine_id(raw_team, team_name_raw)
    team_code = raw_team.upper() if raw_team else team_id

    # Position — API uses 'position' (abbreviation) and 'positionName' (full)
    raw_pos  = _clean(player.get("position", ""))
    pos_name = _clean(player.get("positionName", ""))
    if not raw_pos and pos_name:
        # Map positionName to abbreviation using the same logic as the JS
        pn = pos_name.lower()
        if "attack" in pn:
            raw_pos = "A"
        elif "defensive midfield" in pn or "ssdm" in pn:
            raw_pos = "SSDM"
        elif "long stick" in pn or "lsm" in pn:
            raw_pos = "LSM"
        elif "faceoff" in pn:
            raw_pos = "FO"
        elif "midfield" in pn:
            raw_pos = "M"
        elif "defense" in pn or "defence" in pn:
            raw_pos = "D"
        elif "goal" in pn:
            raw_pos = "G"
    position  = _norm_pos(raw_pos)
    pos_group = _POS_GROUP.get(position, "Unknown")

    jersey = _clean(player.get("jerseyNum") or player.get("jerseyNumber") or player.get("number", ""))

    # Roster status
    status_raw = _clean(player.get("rosterStatus", ""))
    is_active  = status_raw.lower() in ("", "active", "1", "true", "yes")
    status     = status_raw if status_raw else ("Active" if is_active else "Inactive")

    return {
        "player_id":      raw_pid,
        "player_name":    raw_name,
        "first_name":     first,
        "last_name":      last,
        "team_id":        team_id,
        "team_name":      ENGINE_TEAM_NAMES.get(team_id, team_name_raw),
        "team_code":      team_code,
        "position":       position,
        "position_group": pos_group,
        "jersey_number":  jersey,
        "status":         status,
        "is_active":      is_active,
        "game_id":        game_meta.get("game_id", ""),
        "game_slug":      game_meta.get("slug", ""),
        "game_number":    game_meta.get("game_number", ""),
        "home_team_id":   game_meta.get("home_team_id", ""),
        "away_team_id":   game_meta.get("away_team_id", ""),
        "year":           year,
        "week":           week,
        "scraped_at":     scraped_at,
    }


def parse_response(data: Dict, year: int, week: int) -> pd.DataFrame:
    """
    Parse the raw API response into a flat player DataFrame.

    Handles the PLL API shapes:
      Shape GD (primary):
        {"data": {"items": [{"eventId":..., "homeTeam": {"gamedayRoster": [...]},
                              "awayTeam": {"gamedayRoster": [...]}}]}}
      Shape A: {"games": [...]} or {"data": {"games": [...]}}
      Shape C: {"players": [...]}
      Shape D: top-level array
    """
    if not data:
        return pd.DataFrame(columns=GAMEDAY_COLUMNS)

    scraped_at = dt.datetime.now(dt.timezone.utc).isoformat()
    rows: List[Dict] = []

    # Unwrap outer envelope: {"data": {...}} or {"data": [...]}
    inner = data
    if isinstance(data, dict) and "data" in data:
        if isinstance(data["data"], dict):
            inner = data["data"]
        elif isinstance(data["data"], list):
            inner = {"games": data["data"]}

    # ── Shape GD: items list with gamedayRoster ───────────────────────────
    # Primary shape for /api/v4/events/gameday-rosters
    items_list = inner.get("items") if isinstance(inner, dict) else None
    if isinstance(items_list, list) and items_list:
        for game_obj in items_list:
            if not isinstance(game_obj, dict):
                continue
            meta = _extract_game_meta(game_obj)
            for side_key in ("homeTeam", "awayTeam"):
                team_obj = game_obj.get(side_key)
                if not isinstance(team_obj, dict):
                    continue
                roster = team_obj.get("gamedayRoster")
                if not isinstance(roster, list):
                    continue
                for player in roster:
                    if isinstance(player, dict):
                        row = _parse_gamedayRoster_player(
                            player, team_obj, meta, year, week, scraped_at
                        )
                        if row:
                            rows.append(row)
        if rows:
            # Successfully parsed the primary shape — skip other branches
            df = pd.DataFrame(rows)
            for col in GAMEDAY_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            return df[GAMEDAY_COLUMNS].copy()

    # ── Shape A/B: games/events list ─────────────────────────────────────
    games_list = None
    if isinstance(inner, dict):
        for key in ("games", "events", "matchups", "results"):
            if key in inner and isinstance(inner[key], list):
                games_list = inner[key]
                break

    if games_list is not None:
        for game_obj in games_list:
            if not isinstance(game_obj, dict):
                continue
            meta    = _extract_game_meta(game_obj)
            players = []
            for pkey in ("players", "roster", "athletes", "rosters"):
                p = game_obj.get(pkey)
                if isinstance(p, list):
                    players = p
                    break
                if isinstance(p, dict):
                    for v in p.values():
                        if isinstance(v, list):
                            players.extend(v)
                    break
            for player in players:
                if isinstance(player, dict):
                    row = _parse_player(player, meta, year, week, scraped_at)
                    if row:
                        rows.append(row)

    # ── Shape C: flat players list ────────────────────────────────────────
    elif isinstance(inner, dict) and "players" in inner:
        for player in inner["players"]:
            if isinstance(player, dict):
                row = _parse_player(player, {}, year, week, scraped_at)
                if row:
                    rows.append(row)

    # ── Shape D: top-level array ──────────────────────────────────────────
    elif isinstance(inner, list):
        for item in inner:
            if not isinstance(item, dict):
                continue
            meta    = _extract_game_meta(item)
            players = item.get("players") or item.get("roster") or item.get("athletes") or []
            for player in players:
                if isinstance(player, dict):
                    row = _parse_player(player, meta, year, week, scraped_at)
                    if row:
                        rows.append(row)

    # ── Fallback: scan all lists in the response ──────────────────────────
    if not rows and isinstance(inner, dict):
        print("  WARNING: Unknown response shape — scanning all nested lists for player data.")
        print(f"  Top-level keys: {list(inner.keys())}")
        for v in inner.values():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                for item in v:
                    row = _parse_player(item, {}, year, week, scraped_at)
                    if row:
                        rows.append(row)

    if not rows:
        print(f"  WARNING: No player rows parsed for year={year} week={week}.")
        top_keys = list(inner.keys()) if isinstance(inner, dict) else type(inner).__name__
        print(f"  Response keys: {top_keys}")
        return pd.DataFrame(columns=GAMEDAY_COLUMNS)

    df = pd.DataFrame(rows)
    for col in GAMEDAY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[GAMEDAY_COLUMNS].copy()


# ── Current week detection ────────────────────────────────────────────────
def current_pll_week(year: int = CURRENT_YEAR) -> int:
    """
    Estimate the current PLL week from today's date.
    Season typically starts in late April/early May.
    This is a heuristic — the app will try the detected week and fall back
    to week-1 if the API returns nothing.
    """
    today       = dt.date.today()
    season_start = dt.date(year, 4, 26)   # approximate 2026 season start
    if today < season_start:
        return 1
    days_in = (today - season_start).days
    week    = max(1, min(LAST_WEEK, 1 + days_in // 7))
    return week


# ── Write helpers ─────────────────────────────────────────────────────────
def save_gameday_csv(df: pd.DataFrame, year: int, week: int) -> Path:
    GAMEDAY_DIR.mkdir(parents=True, exist_ok=True)
    path = GAMEDAY_DIR / f"gameday_{year}_week{week:02d}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")
    return path


def save_latest(df: pd.DataFrame) -> None:
    GAMEDAY_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LATEST_PATH, index=False)
    print(f"  Updated latest → {LATEST_PATH}")


# ── Main scrape function ──────────────────────────────────────────────────
def scrape_week(year: int, week: int, token: str) -> pd.DataFrame:
    print(f"\nFetching gameday rosters: year={year} week={week}")
    data = fetch_gameday_rosters(year, week, token)
    if not data:
        print(f"  No data returned for week {week} (not yet available).")
        return pd.DataFrame(columns=GAMEDAY_COLUMNS)
    df = parse_response(data, year, week)
    if not df.empty:
        save_gameday_csv(df, year, week)
        teams = df["team_id"].nunique()
        active = df["is_active"].sum()
        print(f"  Parsed {len(df)} players across {teams} teams ({active} active).")
    return df


def scrape_current_week(year: int = CURRENT_YEAR, token: Optional[str] = None) -> pd.DataFrame:
    token = token or _get_token()
    week  = current_pll_week(year)
    print(f"Auto-detected week: {week}")
    df = scrape_week(year, week, token)
    # Try week-1 if current week returned nothing
    if df.empty and week > 1:
        print(f"  Falling back to week {week - 1}.")
        df = scrape_week(year, week - 1, token)
    if not df.empty:
        save_latest(df)
    return df


def scrape_all_weeks(year: int = CURRENT_YEAR, token: Optional[str] = None) -> pd.DataFrame:
    token  = token or _get_token()
    frames = []
    for week in range(FIRST_WEEK, LAST_WEEK + 1):
        df = scrape_week(year, week, token)
        if not df.empty:
            frames.append(df)
        elif week > current_pll_week(year):
            break   # stop when we reach a future week
    if not frames:
        return pd.DataFrame(columns=GAMEDAY_COLUMNS)
    all_df = pd.concat(frames, ignore_index=True)
    save_latest(all_df[all_df["week"] == all_df["week"].max()])
    return all_df


# ── Load helper (used by the app) ─────────────────────────────────────────
def load_gameday_roster(
    year: int,
    week: Optional[int] = None,
    game_home_id: Optional[str] = None,
    game_away_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Load cached gameday roster for a specific game.
    Called by GamedayRosterFilter in projection_engine_v3.py.

    Priority:
      1. Per-week CSV matching the game's week/year
      2. gameday_latest.csv
      3. Empty DataFrame (triggers fallback to current_rosters.csv)
    """
    status = {"available": False, "source": "none", "week": week, "year": year}

    # Try exact week file first
    if week is not None:
        path = GAMEDAY_DIR / f"gameday_{year}_week{week:02d}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                if game_home_id or game_away_id:
                    mask = pd.Series(True, index=df.index)
                    if game_home_id:
                        mask &= df["home_team_id"].str.upper() == str(game_home_id).upper()
                    if game_away_id:
                        mask &= df["away_team_id"].str.upper() == str(game_away_id).upper()
                    df = df[mask].copy()
                if not df.empty:
                    status.update({"available": True, "source": str(path), "rows": len(df)})
                    return df, status
            except Exception as exc:
                status["error"] = str(exc)

    # Try latest
    if LATEST_PATH.exists():
        try:
            df = pd.read_csv(LATEST_PATH)
            if year:
                df = df[df["year"] == year].copy()
            if game_home_id or game_away_id:
                mask = pd.Series(True, index=df.index)
                if game_home_id:
                    mask &= df["home_team_id"].str.upper() == str(game_home_id).upper()
                if game_away_id:
                    mask &= df["away_team_id"].str.upper() == str(game_away_id).upper()
                df = df[mask].copy()
            if not df.empty:
                status.update({"available": True, "source": str(LATEST_PATH), "rows": len(df)})
                return df, status
        except Exception as exc:
            status["error"] = str(exc)

    status["reason"] = "No gameday roster cache found. Run pll_gameday_roster_cache.py to populate."
    return pd.DataFrame(columns=GAMEDAY_COLUMNS), status


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Scrape PLL official gameday rosters and cache to CSV."
    )
    parser.add_argument("--year",       type=int, default=CURRENT_YEAR)
    parser.add_argument("--week",       type=int, default=None,
                        help="Week number (1-14). Omit to auto-detect current week.")
    parser.add_argument("--all-weeks",  action="store_true",
                        help="Scrape all weeks of the season.")
    parser.add_argument("--token",      type=str, default=None,
                        help="Bearer token (default: PLL_BEARER_TOKEN env var).")
    args = parser.parse_args()

    token = args.token or _get_token()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    print(f"PLL Gameday Roster Cache — started {started}")

    if args.all_weeks:
        df = scrape_all_weeks(args.year, token)
    elif args.week:
        df = scrape_week(args.year, args.week, token)
        if not df.empty:
            save_latest(df)
    else:
        df = scrape_current_week(args.year, token)

    if not df.empty:
        print(f"\nDone. {len(df)} total rows.")
        print(df.groupby(["team_id", "is_active"]).size().reset_index(name="count").to_string(index=False))
    else:
        print("\nNo data collected.")


if __name__ == "__main__":
    main()
