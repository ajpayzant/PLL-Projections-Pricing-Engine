# ============================================================
# PLL DATA PLATFORM — GITHUB WAREHOUSE BUILDER
# Source of truth: final Colab notebook builder blocks, ported for GitHub.
# ============================================================

from __future__ import annotations

import os
import re
import json
import gzip
import time
import hashlib
import datetime as dt
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
import duckdb

from tqdm.auto import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

pd.set_option("display.max_columns", 250)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 250)


def display(obj=None, *args, **kwargs):
    """Notebook-compatible display shim for GitHub Actions logs."""
    if obj is None:
        print()
        return

    try:
        if isinstance(obj, pd.DataFrame):
            if len(obj) > 40:
                print(obj.head(40).to_string(index=False))
                print(f"... ({len(obj)} rows total)")
            else:
                print(obj.to_string(index=False))
        else:
            print(obj)
    except Exception:
        print(repr(obj))


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass
    return out or default


REPO_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# BLOCK 1 — CONFIG, PATHS, TOKEN, SESSION, HELPERS
# ============================================================

# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = REPO_ROOT / "data"

SOURCE_DATA_DIR = PROJECT_ROOT / "source_data"
API_RESPONSES_DIR = SOURCE_DATA_DIR / "api_responses"

STANDARDIZED_DATA_DIR = PROJECT_ROOT / "standardized_data"
GAME_TABLES_DIR = STANDARDIZED_DATA_DIR / "game_tables"
REFERENCE_TABLES_DIR = STANDARDIZED_DATA_DIR / "reference_tables"

CURATED_DATA_DIR = PROJECT_ROOT / "curated_data"
CURATED_ALL_DIR = CURATED_DATA_DIR / "all_requested_seasons"

ANALYTICS_DATABASE_DIR = PROJECT_ROOT / "analytics_database"
QUALITY_CHECKS_DIR = PROJECT_ROOT / "quality_checks"
CONFIG_DIR = PROJECT_ROOT / "config"
EXPORT_DIR = PROJECT_ROOT / "exports"

RUN_ID = dt.datetime.now(dt.timezone.utc).strftime("run_%Y%m%d_%H%M%S")
RUN_CHECK_DIR = QUALITY_CHECKS_DIR / RUN_ID

for p in [
    PROJECT_ROOT,
    SOURCE_DATA_DIR,
    API_RESPONSES_DIR,
    STANDARDIZED_DATA_DIR,
    GAME_TABLES_DIR,
    REFERENCE_TABLES_DIR,
    CURATED_DATA_DIR,
    CURATED_ALL_DIR,
    ANALYTICS_DATABASE_DIR,
    QUALITY_CHECKS_DIR,
    RUN_CHECK_DIR,
    CONFIG_DIR,
    EXPORT_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Run check dir:", RUN_CHECK_DIR)

# -----------------------------
# Main config
# -----------------------------
TARGET_SEASONS = env_int_list("PLL_TARGET_SEASONS", [2022, 2023, 2024, 2025, 2026])
COMPETITION_TYPE = os.getenv("PLL_COMPETITION_TYPE", "regular").strip().lower()

EXPECTED_REGULAR_GAMES = {
    2022: 40,
    2023: 40,
    2024: 40,
    2025: 40,
    2026: None,   # ongoing / schedule-aware
}

PLL_STATS_SITE = "https://stats.premierlacrosseleague.com"
PLL_API_BASE = "https://api.stats.premierlacrosseleague.com/api/v4"
TIME_ZONE = "America/Los_Angeles"

FORCE_RECOLLECT = env_bool("PLL_FORCE_RECOLLECT", False)
FORCE_REDISCOVER = env_bool("PLL_FORCE_REDISCOVER", False)

MANUAL_SLUG_INVENTORY_FILE = CONFIG_DIR / "manual_slug_inventory.csv"

# -----------------------------
# Team mappings
# -----------------------------
TEAM_ID_CANONICAL_MAP = {
    "ATL": "ATL",
    "OUT": "OUT",
    "CAN": "CAN",
    "RED": "RED",
    "WAT": "WAT",
    "WHP": "WHP",
    "CHA": "CHA",
    "ARC": "ARC",
    "CHR": "OUT",   # Chrome historical franchise rolls into Outlaws.
}

TEAM_NAME_CANONICAL_MAP = {
    "ATL": "Atlas",
    "OUT": "Outlaws",
    "CAN": "Cannons",
    "RED": "Redwoods",
    "WAT": "Waterdogs",
    "WHP": "Whipsnakes",
    "CHA": "Chaos",
    "ARC": "Archers",
    "CHR": "Outlaws",
}

TEAM_NAME_LOOKUP_RAW = {
    "ATL": "Atlas",
    "OUT": "Outlaws",
    "CAN": "Cannons",
    "RED": "Redwoods",
    "WAT": "Waterdogs",
    "WHP": "Whipsnakes",
    "CHA": "Chaos",
    "ARC": "Archers",
    "CHR": "Chrome",
}

TEAM_DISPLAY_NAME_LOOKUP = {
    "ATL": "New York Atlas",
    "OUT": "Denver Outlaws",
    "CAN": "Boston Cannons",
    "RED": "California Redwoods",
    "WAT": "Philadelphia Waterdogs",
    "WHP": "Maryland Whipsnakes",
    "CHA": "Carolina Chaos",
    "ARC": "Utah Archers",
}

def canonical_team_id(team_id):
    if pd.isna(team_id):
        return pd.NA
    return TEAM_ID_CANONICAL_MAP.get(str(team_id).strip(), str(team_id).strip())

def canonical_team_name(team_id_raw, fallback_name=None):
    if pd.isna(team_id_raw):
        return fallback_name if fallback_name is not None else pd.NA
    team_id_raw = str(team_id_raw).strip()
    return TEAM_NAME_CANONICAL_MAP.get(
        team_id_raw,
        fallback_name if fallback_name is not None else team_id_raw
    )

def resolve_team_name_raw(team_id_raw, candidate_name=None):
    if pd.isna(team_id_raw) and pd.isna(candidate_name):
        return pd.NA

    raw_id = None if pd.isna(team_id_raw) else str(team_id_raw).strip()
    raw_name = None if pd.isna(candidate_name) else str(candidate_name).strip()

    if raw_name and raw_id and raw_name != raw_id:
        return raw_name

    if raw_name and not raw_id:
        return raw_name

    if raw_id:
        return TEAM_NAME_LOOKUP_RAW.get(raw_id, raw_id)

    return pd.NA

# -----------------------------
# Token
# -----------------------------
def clean_token_value(x):
    if x is None:
        return ""
    x = str(x).strip()
    x = x.replace("^", "").strip()
    x = re.sub(r"\s+", " ", x).strip()
    return x

PLL_BEARER_TOKEN = clean_token_value(
    os.environ.get("PLL_BEARER_TOKEN", "")
    or os.environ.get("PLL_API_TOKEN", "")
    or os.environ.get("PLL_TOKEN", "")
)

if not PLL_BEARER_TOKEN:
    raise RuntimeError(
        "PLL_BEARER_TOKEN is missing. Add it as a GitHub Actions repository secret."
    )

def token_preview(tok):
    return "SET" if tok else "MISSING"

print("Token loaded:", bool(PLL_BEARER_TOKEN))
print("Token preview:", token_preview(PLL_BEARER_TOKEN))

# -----------------------------
# HTTP session
# -----------------------------
def build_session(bearer_token=""):
    s = requests.Session()

    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "no-cache",
        "content-type": "application/json",
        "origin": "https://stats.premierlacrosseleague.com",
        "pragma": "no-cache",
        "referer": "https://stats.premierlacrosseleague.com/",
        "time-zone": TIME_ZONE,
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/146.0.0.0 Safari/537.36"
        ),
    }

    if bearer_token:
        tok = clean_token_value(bearer_token)
        headers["authorization"] = tok if tok.lower().startswith("bearer ") else f"Bearer {tok}"
        headers["authsource"] = "stats"

    s.headers.update(headers)
    return s

SESSION = build_session(PLL_BEARER_TOKEN)

print("Authorization header present:", "authorization" in SESSION.headers)

# -----------------------------
# URL builders
# -----------------------------
def event_list_url(year, season_segment=COMPETITION_TYPE):
    return f"{PLL_API_BASE}/events?year={year}&seasonSegment={season_segment}"

def event_summary_url(slug):
    return f"{PLL_API_BASE}/events/{slug}"

def player_game_stats_url(slug):
    return f"{PLL_API_BASE}/events/{slug}/players/stats"

def team_game_stats_url(slug):
    return f"{PLL_API_BASE}/events/{slug}/teams/stats"

# -----------------------------
# General helpers
# -----------------------------
def now_utc_iso():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def write_gzip_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def read_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

@retry(
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(4),
    reraise=True,
)
def fetch_url(url, session=None, timeout=30):
    if session is None:
        session = SESSION
    return session.get(url, timeout=timeout)

def fetch_json_with_cache(url, cache_path, session=None, timeout=30, force=False):
    if session is None:
        session = SESSION

    cache_path = Path(cache_path)

    if cache_path.exists() and not force:
        try:
            payload = read_gzip_json(cache_path)
            return payload, 200, "cached"
        except Exception:
            try:
                cache_path.unlink()
            except Exception:
                pass

    r = fetch_url(url, session=session, timeout=timeout)

    try:
        payload = r.json()
    except Exception:
        payload = None

    if r.status_code == 200 and payload is not None:
        write_gzip_json(cache_path, payload)

    return payload, r.status_code, "downloaded"

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def snake_case(s):
    s = str(s)
    s = re.sub(r"[%/\-]+", "_", s)
    s = re.sub(r"[^0-9A-Za-z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s

def to_num_scalar(x):
    try:
        v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    except Exception:
        v = np.nan
    return v

def coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def safe_nullable_int(series):
    s = pd.to_numeric(series, errors="coerce")
    non_null = s.dropna()
    if non_null.empty:
        return s.astype("Int64")
    if np.isclose(non_null % 1, 0).all():
        return s.round().astype("Int64")
    return s

def normalize_person_name(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9 ]+", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x if x else None

def extract_game_number_from_slug(slug):
    if pd.isna(slug):
        return pd.NA

    slug = str(slug)

    m1 = re.search(r"_game_(\d+)$", slug)
    if m1:
        return int(m1.group(1))

    m2 = re.search(r"^game-(\d+)-\d{4}-\d{2}-\d{2}$", slug)
    if m2:
        return int(m2.group(1))

    m3 = re.search(r"^(\d{4})-ev-(\d+)$", slug)
    if m3:
        return int(m3.group(2))

    return pd.NA

def extract_home_team_obj(data):
    return data.get("homeTeam", {}) or {}

def extract_away_team_obj(data):
    for key in ["visitorTeam", "awayTeam", "visitor", "away"]:
        obj = data.get(key, {}) or {}
        if obj:
            return obj
    return {}

def extract_team_id_from_obj(obj):
    if not isinstance(obj, dict):
        return pd.NA
    return obj.get("officialId") or obj.get("teamId") or obj.get("id")

def extract_team_name_from_obj(obj):
    if not isinstance(obj, dict):
        return pd.NA
    return (
        obj.get("name")
        or obj.get("fullName")
        or obj.get("teamName")
        or obj.get("nickname")
        or obj.get("officialId")
        or obj.get("teamId")
        or obj.get("id")
    )

def validate_event_payload(payload, season):
    data = safe_get(payload, "data", default={}) if payload else {}

    year_val = to_num_scalar(data.get("year"))
    event_id = data.get("eventId")
    event_numeric_id = data.get("id")
    season_segment = data.get("seasonSegment")
    slugname = data.get("slugname")
    start_time_unix = to_num_scalar(data.get("startTime"))
    event_status = data.get("eventStatus")

    valid = bool(
        not pd.isna(year_val)
        and int(year_val) == int(season)
        and event_id
        and season_segment == COMPETITION_TYPE
    )

    return {
        "valid": valid,
        "year": None if pd.isna(year_val) else int(year_val),
        "event_id": event_id,
        "event_numeric_id": event_numeric_id,
        "competition_type": season_segment,
        "slugname": slugname,
        "start_time_unix": None if pd.isna(start_time_unix) else int(start_time_unix),
        "event_status": event_status,
    }

def recursive_leaf_pairs(obj, prefix=""):
    pairs = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            pairs.extend(recursive_leaf_pairs(v, p))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            pairs.extend(recursive_leaf_pairs(v, p))

    else:
        pairs.append((prefix, obj))

    return pairs

def find_numeric_leaf_candidates(obj, normalized_terms):
    pairs = recursive_leaf_pairs(obj)
    out = []

    for raw_path, val in pairs:
        path_norm = snake_case(raw_path)
        if all(term in path_norm for term in normalized_terms):
            num = to_num_scalar(val)
            if not pd.isna(num):
                out.append((raw_path, num))

    return out

def coalesce_numeric_with_alt(item, direct_keys, alt_term_groups, allow_zero=True):
    for k in direct_keys:
        if k in item:
            val = to_num_scalar(item.get(k))
            if not pd.isna(val):
                if allow_zero or val != 0:
                    return val

    for term_group in alt_term_groups:
        cands = find_numeric_leaf_candidates(item, term_group)
        if cands:
            cands_sorted = sorted(cands, key=lambda x: (x[1] == 0, len(x[0])))
            best_val = cands_sorted[0][1]
            if allow_zero or best_val != 0:
                return best_val

    return np.nan

def derive_one_point_goals(total_goals, raw_one_point_goals, two_point_goals):
    tg = to_num_scalar(total_goals)
    rg = to_num_scalar(raw_one_point_goals)
    tw = to_num_scalar(two_point_goals)

    if not pd.isna(tg) and not pd.isna(tw):
        calc = tg - tw
        if pd.isna(rg) or not np.isclose(rg, calc):
            return calc

    return rg

def derive_scoring_points(one_point_goals, two_point_goals):
    one = to_num_scalar(one_point_goals)
    two = to_num_scalar(two_point_goals)

    if pd.isna(one) and pd.isna(two):
        return np.nan

    return (0 if pd.isna(one) else one) + 2 * (0 if pd.isna(two) else two)

def derive_player_points(raw_points, scoring_points, assists):
    rp = to_num_scalar(raw_points)
    sp = to_num_scalar(scoring_points)
    ast = to_num_scalar(assists)

    if not pd.isna(sp) and not pd.isna(ast):
        calc = sp + ast
        if pd.isna(rp) or not np.isclose(rp, calc):
            return calc

    return rp

def mode_or_first(s):
    s2 = s.dropna()
    if len(s2) == 0:
        return pd.NA
    mode = s2.mode()
    if len(mode) > 0:
        return mode.iloc[0]
    return s2.iloc[0]

def latest_non_null_by_game(g, col):
    if col not in g.columns:
        return pd.NA
    s = g.sort_values(["season", "game_number", "game_id"])[col].dropna()
    if len(s) == 0:
        return pd.NA
    return s.iloc[-1]

print("Config/helper block complete.")

# ============================================================
# BLOCK 2 — API SANITY CHECK
# ============================================================

sanity_rows = []

test_urls = [
    ("event_list_2026", event_list_url(2026)),
    ("event_summary_2025_game_1", event_summary_url("2025_game_1")),
    ("event_summary_2026_ev_1", event_summary_url("2026-ev-1")),
]

for label, url in test_urls:
    try:
        r = SESSION.get(url, timeout=30)

        try:
            payload = r.json()
        except Exception:
            payload = None

        items = safe_get(payload, "data", "items", default=None) if payload else None
        data = safe_get(payload, "data", default=None) if payload else None

        sanity_rows.append({
            "label": label,
            "url": url,
            "status_code": r.status_code,
            "has_json": payload is not None,
            "has_data": data is not None,
            "has_items": isinstance(items, list),
            "items_count": len(items) if isinstance(items, list) else None,
            "text_preview": r.text[:300],
        })

    except Exception as e:
        sanity_rows.append({
            "label": label,
            "url": url,
            "status_code": None,
            "has_json": False,
            "has_data": False,
            "has_items": False,
            "items_count": None,
            "text_preview": str(e)[:300],
        })

api_sanity_check = pd.DataFrame(sanity_rows)
api_sanity_check.to_csv(RUN_CHECK_DIR / "api_sanity_check.csv", index=False)

display(api_sanity_check)

if not api_sanity_check["has_data"].any():
    raise RuntimeError("API sanity check failed. Check token/API access before continuing.")

print("API sanity check passed.")

# ============================================================
# BLOCK 3 — DISCOVERY: FULL SCHEDULE + COMPLETED STAT INVENTORY
# ============================================================

def ensure_manual_slug_template():
    if not MANUAL_SLUG_INVENTORY_FILE.exists():
        pd.DataFrame(columns=["season", "slug", "note"]).to_csv(MANUAL_SLUG_INVENTORY_FILE, index=False)
        print(f"Created optional manual slug template: {MANUAL_SLUG_INVENTORY_FILE}")

ensure_manual_slug_template()

def load_manual_slug_inventory():
    if not MANUAL_SLUG_INVENTORY_FILE.exists():
        return pd.DataFrame(columns=["season", "slug", "note"])

    df = pd.read_csv(MANUAL_SLUG_INVENTORY_FILE)

    for c in ["season", "slug", "note"]:
        if c not in df.columns:
            df[c] = pd.NA

    df = df.dropna(subset=["season", "slug"]).copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["slug"] = df["slug"].astype(str).str.strip()
    df = df[df["season"].isin(TARGET_SEASONS)].copy()

    return df[["season", "slug", "note"]]

def fetch_event_list_for_year(year, season_segment=COMPETITION_TYPE):
    candidate_urls = [
        f"{PLL_API_BASE}/events?year={year}&seasonSegment={season_segment}",
        f"{PLL_API_BASE}/events?seasonSegment={season_segment}&year={year}",
        f"{PLL_API_BASE}/events?year={year}",
    ]

    probe_rows = []
    best_payload = None

    for url in candidate_urls:
        try:
            r = SESSION.get(url, timeout=30)

            try:
                payload = r.json()
            except Exception:
                payload = None

            items = safe_get(payload, "data", "items", default=[]) if payload else []

            probe_rows.append({
                "season": year,
                "url": url,
                "status_code": r.status_code,
                "ok": r.ok,
                "has_json": payload is not None,
                "items_count": len(items) if isinstance(items, list) else 0,
                "text_preview": r.text[:500],
            })

            if r.status_code == 200 and isinstance(items, list) and len(items) > 0 and best_payload is None:
                best_payload = payload

        except Exception as e:
            probe_rows.append({
                "season": year,
                "url": url,
                "status_code": None,
                "ok": False,
                "has_json": False,
                "items_count": 0,
                "text_preview": str(e)[:500],
            })

    return best_payload, pd.DataFrame(probe_rows)

def parse_event_list_payload(payload, year, season_segment=COMPETITION_TYPE):
    items = safe_get(payload, "data", "items", default=[]) if payload else []
    rows = []

    for item in items:
        if not isinstance(item, dict):
            continue

        item_year = item.get("year")
        item_segment = item.get("seasonSegment")

        if item_year is not None:
            try:
                if int(item_year) != int(year):
                    continue
            except Exception:
                pass

        if item_segment is not None and item_segment != season_segment:
            continue

        slug = item.get("slugname") or item.get("slug") or item.get("eventSlug")
        if not slug:
            continue

        start_time_unix = to_num_scalar(item.get("startTime"))
        game_date_guess = (
            pd.to_datetime(start_time_unix, unit="s", utc=True)
            if not pd.isna(start_time_unix)
            else pd.NaT
        )

        home_obj = extract_home_team_obj(item)
        away_obj = extract_away_team_obj(item)

        home_team_id_raw = extract_team_id_from_obj(home_obj)
        away_team_id_raw = extract_team_id_from_obj(away_obj)

        home_team_name_raw = extract_team_name_from_obj(home_obj)
        away_team_name_raw = extract_team_name_from_obj(away_obj)

        event_status = item.get("eventStatus")
        event_status_num = to_num_scalar(event_status)

        rows.append({
            "season": int(year),
            "slug": str(slug),
            "event_id": item.get("eventId"),
            "event_numeric_id": item.get("id"),
            "year": item_year,
            "competition_type": item_segment,
            "slugname": slug,
            "start_time_unix": start_time_unix,
            "game_date_guess": game_date_guess,
            "away_team_id_raw": away_team_id_raw,
            "away_team_name_raw": away_team_name_raw,
            "home_team_id_raw": home_team_id_raw,
            "home_team_name_raw": home_team_name_raw,
            "away_score": item.get("visitorScore") if item.get("visitorScore") is not None else item.get("awayScore"),
            "home_score": item.get("homeScore"),
            "event_status": event_status,
            "event_status_num": event_status_num,
            "event_status_label": "final" if event_status_num == 3 else ("scheduled" if event_status_num == 0 else "unknown"),
            "source": "event_list_endpoint",
            "discovery_source": "event_list_endpoint",
        })

    out = pd.DataFrame(rows)

    if len(out) == 0:
        return out

    out = out.drop_duplicates(subset=["season", "slug"]).copy()
    out = out.sort_values(["game_date_guess", "event_numeric_id", "slug"], na_position="last").reset_index(drop=True)
    out["game_number"] = np.arange(1, len(out) + 1)
    out["game_number_guess"] = out["game_number"]
    out["valid"] = True
    out["status_code"] = 200
    out["error"] = None

    return out

def probe_slug_summary(slug, season, discovery_source):
    cache_path = API_RESPONSES_DIR / f"season_{season}" / f"game_{slug}" / "event_summary.json.gz"

    payload, status_code, fetch_mode = fetch_json_with_cache(
        event_summary_url(slug),
        cache_path,
        force=FORCE_RECOLLECT
    )

    meta = validate_event_payload(payload, season)

    return {
        "season": season,
        "slug": slug,
        "status_code": status_code,
        "valid": bool(status_code == 200 and meta["valid"]),
        "year": meta["year"],
        "event_id": meta["event_id"],
        "event_numeric_id": meta["event_numeric_id"],
        "competition_type": meta["competition_type"],
        "slugname": meta["slugname"],
        "source": fetch_mode,
        "discovery_source": discovery_source,
        "start_time_unix": meta["start_time_unix"],
        "game_date_guess": pd.to_datetime(meta["start_time_unix"], unit="s", utc=True) if meta["start_time_unix"] else pd.NaT,
        "game_number_guess": extract_game_number_from_slug(slug),
        "event_status": meta["event_status"],
        "event_status_num": to_num_scalar(meta["event_status"]),
        "error": None,
    }

def validate_event_list_slugs(schedule_df, year):
    rows = []

    if len(schedule_df) == 0:
        return pd.DataFrame()

    for _, r in tqdm(schedule_df.iterrows(), total=len(schedule_df), desc=f"Validating event-list slugs {year}"):
        slug = str(r["slug"])

        try:
            row = probe_slug_summary(
                slug=slug,
                season=year,
                discovery_source="event_list_endpoint_validated_summary"
            )

            row["game_number"] = r.get("game_number")
            row["event_list_event_id"] = r.get("event_id")
            row["event_list_event_numeric_id"] = r.get("event_numeric_id")
            row["event_list_event_status"] = r.get("event_status")
            row["event_list_event_status_num"] = r.get("event_status_num")
            row["event_list_event_status_label"] = r.get("event_status_label")
            row["event_list_game_date_guess"] = r.get("game_date_guess")

            rows.append(row)

        except Exception as e:
            rows.append({
                "season": year,
                "slug": slug,
                "status_code": None,
                "valid": False,
                "year": None,
                "event_id": None,
                "event_numeric_id": None,
                "competition_type": None,
                "slugname": None,
                "source": "error",
                "discovery_source": "event_list_endpoint_validated_summary",
                "start_time_unix": None,
                "game_date_guess": pd.NaT,
                "game_number_guess": r.get("game_number"),
                "game_number": r.get("game_number"),
                "event_status": r.get("event_status"),
                "event_status_num": r.get("event_status_num"),
                "event_list_event_status": r.get("event_status"),
                "event_list_event_status_num": r.get("event_status_num"),
                "event_list_event_status_label": r.get("event_status_label"),
                "error": str(e)[:500],
            })

        time.sleep(0.03)

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(["game_number", "game_date_guess", "slug"], na_position="last").reset_index(drop=True)

    return out

def scan_cached_event_summaries(season):
    rows = []
    season_dir = API_RESPONSES_DIR / f"season_{season}"

    if not season_dir.exists():
        return pd.DataFrame(rows)

    for fp in sorted(season_dir.rglob("event_summary.json.gz")):
        slug = fp.parent.name.replace("game_", "", 1)

        try:
            payload = read_gzip_json(fp)
            meta = validate_event_payload(payload, season)

            rows.append({
                "season": season,
                "slug": slug,
                "status_code": 200,
                "valid": bool(meta["valid"]),
                "year": meta["year"],
                "event_id": meta["event_id"],
                "event_numeric_id": meta["event_numeric_id"],
                "competition_type": meta["competition_type"],
                "slugname": meta["slugname"],
                "source": "cached",
                "discovery_source": "cached_summary_scan",
                "start_time_unix": meta["start_time_unix"],
                "game_date_guess": pd.to_datetime(meta["start_time_unix"], unit="s", utc=True) if meta["start_time_unix"] else pd.NaT,
                "game_number_guess": extract_game_number_from_slug(slug),
                "game_number": pd.NA,
                "event_status": meta["event_status"],
                "event_status_num": to_num_scalar(meta["event_status"]),
                "error": None,
            })

        except Exception as e:
            rows.append({
                "season": season,
                "slug": slug,
                "status_code": None,
                "valid": False,
                "source": "cached",
                "discovery_source": "cached_summary_scan",
                "error": str(e)[:500],
            })

    return pd.DataFrame(rows)

def discover_numeric_season(season, max_guess=90, stop_after_consecutive_misses=12):
    rows = []
    valid_count = 0
    consecutive_misses = 0

    for game_number in range(1, max_guess + 1):
        slug = f"{season}_game_{game_number}"

        row = probe_slug_summary(slug, season, "numeric_probe")
        row["game_number"] = game_number
        row["game_number_guess"] = game_number
        rows.append(row)

        if row["valid"]:
            valid_count += 1
            consecutive_misses = 0
        else:
            consecutive_misses += 1

        if valid_count > 0 and consecutive_misses >= stop_after_consecutive_misses:
            break

        time.sleep(0.03)

    return pd.DataFrame(rows)

def discover_dated_season(season, start_date, end_date, max_game_number=65, stop_after_consecutive_missing_numbers=10):
    rows = []
    valid_count = 0
    consecutive_missing_numbers = 0
    date_list = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d").tolist()

    for game_number in range(1, max_game_number + 1):
        found_this_number = False

        for d in date_list:
            slug = f"game-{game_number}-{d}"

            row = probe_slug_summary(slug, season, f"dated_probe_{season}")
            row["game_number"] = game_number
            row["game_number_guess"] = game_number
            rows.append(row)

            if row["valid"]:
                valid_count += 1
                found_this_number = True
                break

            time.sleep(0.01)

        if found_this_number:
            consecutive_missing_numbers = 0
        else:
            consecutive_missing_numbers += 1

        if valid_count > 0 and consecutive_missing_numbers >= stop_after_consecutive_missing_numbers:
            break

    return pd.DataFrame(rows)

def build_discovery_inventories():
    event_list_probe_frames = []
    schedule_frames = []
    validated_frames = []

    # 1. Preferred discovery: official event-list endpoint.
    for season in TARGET_SEASONS:
        payload, probe_df = fetch_event_list_for_year(season, COMPETITION_TYPE)
        event_list_probe_frames.append(probe_df)

        parsed_schedule = parse_event_list_payload(payload, season, COMPETITION_TYPE)

        if len(parsed_schedule) > 0:
            schedule_frames.append(parsed_schedule)

            validated = validate_event_list_slugs(parsed_schedule, season)
            if len(validated) > 0:
                validated_frames.append(validated)

    event_list_probe_summary = (
        pd.concat(event_list_probe_frames, ignore_index=True)
        if event_list_probe_frames
        else pd.DataFrame()
    )

    event_list_schedule_inventory = (
        pd.concat(schedule_frames, ignore_index=True)
        if schedule_frames
        else pd.DataFrame()
    )

    validated_inventory = (
        pd.concat(validated_frames, ignore_index=True)
        if validated_frames
        else pd.DataFrame()
    )

    # 2. Fallback discovery where event-list endpoint fails or is incomplete.
    fallback_frames = []

    manual_df = load_manual_slug_inventory()
    if len(manual_df) > 0:
        manual_rows = []
        for _, r in manual_df.iterrows():
            season = int(r["season"])
            slug = str(r["slug"]).strip()
            row = probe_slug_summary(slug, season, "manual_slug_inventory")
            row["manual_note"] = r.get("note")
            manual_rows.append(row)
        fallback_frames.append(pd.DataFrame(manual_rows))

    for season in TARGET_SEASONS:
        expected = EXPECTED_REGULAR_GAMES.get(season)

        current_valid = validated_inventory[
            (pd.to_numeric(validated_inventory.get("season", pd.Series(dtype=float)), errors="coerce") == season)
            & (validated_inventory.get("valid", pd.Series(dtype=bool)) == True)
        ] if len(validated_inventory) > 0 else pd.DataFrame()

        need_fallback = len(current_valid) == 0 or (expected is not None and len(current_valid) < expected)

        if not need_fallback:
            continue

        cached_df = scan_cached_event_summaries(season)
        if len(cached_df) > 0:
            fallback_frames.append(cached_df)

        numeric_df = discover_numeric_season(season)
        if len(numeric_df) > 0:
            fallback_frames.append(numeric_df)

        if season == 2023:
            dated_df = discover_dated_season(season, "2023-06-01", "2023-09-30")
            if len(dated_df) > 0:
                fallback_frames.append(dated_df)

        if season == 2022:
            dated_df = discover_dated_season(season, "2022-06-01", "2022-09-30")
            if len(dated_df) > 0:
                fallback_frames.append(dated_df)

    fallback_inventory = (
        pd.concat(fallback_frames, ignore_index=True)
        if fallback_frames
        else pd.DataFrame()
    )

    # 3. Combine validated event-list and fallback.
    discovery_log_parts = []
    if len(validated_inventory) > 0:
        discovery_log_parts.append(validated_inventory)
    if len(fallback_inventory) > 0:
        discovery_log_parts.append(fallback_inventory)

    game_discovery_log = (
        pd.concat(discovery_log_parts, ignore_index=True)
        if discovery_log_parts
        else pd.DataFrame()
    )

    if len(game_discovery_log) == 0:
        return (
            event_list_probe_summary,
            event_list_schedule_inventory,
            game_discovery_log,
            pd.DataFrame(),
            pd.DataFrame(),
        )

    valid_discovered = game_discovery_log[game_discovery_log["valid"] == True].copy()

    # Prefer event-list validated rows over fallback rows.
    source_rank = {
        "event_list_endpoint_validated_summary": 1,
        "manual_slug_inventory": 2,
        "cached_summary_scan": 3,
        "numeric_probe": 4,
    }

    valid_discovered["discovery_rank"] = valid_discovered["discovery_source"].map(source_rank).fillna(9)

    valid_discovered = valid_discovered.sort_values(
        ["season", "event_id", "discovery_rank", "game_date_guess", "slug"],
        na_position="last"
    )

    valid_discovered = valid_discovered.drop_duplicates(
        subset=["season", "event_id"],
        keep="first"
    ).copy()

    valid_discovered = valid_discovered.sort_values(
        ["season", "game_date_guess", "game_number", "slug"],
        na_position="last"
    ).reset_index(drop=True)

    # Fill game_number by season if missing.
    valid_discovered["game_number"] = pd.to_numeric(valid_discovered["game_number"], errors="coerce")
    valid_discovered["game_number"] = valid_discovered.groupby("season").cumcount() + 1

    # 4. Build full schedule inventory.
    # If event-list schedule exists, use it for schedule. Otherwise use valid discovered rows.
    if len(event_list_schedule_inventory) > 0:
        schedule_inventory = event_list_schedule_inventory.copy()
    else:
        schedule_inventory = valid_discovered.copy()

    schedule_inventory = schedule_inventory.sort_values(
        ["season", "game_number", "game_date_guess", "slug"],
        na_position="last"
    ).reset_index(drop=True)

    # 5. Build stat-available inventory.
    #
    # Final rule:
    # - Historical completed seasons 2022-2025: use all validated regular-season games.
    #   Some historical PLL event-list rows can have imperfect event_status values even though stats exist.
    # - Ongoing/current/future seasons 2026+: use only event_status == 3 so scheduled games do not pollute stat tables.

    stat_parts = []

    for season in TARGET_SEASONS:
        discovered_season = valid_discovered[
            pd.to_numeric(valid_discovered["season"], errors="coerce") == season
        ].copy()

        schedule_season = schedule_inventory[
            pd.to_numeric(schedule_inventory["season"], errors="coerce") == season
        ].copy()

        if len(discovered_season) == 0:
            continue

        if season <= 2025:
            # Historical seasons: all validated regular-season games should be stat-available.
            stat_season = discovered_season.copy()

        else:
            # Ongoing/future seasons: only final games should be included in stat tables.
            if len(schedule_season) > 0 and "event_status_num" in schedule_season.columns:
                final_slugs = (
                    schedule_season[
                        pd.to_numeric(schedule_season["event_status_num"], errors="coerce") == 3
                    ]["slug"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

                stat_season = discovered_season[
                    discovered_season["slug"].astype(str).isin(final_slugs)
                ].copy()

            else:
                stat_season = pd.DataFrame(columns=discovered_season.columns)

        if len(stat_season) > 0:
            stat_season = stat_season.sort_values(
                ["game_date_guess", "game_number", "slug"],
                na_position="last"
            ).copy()

            stat_season["game_number"] = np.arange(1, len(stat_season) + 1)

            stat_parts.append(stat_season)

    stat_inventory = (
        pd.concat(stat_parts, ignore_index=True)
        if stat_parts
        else pd.DataFrame()
    )

    stat_inventory = stat_inventory.sort_values(
        ["season", "game_number", "game_date_guess", "slug"],
        na_position="last"
    ).reset_index(drop=True)

    return (
        event_list_probe_summary,
        schedule_inventory,
        game_discovery_log,
        valid_discovered,
        stat_inventory,
    )

event_list_probe_summary, season_schedule_inventory, game_discovery_log, season_slug_inventory, stat_slug_inventory = build_discovery_inventories()

# Save discovery outputs.
event_list_probe_summary.to_csv(RUN_CHECK_DIR / "event_list_probe_summary.csv", index=False)
season_schedule_inventory.to_csv(RUN_CHECK_DIR / "season_schedule_inventory_all_games.csv", index=False)
game_discovery_log.to_csv(RUN_CHECK_DIR / "game_discovery_log.csv", index=False)
season_slug_inventory.to_csv(RUN_CHECK_DIR / "season_slug_inventory_validated.csv", index=False)
stat_slug_inventory.to_csv(RUN_CHECK_DIR / "stat_slug_inventory_completed_games.csv", index=False)

# Build season_to_slugs for stat collection only.
season_to_slugs = {}

for season in TARGET_SEASONS:
    df = stat_slug_inventory[pd.to_numeric(stat_slug_inventory["season"], errors="coerce") == season].copy()
    df = df.sort_values(["game_number", "slug"])
    season_to_slugs[season] = df["slug"].dropna().astype(str).tolist()

print("Full schedule games by season:")
display(
    season_schedule_inventory
    .groupby("season", dropna=False)
    .agg(full_schedule_games=("slug", "nunique"))
    .reset_index()
)

print("Stat-available/completed games by season:")
display(
    stat_slug_inventory
    .groupby("season", dropna=False)
    .agg(stat_available_games=("slug", "nunique"))
    .reset_index()
)

print("Resolved stat-available slugs by season:")
for season in TARGET_SEASONS:
    expected = EXPECTED_REGULAR_GAMES.get(season)
    found = len(season_to_slugs.get(season, []))
    print(f" - {season}: {found} stat games found | expected={expected}")

display(
    season_schedule_inventory[
        [c for c in [
            "season", "game_number", "slug", "event_id", "game_date_guess",
            "away_team_id_raw", "home_team_id_raw", "away_score", "home_score",
            "event_status", "event_status_label"
        ] if c in season_schedule_inventory.columns]
    ].tail(60)
)

print("Discovery complete.")

# ============================================================
# BLOCK 4 — API COLLECTION, COMPLETED GAMES ONLY, NO PLAY-BY-PLAY
# ============================================================

def scrape_game_surfaces_no_pbp(season_to_slugs_map):
    rows = []

    for season, slug_list in season_to_slugs_map.items():
        for slug in tqdm(slug_list, desc=f"Collecting season {season}"):
            game_dir = API_RESPONSES_DIR / f"season_{season}" / f"game_{slug}"
            game_dir.mkdir(parents=True, exist_ok=True)

            surfaces = [
                ("event_summary", event_summary_url, "event_summary.json.gz"),
                ("player_game_stats", player_game_stats_url, "player_game_stats.json.gz"),
                ("team_game_stats", team_game_stats_url, "team_game_stats.json.gz"),
            ]

            for source_name, url_builder, filename in surfaces:
                cache_path = game_dir / filename

                try:
                    payload, status_code, fetch_mode = fetch_json_with_cache(
                        url_builder(slug),
                        cache_path,
                        force=FORCE_RECOLLECT
                    )

                    items = safe_get(payload, "data", "items", default=None) if payload else None

                    rows.append({
                        "season": season,
                        "game_slug": slug,
                        "source_name": source_name,
                        "http_status": status_code,
                        "fetch_mode": fetch_mode,
                        "raw_path": str(cache_path) if status_code == 200 else None,
                        "has_payload": payload is not None,
                        "has_items": isinstance(items, list),
                        "items_count": len(items) if isinstance(items, list) else None,
                        "error": None,
                    })

                    if fetch_mode == "downloaded":
                        time.sleep(0.04)

                except Exception as e:
                    rows.append({
                        "season": season,
                        "game_slug": slug,
                        "source_name": source_name,
                        "http_status": None,
                        "fetch_mode": "error",
                        "raw_path": None,
                        "has_payload": False,
                        "has_items": False,
                        "items_count": None,
                        "error": str(e)[:500],
                    })

    return pd.DataFrame(rows)

api_collection_log = scrape_game_surfaces_no_pbp(season_to_slugs)
api_collection_log.to_csv(RUN_CHECK_DIR / "api_collection_log.csv", index=False)

print("Collection status counts:")
display(
    api_collection_log
    .groupby(["season", "source_name", "fetch_mode", "http_status"], dropna=False)
    .size()
    .reset_index(name="n")
    .sort_values(["season", "source_name", "fetch_mode", "http_status"])
)

print("Collection item counts:")
display(
    api_collection_log
    .groupby(["season", "source_name"], dropna=False)
    .agg(
        games=("game_slug", "nunique"),
        min_items=("items_count", "min"),
        max_items=("items_count", "max"),
        total_items=("items_count", "sum"),
    )
    .reset_index()
)

display(api_collection_log.head(25))

print("API collection complete.")

# ============================================================
# BLOCK 5 — STANDARDIZED GAME TABLES
# ============================================================

game_manifest_rows = []
team_game_rows = []
player_game_rows = []
skipped_game_rows = []

for season in TARGET_SEASONS:
    for slug in season_to_slugs.get(season, []):
        game_dir = API_RESPONSES_DIR / f"season_{season}" / f"game_{slug}"

        summary_path = game_dir / "event_summary.json.gz"
        team_path = game_dir / "team_game_stats.json.gz"
        player_path = game_dir / "player_game_stats.json.gz"

        if not all([summary_path.exists(), team_path.exists(), player_path.exists()]):
            skipped_game_rows.append({
                "season": season,
                "slug": slug,
                "reason": "missing_required_api_surface",
                "summary_exists": summary_path.exists(),
                "team_exists": team_path.exists(),
                "player_exists": player_path.exists(),
            })
            continue

        try:
            summary_payload = read_gzip_json(summary_path)
            team_payload = read_gzip_json(team_path)
            player_payload = read_gzip_json(player_path)
        except Exception as e:
            skipped_game_rows.append({
                "season": season,
                "slug": slug,
                "reason": "could_not_read_cached_json",
                "error": str(e)[:500],
            })
            continue

        summary_data = safe_get(summary_payload, "data", default={}) or {}
        team_items = safe_get(team_payload, "data", "items", default=[]) or []
        player_items = safe_get(player_payload, "data", "items", default=[]) or []

        season_segment = summary_data.get("seasonSegment")

        if season_segment != COMPETITION_TYPE:
            skipped_game_rows.append({
                "season": season,
                "slug": slug,
                "reason": "non_regular_season_segment",
                "season_segment": season_segment,
            })
            continue

        if len(team_items) != 2:
            skipped_game_rows.append({
                "season": season,
                "slug": slug,
                "reason": "team_items_not_equal_2",
                "team_items": len(team_items),
                "player_items": len(player_items),
            })
            continue

        if len(player_items) == 0:
            skipped_game_rows.append({
                "season": season,
                "slug": slug,
                "reason": "no_player_items",
                "team_items": len(team_items),
                "player_items": len(player_items),
            })
            continue

        participant_ids_raw = [x.get("officialId") for x in team_items if x.get("officialId")]
        participant_ids_raw = list(dict.fromkeys(participant_ids_raw))

        home_obj = extract_home_team_obj(summary_data)
        away_obj = extract_away_team_obj(summary_data)

        home_team_id_raw = extract_team_id_from_obj(home_obj)
        away_team_id_raw = extract_team_id_from_obj(away_obj)

        home_team_name_raw_candidate = extract_team_name_from_obj(home_obj)
        away_team_name_raw_candidate = extract_team_name_from_obj(away_obj)

        if home_team_id_raw and not away_team_id_raw and len(participant_ids_raw) == 2:
            other_ids = [tid for tid in participant_ids_raw if tid != home_team_id_raw]
            if len(other_ids) == 1:
                away_team_id_raw = other_ids[0]

        if not home_team_id_raw and len(participant_ids_raw) == 2:
            home_team_id_raw = participant_ids_raw[0]
            away_team_id_raw = participant_ids_raw[1]

        home_team_name_raw = resolve_team_name_raw(home_team_id_raw, home_team_name_raw_candidate)
        away_team_name_raw = resolve_team_name_raw(away_team_id_raw, away_team_name_raw_candidate)

        home_team_id = canonical_team_id(home_team_id_raw)
        away_team_id = canonical_team_id(away_team_id_raw)

        home_team_name = canonical_team_name(home_team_id_raw, home_team_name_raw)
        away_team_name = canonical_team_name(away_team_id_raw, away_team_name_raw)

        game_slug = summary_data.get("slugname", slug)
        game_id = summary_data.get("eventId")
        event_numeric_id = summary_data.get("id")
        week = summary_data.get("week")
        league = summary_data.get("league")

        start_time_unix = to_num_scalar(summary_data.get("startTime"))
        start_time_utc = (
            pd.to_datetime(pd.Series([start_time_unix]), unit="s", utc=True).iloc[0]
            if not pd.isna(start_time_unix)
            else pd.NaT
        )
        game_date_utc = start_time_utc.date() if pd.notna(start_time_utc) else pd.NaT

        inv_row = stat_slug_inventory[
            (pd.to_numeric(stat_slug_inventory["season"], errors="coerce") == season)
            & (stat_slug_inventory["slug"].astype(str) == str(slug))
        ]

        if len(inv_row) > 0:
            game_number = int(pd.to_numeric(inv_row["game_number"], errors="coerce").iloc[0])
            game_number_from_slug = pd.to_numeric(inv_row["game_number_guess"], errors="coerce").iloc[0]
            schedule_slug = inv_row["slug"].iloc[0]
            schedule_event_status = inv_row["event_status"].iloc[0] if "event_status" in inv_row.columns else pd.NA
        else:
            game_number = extract_game_number_from_slug(game_slug)
            game_number_from_slug = extract_game_number_from_slug(game_slug)
            schedule_slug = slug
            schedule_event_status = summary_data.get("eventStatus")

        away_score = to_num_scalar(summary_data.get("visitorScore"))
        home_score = to_num_scalar(summary_data.get("homeScore"))

        winner_team_id_raw = pd.NA
        loser_team_id_raw = pd.NA
        winner_team_id = pd.NA
        loser_team_id = pd.NA

        if pd.notna(home_score) and pd.notna(away_score):
            if home_score > away_score:
                winner_team_id_raw = home_team_id_raw
                loser_team_id_raw = away_team_id_raw
                winner_team_id = home_team_id
                loser_team_id = away_team_id
            elif away_score > home_score:
                winner_team_id_raw = away_team_id_raw
                loser_team_id_raw = home_team_id_raw
                winner_team_id = away_team_id
                loser_team_id = home_team_id

        game_manifest_rows.append({
            "season": season,
            "competition_type": season_segment,
            "schedule_slug": schedule_slug,
            "game_slug": game_slug,
            "game_number_from_slug": game_number_from_slug,
            "game_number": game_number,
            "game_id": game_id,
            "event_numeric_id": event_numeric_id,
            "event_status": summary_data.get("eventStatus"),
            "schedule_event_status": schedule_event_status,
            "week": week,
            "league": league,
            "start_time_unix": start_time_unix,
            "start_time_utc": start_time_utc,
            "game_date_utc": game_date_utc,
            "venue": summary_data.get("venue"),
            "venue_location": summary_data.get("venueLocation"),
            "location": summary_data.get("location"),
            "period": summary_data.get("period"),
            "clock_minutes": summary_data.get("clockMinutes"),
            "clock_seconds": summary_data.get("clockSeconds"),
            "away_team_id_raw": away_team_id_raw,
            "away_team_name_raw": away_team_name_raw,
            "home_team_id_raw": home_team_id_raw,
            "home_team_name_raw": home_team_name_raw,
            "away_team_id": away_team_id,
            "away_team_name": away_team_name,
            "home_team_id": home_team_id,
            "home_team_name": home_team_name,
            "away_score": away_score,
            "home_score": home_score,
            "winner_team_id_raw": winner_team_id_raw,
            "loser_team_id_raw": loser_team_id_raw,
            "winner_team_id": winner_team_id,
            "loser_team_id": loser_team_id,
            "event_summary_path": str(summary_path),
            "team_game_stats_path": str(team_path),
            "player_game_stats_path": str(player_path),
        })

        side_map = {
            home_team_id_raw: {
                "team_id_raw": home_team_id_raw,
                "team_name_raw": home_team_name_raw,
                "team_id": home_team_id,
                "team_name": home_team_name,
                "opponent_team_id_raw": away_team_id_raw,
                "opponent_team_name_raw": away_team_name_raw,
                "opponent_team_id": away_team_id,
                "opponent_team_name": away_team_name,
                "is_home": 1,
            },
            away_team_id_raw: {
                "team_id_raw": away_team_id_raw,
                "team_name_raw": away_team_name_raw,
                "team_id": away_team_id,
                "team_name": away_team_name,
                "opponent_team_id_raw": home_team_id_raw,
                "opponent_team_name_raw": home_team_name_raw,
                "opponent_team_id": home_team_id,
                "opponent_team_name": home_team_name,
                "is_home": 0,
            },
        }

        # -----------------------------
        # Team rows
        # -----------------------------
        for item in team_items:
            team_id_raw = item.get("officialId")
            side = side_map.get(team_id_raw, {})

            resolved_team_name_raw = side.get(
                "team_name_raw",
                resolve_team_name_raw(team_id_raw)
            )

            goals = to_num_scalar(item.get("goals"))
            two_point_goals = to_num_scalar(item.get("twoPointGoals"))
            one_point_goals = derive_one_point_goals(goals, item.get("onePointGoals"), two_point_goals)

            touches = coalesce_numeric_with_alt(
                item,
                direct_keys=["touches"],
                alt_term_groups=[["touches"]],
                allow_zero=True,
            )

            total_passes = coalesce_numeric_with_alt(
                item,
                direct_keys=["totalPasses"],
                alt_term_groups=[["totalpasses"], ["passes"]],
                allow_zero=True,
            )

            time_in_possession = coalesce_numeric_with_alt(
                item,
                direct_keys=["timeInPossesion", "timeInPossession"],
                alt_term_groups=[["timeinpossesion"], ["timeinpossession"]],
                allow_zero=True,
            )

            time_in_possession_pct = coalesce_numeric_with_alt(
                item,
                direct_keys=["timeInPossesionPct", "timeInPossessionPct"],
                alt_term_groups=[["timeinpossesionpct"], ["timeinpossessionpct"]],
                allow_zero=True,
            )

            total_possessions = coalesce_numeric_with_alt(
                item,
                direct_keys=["totalPossessions"],
                alt_term_groups=[["totalpossessions"], ["possessions"]],
                allow_zero=True,
            )

            team_game_rows.append({
                "season": season,
                "competition_type": season_segment,
                "game_id": game_id,
                "schedule_slug": schedule_slug,
                "game_slug": game_slug,
                "game_number": game_number,
                "week": week,
                "game_date_utc": game_date_utc,
                "team_id_raw": team_id_raw,
                "team_name_raw": resolved_team_name_raw,
                "opponent_team_id_raw": side.get("opponent_team_id_raw"),
                "opponent_team_name_raw": side.get("opponent_team_name_raw"),
                "team_id": side.get("team_id", canonical_team_id(team_id_raw)),
                "team_name": side.get("team_name", canonical_team_name(team_id_raw, resolved_team_name_raw)),
                "opponent_team_id": side.get("opponent_team_id"),
                "opponent_team_name": side.get("opponent_team_name"),
                "is_home": side.get("is_home"),
                "scores": to_num_scalar(item.get("scores")),
                "goals": goals,
                "one_point_goals": one_point_goals,
                "two_point_goals": two_point_goals,
                "assists": to_num_scalar(item.get("assists")),
                "shots": to_num_scalar(item.get("shots")),
                "shot_pct": to_num_scalar(item.get("shotPct")),
                "shots_on_goal": to_num_scalar(item.get("shotsOnGoal")),
                "shots_on_goal_pct": to_num_scalar(item.get("shotsOnGoalPct")),
                "two_point_shots": to_num_scalar(item.get("twoPointShots")),
                "two_point_shot_pct": to_num_scalar(item.get("twoPointShotPct")),
                "two_point_shots_on_goal": to_num_scalar(item.get("twoPointShotsOnGoal")),
                "ground_balls": to_num_scalar(item.get("groundBalls")),
                "turnovers": to_num_scalar(item.get("turnovers")),
                "caused_turnovers": to_num_scalar(item.get("causedTurnovers")),
                "faceoff_pct": to_num_scalar(item.get("faceoffPct")),
                "faceoffs": to_num_scalar(item.get("faceoffs")),
                "faceoffs_won": to_num_scalar(item.get("faceoffsWon")),
                "faceoffs_lost": to_num_scalar(item.get("faceoffsLost")),
                "saves": to_num_scalar(item.get("saves")),
                "clean_saves": to_num_scalar(item.get("cleanSaves")),
                "messy_saves": to_num_scalar(item.get("messySaves")),
                "save_pct": to_num_scalar(item.get("savePct")),
                "clean_save_pct": to_num_scalar(item.get("cleanSavePct")),
                "scores_against": to_num_scalar(item.get("scoresAgainst")),
                "goals_against": to_num_scalar(item.get("goalsAgainst")),
                "num_penalties": to_num_scalar(item.get("numPenalties")),
                "pim": to_num_scalar(item.get("pim")),
                "power_play_pct": to_num_scalar(item.get("powerPlayPct")),
                "power_play_goals": to_num_scalar(item.get("powerPlayGoals")),
                "power_play_shots": to_num_scalar(item.get("powerPlayShots")),
                "power_play_goals_against": to_num_scalar(item.get("powerPlayGoalsAgainst")),
                "power_play_goals_against_pct": to_num_scalar(item.get("powerPlayGoalsAgainstPct")),
                "times_man_up": to_num_scalar(item.get("timesManUp")),
                "times_short_handed": to_num_scalar(item.get("timesShortHanded")),
                "man_down_pct": to_num_scalar(item.get("manDownPct")),
                "ride_attempts": to_num_scalar(item.get("rideAttempts")),
                "clear_attempts": to_num_scalar(item.get("clearAttempts")),
                "clears": to_num_scalar(item.get("clears")),
                "clear_pct": to_num_scalar(item.get("clearPct")),
                "shot_clock_expirations": to_num_scalar(item.get("shotClockExpirations")),
                "two_point_goals_against": to_num_scalar(item.get("twoPointGoalsAgainst")),
                "touches": touches,
                "total_passes": total_passes,
                "time_in_possession": time_in_possession,
                "time_in_possession_pct": time_in_possession_pct,
                "total_possessions": total_possessions,
                "source_path": str(team_path),
            })

        # -----------------------------
        # Player rows
        # -----------------------------
        for item in player_items:
            team_id_raw = item.get("teamId")
            side = side_map.get(team_id_raw, {})

            resolved_team_name_raw = side.get(
                "team_name_raw",
                resolve_team_name_raw(team_id_raw)
            )

            first_name = item.get("firstName")
            last_name = item.get("lastName")
            full_name = f"{first_name or ''} {last_name or ''}".strip()

            goals = to_num_scalar(item.get("goals"))
            two_point_goals = to_num_scalar(item.get("twoPointGoals"))
            one_point_goals = derive_one_point_goals(goals, item.get("onePointGoals"), two_point_goals)
            scoring_points = derive_scoring_points(one_point_goals, two_point_goals)
            points_total = derive_player_points(item.get("points"), scoring_points, item.get("assists"))

            shots = to_num_scalar(item.get("shots"))
            shots_on_goal = to_num_scalar(item.get("shotsOnGoal"))
            shots_on_goal_rate = np.nan if pd.isna(shots) or shots == 0 else shots_on_goal / shots

            player_game_rows.append({
                "season": season,
                "competition_type": season_segment,
                "game_id": game_id,
                "schedule_slug": schedule_slug,
                "game_slug": game_slug,
                "game_number": game_number,
                "week": week,
                "game_date_utc": game_date_utc,
                "team_id_raw": team_id_raw,
                "team_name_raw": resolved_team_name_raw,
                "opponent_team_id_raw": side.get("opponent_team_id_raw"),
                "opponent_team_name_raw": side.get("opponent_team_name_raw"),
                "team_id": side.get("team_id", canonical_team_id(team_id_raw)),
                "team_name": side.get("team_name", canonical_team_name(team_id_raw, resolved_team_name_raw)),
                "opponent_team_id": side.get("opponent_team_id"),
                "opponent_team_name": side.get("opponent_team_name"),
                "is_home": side.get("is_home"),
                "player_id": item.get("officialId"),
                "first_name": first_name,
                "last_name": last_name,
                "full_name": full_name,
                "player_name_key": normalize_person_name(full_name),
                "player_slug": item.get("slug"),
                "profile_url": item.get("profileUrl"),
                "position": item.get("position"),
                "position_name": item.get("positionName"),
                "jersey_number": to_num_scalar(item.get("jerseyNum")),
                "games_played_source": to_num_scalar(item.get("gamesPlayed")),
                "points": points_total,
                "scoring_points": scoring_points,
                "one_point_goals": one_point_goals,
                "two_point_goals": two_point_goals,
                "goals": goals,
                "assists": to_num_scalar(item.get("assists")),
                "shots": shots,
                "shot_pct": to_num_scalar(item.get("shotPct")),
                "shots_on_goal": shots_on_goal,
                "shots_on_goal_rate": shots_on_goal_rate,
                "two_point_shots": to_num_scalar(item.get("twoPointShots")),
                "saves": to_num_scalar(item.get("saves")),
                "clean_saves": to_num_scalar(item.get("cleanSaves")),
                "messy_saves": to_num_scalar(item.get("messySaves")),
                "save_pct": to_num_scalar(item.get("savePct")),
                "clean_save_pct": to_num_scalar(item.get("cleanSavePct")),
                "scores_against_average": to_num_scalar(item.get("GAA")),
                "two_point_gaa": to_num_scalar(item.get("twoPtGaa")),
                "scores_against": to_num_scalar(item.get("scoresAgainst")),
                "saa": to_num_scalar(item.get("saa")),
                "ground_balls": to_num_scalar(item.get("groundBalls")),
                "turnovers": to_num_scalar(item.get("turnovers")),
                "caused_turnovers": to_num_scalar(item.get("causedTurnovers")),
                "faceoffs_won": to_num_scalar(item.get("faceoffsWon")),
                "faceoffs_lost": to_num_scalar(item.get("faceoffsLost")),
                "faceoffs": to_num_scalar(item.get("faceoffs")),
                "faceoff_pct": to_num_scalar(item.get("faceoffPct")),
                "goals_against": to_num_scalar(item.get("goalsAgainst")),
                "two_point_goals_against": to_num_scalar(item.get("twoPointGoalsAgainst")),
                "num_penalties": to_num_scalar(item.get("numPenalties")),
                "pim": to_num_scalar(item.get("pim")),
                "fo_record": item.get("foRecord"),
                "assist_opportunities": to_num_scalar(item.get("assistOpportunities")),
                "touches": to_num_scalar(item.get("touches")),
                "total_passes": to_num_scalar(item.get("totalPasses")),
                "source_path": str(player_path),
            })

game_manifest = pd.DataFrame(game_manifest_rows)
team_game_stats = pd.DataFrame(team_game_rows)
player_game_stats = pd.DataFrame(player_game_rows)
skipped_games = pd.DataFrame(skipped_game_rows)

if len(game_manifest) > 0:
    game_manifest = game_manifest.sort_values(["season", "game_number", "game_slug"]).reset_index(drop=True)

if len(team_game_stats) > 0:
    team_game_stats = team_game_stats.sort_values(["season", "game_number", "team_id", "game_id"]).reset_index(drop=True)

if len(player_game_stats) > 0:
    player_game_stats = player_game_stats.sort_values(["season", "game_number", "team_id", "full_name", "game_id"]).reset_index(drop=True)

team_non_numeric = {
    "competition_type", "game_id", "schedule_slug", "game_slug", "game_date_utc",
    "team_id_raw", "team_name_raw", "opponent_team_id_raw", "opponent_team_name_raw",
    "team_id", "team_name", "opponent_team_id", "opponent_team_name", "source_path"
}

player_non_numeric = {
    "competition_type", "game_id", "schedule_slug", "game_slug", "game_date_utc",
    "team_id_raw", "team_name_raw", "opponent_team_id_raw", "opponent_team_name_raw",
    "team_id", "team_name", "opponent_team_id", "opponent_team_name",
    "player_id", "first_name", "last_name", "full_name", "player_name_key",
    "player_slug", "profile_url", "position", "position_name", "fo_record", "source_path"
}

if len(team_game_stats) > 0:
    team_game_stats = coerce_numeric(team_game_stats, [c for c in team_game_stats.columns if c not in team_non_numeric])

if len(player_game_stats) > 0:
    player_game_stats = coerce_numeric(player_game_stats, [c for c in player_game_stats.columns if c not in player_non_numeric])

for c in ["season", "game_number", "week", "is_home"]:
    if c in team_game_stats.columns:
        team_game_stats[c] = safe_nullable_int(team_game_stats[c])

for c in ["season", "game_number", "week", "is_home", "jersey_number", "games_played_source", "faceoffs", "faceoffs_won", "faceoffs_lost", "shots_on_goal"]:
    if c in player_game_stats.columns:
        player_game_stats[c] = safe_nullable_int(player_game_stats[c])

print("Standardized table shapes:")
print("game_manifest:", game_manifest.shape)
print("team_game_stats:", team_game_stats.shape)
print("player_game_stats:", player_game_stats.shape)
print("skipped_games:", skipped_games.shape)

display(game_manifest.head())
display(team_game_stats.head())
display(player_game_stats.head())

# Save standardized tables.
game_manifest.to_parquet(GAME_TABLES_DIR / "game_manifest.parquet", index=False)
team_game_stats.to_parquet(GAME_TABLES_DIR / "team_game_stats.parquet", index=False)
player_game_stats.to_parquet(GAME_TABLES_DIR / "player_game_stats.parquet", index=False)

game_manifest.to_csv(GAME_TABLES_DIR / "game_manifest.csv", index=False)
team_game_stats.to_csv(GAME_TABLES_DIR / "team_game_stats.csv", index=False)
player_game_stats.to_csv(GAME_TABLES_DIR / "player_game_stats.csv", index=False)
skipped_games.to_csv(RUN_CHECK_DIR / "skipped_games.csv", index=False)

print("Standardized tables saved.")

# ============================================================
# BLOCK 6 — POSSESSION CLEANUP + STAT COLUMN DEFINITIONS
# ============================================================
# Run after:
# - game_manifest
# - team_game_stats
# - player_game_stats
#
# Run before:
# - curated season/career/split tables
# - defensive/opponent marts
# - DuckDB warehouse save

def seconds_to_mmss_value(x):
    if x is None or pd.isna(x):
        return None

    x = int(round(float(x)))
    sign = "-" if x < 0 else ""
    x = abs(x)

    return f"{sign}{x // 60}:{x % 60:02d}"




def seconds_to_mmss_safe(x):
    """Compatibility alias used by the team style profile builder.

    The Colab notebook used seconds_to_mmss_value() in the possession cleanup
    block and later referenced seconds_to_mmss_safe() in the team style profile
    block. Keeping this alias preserves the same formatting behavior while
    making the GitHub script executable as one consolidated file.
    """
    return seconds_to_mmss_value(x)

def seconds_to_hhmmss_value(x):
    if x is None or pd.isna(x):
        return None

    x = int(round(float(x)))
    sign = "-" if x < 0 else ""
    x = abs(x)

    h = x // 3600
    m = (x % 3600) // 60
    s = x % 60

    if h > 0:
        return f"{sign}{h}:{m:02d}:{s:02d}"

    return f"{sign}{m}:{s:02d}"


def pct_display_value(x):
    if x is None or pd.isna(x):
        return None

    return f"{float(x) * 100:.2f}%"


def patch_team_possession_fields(team_df):
    out = team_df.copy()

    if len(out) == 0:
        return out, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ------------------------------------------------------------
    # Ensure expected columns exist
    # ------------------------------------------------------------
    required_cols = [
        "season",
        "game_id",
        "game_slug",
        "schedule_slug",
        "game_number",
        "game_date_utc",
        "team_id",
        "team_name",
        "opponent_team_id",
        "opponent_team_name",
        "scores",
        "scores_against",
        "shots",
        "turnovers",
        "shot_clock_expirations",
        "touches",
        "total_passes",
        "time_in_possession",
        "time_in_possession_pct",
        "total_possessions",
    ]

    for col in required_cols:
        if col not in out.columns:
            out[col] = np.nan

    numeric_cols = [
        "season",
        "game_number",
        "scores",
        "scores_against",
        "shots",
        "turnovers",
        "shot_clock_expirations",
        "touches",
        "total_passes",
        "time_in_possession",
        "time_in_possession_pct",
        "total_possessions",
    ]

    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # ------------------------------------------------------------
    # Preserve raw possession values
    # ------------------------------------------------------------
    out["total_possessions_raw"] = pd.to_numeric(out["total_possessions"], errors="coerce")
    out["time_in_possession_raw"] = pd.to_numeric(out["time_in_possession"], errors="coerce")
    out["time_in_possession_pct_raw"] = pd.to_numeric(out["time_in_possession_pct"], errors="coerce")

    # ------------------------------------------------------------
    # Official possession handling
    # Only trust totalPossessions in seasons where it is populated.
    # ------------------------------------------------------------
    possession_field_quality = (
        out
        .groupby("season", dropna=False)
        .agg(
            games=("game_id", "nunique"),
            team_rows=("game_id", "count"),
            total_possessions_nonzero=(
                "total_possessions_raw",
                lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())
            ),
            total_possessions_sum=(
                "total_possessions_raw",
                lambda s: pd.to_numeric(s, errors="coerce").sum()
            )
        )
        .reset_index()
    )

    usable_possession_seasons = possession_field_quality[
        possession_field_quality["total_possessions_nonzero"] > 0
    ]["season"].tolist()

    out["official_total_possessions"] = np.where(
        out["season"].isin(usable_possession_seasons),
        out["total_possessions_raw"],
        np.nan
    )

    out["official_total_possessions"] = np.where(
        pd.to_numeric(out["official_total_possessions"], errors="coerce") > 0,
        out["official_total_possessions"],
        np.nan
    )

    # ------------------------------------------------------------
    # Offensive sequence proxy
    # ------------------------------------------------------------
    out["offensive_sequence_proxy"] = (
        out["shots"].fillna(0)
        + out["turnovers"].fillna(0)
        + out["shot_clock_expirations"].fillna(0)
    )

    # ------------------------------------------------------------
    # Implied possession clock
    # ------------------------------------------------------------
    out["implied_game_clock_seconds"] = np.where(
        (out["time_in_possession_raw"].notna())
        & (out["time_in_possession_raw"] > 0)
        & (out["time_in_possession_pct_raw"].notna())
        & (out["time_in_possession_pct_raw"] > 0),
        out["time_in_possession_raw"] / out["time_in_possession_pct_raw"],
        np.nan
    )

    # ------------------------------------------------------------
    # Game-level possession quality
    # ------------------------------------------------------------
    game_possession_quality = (
        out
        .groupby(
            ["season", "game_id", "game_slug", "game_number", "game_date_utc"],
            dropna=False
        )
        .agg(
            team_rows=("team_id", "count"),
            combined_time_in_possession_raw=("time_in_possession_raw", "sum"),
            combined_time_in_possession_pct_raw=("time_in_possession_pct_raw", "sum"),
            combined_touches=("touches", "sum"),
            combined_passes=("total_passes", "sum"),
            combined_offensive_sequence_proxy=("offensive_sequence_proxy", "sum"),
            min_implied_game_clock=("implied_game_clock_seconds", "min"),
            max_implied_game_clock=("implied_game_clock_seconds", "max"),
            median_implied_game_clock=("implied_game_clock_seconds", "median"),
        )
        .reset_index()
    )

    game_possession_quality["implied_clock_range"] = (
        game_possession_quality["max_implied_game_clock"]
        - game_possession_quality["min_implied_game_clock"]
    )

    game_possession_quality["possession_data_status"] = np.select(
        [
            game_possession_quality["team_rows"] != 2,

            (
                game_possession_quality["combined_time_in_possession_raw"].fillna(0).eq(0)
                & game_possession_quality["combined_touches"].fillna(0).gt(0)
            ),

            game_possession_quality["implied_clock_range"] > 90,

            game_possession_quality["median_implied_game_clock"] > 3100,

            game_possession_quality["median_implied_game_clock"] < 2500,
        ],
        [
            "bad_team_row_count",
            "missing_possession_time",
            "team_denominator_mismatch",
            "extended_or_ot_clock",
            "short_or_provider_clock",
        ],
        default="normal"
    )

    game_possession_quality["possession_time_available"] = (
        game_possession_quality["possession_data_status"] != "missing_possession_time"
    )

    game_possession_quality["combined_time_in_possession_display"] = np.where(
        game_possession_quality["possession_time_available"],
        game_possession_quality["combined_time_in_possession_raw"].apply(seconds_to_mmss_value),
        None
    )

    game_possession_quality["median_implied_game_clock_display"] = (
        game_possession_quality["median_implied_game_clock"].apply(seconds_to_mmss_value)
    )

    game_possession_quality["possession_data_note"] = np.select(
        [
            game_possession_quality["possession_data_status"].eq("normal"),
            game_possession_quality["possession_data_status"].eq("extended_or_ot_clock"),
            game_possession_quality["possession_data_status"].eq("short_or_provider_clock"),
            game_possession_quality["possession_data_status"].eq("missing_possession_time"),
            game_possession_quality["possession_data_status"].eq("team_denominator_mismatch"),
            game_possession_quality["possession_data_status"].eq("bad_team_row_count"),
        ],
        [
            "Normal possession clock.",
            "Provider clock appears longer than regulation; likely overtime or extended provider denominator.",
            "Provider clock appears shorter than regulation; review before using possession time heavily.",
            "Possession time is unavailable even though touches/passes exist.",
            "Teams imply different possession-clock denominators; review manually.",
            "Game does not have exactly two team rows.",
        ],
        default="Review possession data."
    )

    # ------------------------------------------------------------
    # Merge possession quality back to team rows
    # ------------------------------------------------------------
    merge_cols = [
        "season",
        "game_id",
        "possession_data_status",
        "possession_time_available",
        "median_implied_game_clock",
        "median_implied_game_clock_display",
        "implied_clock_range",
        "possession_data_note",
    ]

    out = out.merge(
        game_possession_quality[merge_cols],
        on=["season", "game_id"],
        how="left"
    )

    # ------------------------------------------------------------
    # Clean possession time
    # True missing TOP games should be NaN, not 0.
    # ------------------------------------------------------------
    missing_top_mask = out["possession_data_status"].eq("missing_possession_time")

    out["time_in_possession"] = np.where(
        missing_top_mask,
        np.nan,
        out["time_in_possession_raw"]
    )

    out["time_in_possession_pct"] = np.where(
        missing_top_mask,
        np.nan,
        out["time_in_possession_pct_raw"]
    )

    out["time_in_possession_available_game"] = np.where(
        out["time_in_possession"].notna()
        & out["time_in_possession_pct"].notna()
        & out["possession_time_available"].fillna(False),
        1,
        0
    )

    out["time_in_possession_display"] = out["time_in_possession"].apply(seconds_to_mmss_value)
    out.loc[out["time_in_possession_available_game"].eq(0), "time_in_possession_display"] = None

    out["time_in_possession_pct_display"] = out["time_in_possession_pct"].apply(pct_display_value)
    out.loc[out["time_in_possession_available_game"].eq(0), "time_in_possession_pct_display"] = None

    out["implied_game_clock_display"] = out["implied_game_clock_seconds"].apply(seconds_to_mmss_value)

    # ------------------------------------------------------------
    # Possession style fields
    # ------------------------------------------------------------
    out["passes_per_touch"] = np.where(
        out["touches"] > 0,
        out["total_passes"] / out["touches"],
        np.nan
    )

    out["seconds_possession_per_touch"] = np.where(
        (out["touches"] > 0) & out["time_in_possession"].notna(),
        out["time_in_possession"] / out["touches"],
        np.nan
    )

    out["touches_per_offensive_sequence_proxy"] = np.where(
        out["offensive_sequence_proxy"] > 0,
        out["touches"] / out["offensive_sequence_proxy"],
        np.nan
    )

    out["passes_per_offensive_sequence_proxy"] = np.where(
        out["offensive_sequence_proxy"] > 0,
        out["total_passes"] / out["offensive_sequence_proxy"],
        np.nan
    )

    # ------------------------------------------------------------
    # Team-game possession quality mart
    # ------------------------------------------------------------
    possession_cols = [
        "season",
        "game_id",
        "game_slug",
        "schedule_slug",
        "game_number",
        "game_date_utc",
        "team_id",
        "team_name",
        "opponent_team_id",
        "opponent_team_name",
        "scores",
        "scores_against",
        "touches",
        "total_passes",
        "time_in_possession_raw",
        "time_in_possession",
        "time_in_possession_display",
        "time_in_possession_pct_raw",
        "time_in_possession_pct",
        "time_in_possession_pct_display",
        "implied_game_clock_seconds",
        "implied_game_clock_display",
        "median_implied_game_clock",
        "median_implied_game_clock_display",
        "total_possessions_raw",
        "official_total_possessions",
        "offensive_sequence_proxy",
        "passes_per_touch",
        "seconds_possession_per_touch",
        "touches_per_offensive_sequence_proxy",
        "passes_per_offensive_sequence_proxy",
        "time_in_possession_available_game",
        "possession_time_available",
        "possession_data_status",
        "possession_data_note",
    ]

    possession_cols = [c for c in possession_cols if c in out.columns]
    team_game_possession_quality = out[possession_cols].copy()

    # ------------------------------------------------------------
    # Season-level possession field quality summary
    # ------------------------------------------------------------
    possession_field_quality = (
        out
        .groupby("season", dropna=False)
        .agg(
            games=("game_id", "nunique"),
            team_rows=("game_id", "count"),
            time_in_possession_available_team_rows=("time_in_possession_available_game", "sum"),
            time_in_possession_raw_nonzero=(
                "time_in_possession_raw",
                lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())
            ),
            missing_possession_team_rows=(
                "possession_data_status",
                lambda s: int((s == "missing_possession_time").sum())
            ),
            total_possessions_nonzero=(
                "total_possessions_raw",
                lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())
            ),
            total_possessions_sum=("total_possessions_raw", "sum"),
        )
        .reset_index()
    )

    return out, possession_field_quality, game_possession_quality, team_game_possession_quality


team_game_stats, possession_field_quality, game_possession_quality, team_game_possession_quality = patch_team_possession_fields(team_game_stats)

try:
    possession_field_quality.to_csv(RUN_CHECK_DIR / "possession_field_quality_by_season.csv", index=False)
    game_possession_quality.to_csv(RUN_CHECK_DIR / "game_possession_quality.csv", index=False)
    team_game_possession_quality.to_csv(RUN_CHECK_DIR / "team_game_possession_quality.csv", index=False)
except Exception as e:
    print("Could not save possession QC files:", e)

print("Possession field quality:")
display(possession_field_quality)

print("\nGame possession quality status counts:")
display(
    game_possession_quality["possession_data_status"]
    .value_counts()
    .rename_axis("possession_data_status")
    .reset_index(name="games")
)

print("\nNon-normal possession games:")
display(
    game_possession_quality.loc[
        game_possession_quality["possession_data_status"] != "normal",
        [
            "possession_data_status",
            "season",
            "game_number",
            "game_date_utc",
            "game_slug",
            "combined_time_in_possession_raw",
            "combined_time_in_possession_display",
            "combined_time_in_possession_pct_raw",
            "median_implied_game_clock_display",
            "combined_touches",
            "combined_passes",
            "combined_offensive_sequence_proxy",
            "possession_data_note",
        ]
    ].sort_values(["season", "game_number"])
)


# ============================================================
# PLAYER / TEAM SUM COLUMN DEFINITIONS
# ============================================================

PLAYER_SUM_COLS = [
    "points", "scoring_points", "one_point_goals", "two_point_goals", "goals", "assists",
    "shots", "shots_on_goal", "two_point_shots",
    "saves", "clean_saves", "messy_saves", "scores_against", "saa",
    "ground_balls", "turnovers", "caused_turnovers",
    "faceoffs_won", "faceoffs_lost", "faceoffs",
    "goals_against", "two_point_goals_against",
    "num_penalties", "pim", "assist_opportunities", "touches", "total_passes"
]

TEAM_SUM_COLS = [
    "scores", "goals", "one_point_goals", "two_point_goals", "assists",
    "shots", "shots_on_goal", "two_point_shots", "two_point_shots_on_goal",
    "ground_balls", "turnovers", "caused_turnovers",
    "faceoffs", "faceoffs_won", "faceoffs_lost",
    "saves", "clean_saves", "messy_saves", "scores_against", "goals_against",
    "num_penalties", "pim", "power_play_goals", "power_play_shots",
    "power_play_goals_against", "times_man_up", "times_short_handed",
    "ride_attempts", "clear_attempts", "clears", "shot_clock_expirations",
    "two_point_goals_against", "touches", "total_passes",
    "time_in_possession", "time_in_possession_available_game",
    "official_total_possessions", "offensive_sequence_proxy"
]

PLAYER_SUM_COLS = [c for c in PLAYER_SUM_COLS if c in player_game_stats.columns]
TEAM_SUM_COLS = [c for c in TEAM_SUM_COLS if c in team_game_stats.columns]

print("PLAYER_SUM_COLS:", PLAYER_SUM_COLS)
print("TEAM_SUM_COLS:", TEAM_SUM_COLS)

# ============================================================
# BLOCK 7 — CURATED TABLES, SEASON TOTALS, CAREER TOTALS, SPLITS
# ============================================================

def add_player_rate_columns(df):
    out = df.copy()

    if "shots" in out.columns and "goals" in out.columns:
        out["shot_pct_calc"] = np.where(out["shots"] > 0, out["goals"] / out["shots"], np.nan)

    if "shots" in out.columns and "shots_on_goal" in out.columns:
        out["shots_on_goal_rate_calc"] = np.where(out["shots"] > 0, out["shots_on_goal"] / out["shots"], np.nan)

    if "faceoffs" in out.columns and "faceoffs_won" in out.columns:
        out["faceoff_pct_calc"] = np.where(out["faceoffs"] > 0, out["faceoffs_won"] / out["faceoffs"], np.nan)

    if "saa" in out.columns and "saves" in out.columns:
        out["save_pct_calc"] = np.where(out["saa"] > 0, out["saves"] / out["saa"], np.nan)

    if "games" in out.columns:
        for c in PLAYER_SUM_COLS:
            if c in out.columns:
                out[f"{c}_per_game"] = np.where(out["games"] > 0, out[c] / out["games"], np.nan)

    return out


def add_team_rate_columns(df):
    out = df.copy()

    if "shots" in out.columns and "goals" in out.columns:
        out["shot_pct_calc"] = np.where(out["shots"] > 0, out["goals"] / out["shots"], np.nan)

    if "shots" in out.columns and "shots_on_goal" in out.columns:
        out["shots_on_goal_rate_calc"] = np.where(out["shots"] > 0, out["shots_on_goal"] / out["shots"], np.nan)

    if "faceoffs" in out.columns and "faceoffs_won" in out.columns:
        out["faceoff_pct_calc"] = np.where(out["faceoffs"] > 0, out["faceoffs_won"] / out["faceoffs"], np.nan)

    if "clear_attempts" in out.columns and "clears" in out.columns:
        out["clear_pct_calc"] = np.where(out["clear_attempts"] > 0, out["clears"] / out["clear_attempts"], np.nan)

    if "games" in out.columns:
        for c in TEAM_SUM_COLS:
            if c not in out.columns:
                continue

            if c == "time_in_possession" and "time_in_possession_available_game" in out.columns:
                denom = pd.to_numeric(out["time_in_possession_available_game"], errors="coerce")
                out[f"{c}_per_game"] = np.where(denom > 0, out[c] / denom, np.nan)

                # If no valid TOP games exist, do not let all-missing sums appear as 0.
                out[c] = np.where(denom > 0, out[c], np.nan)

            else:
                out[f"{c}_per_game"] = np.where(out["games"] > 0, out[c] / out["games"], np.nan)

    # Possession style ratios from aggregated totals
    if "touches" in out.columns and "total_passes" in out.columns:
        out["passes_per_touch"] = np.where(out["touches"] > 0, out["total_passes"] / out["touches"], np.nan)

    if "touches" in out.columns and "time_in_possession" in out.columns:
        out["seconds_possession_per_touch"] = np.where(
            out["touches"] > 0,
            out["time_in_possession"] / out["touches"],
            np.nan
        )

    if "offensive_sequence_proxy" in out.columns and "touches" in out.columns:
        out["touches_per_offensive_sequence_proxy"] = np.where(
            out["offensive_sequence_proxy"] > 0,
            out["touches"] / out["offensive_sequence_proxy"],
            np.nan
        )

    if "offensive_sequence_proxy" in out.columns and "total_passes" in out.columns:
        out["passes_per_offensive_sequence_proxy"] = np.where(
            out["offensive_sequence_proxy"] > 0,
            out["total_passes"] / out["offensive_sequence_proxy"],
            np.nan
        )

    # Display-ready possession fields for downstream tables/app
    if "time_in_possession" in out.columns:
        out["time_in_possession_total_display"] = out["time_in_possession"].apply(seconds_to_hhmmss_value)

    if "time_in_possession_per_game" in out.columns:
        out["time_in_possession_per_game_display"] = out["time_in_possession_per_game"].apply(seconds_to_mmss_value)

    return out

# -----------------------------
# Team alias mapping
# -----------------------------
if len(team_game_stats) > 0:
    observed_team_id_raw = sorted(set(team_game_stats["team_id_raw"].dropna().astype(str).tolist()))
else:
    observed_team_id_raw = []

team_alias_mapping = pd.DataFrame({"team_id_raw": observed_team_id_raw})

if len(team_alias_mapping) > 0:
    team_alias_mapping["team_name_raw"] = team_alias_mapping["team_id_raw"].map(lambda x: TEAM_NAME_LOOKUP_RAW.get(x, x))
    team_alias_mapping["team_id"] = team_alias_mapping["team_id_raw"].map(canonical_team_id)
    team_alias_mapping["team_name"] = team_alias_mapping.apply(
        lambda r: canonical_team_name(r["team_id_raw"], r["team_name_raw"]),
        axis=1
    )

team_alias_mapping = team_alias_mapping.drop_duplicates().sort_values(["team_id", "team_id_raw"]).reset_index(drop=True)

# -----------------------------
# Team directory
# -----------------------------
if len(team_game_stats) > 0:
    observed_canonical_team_ids = sorted(set(team_game_stats["team_id"].dropna().astype(str).tolist()))
else:
    observed_canonical_team_ids = []

team_directory = pd.DataFrame({"team_id": observed_canonical_team_ids})

if len(team_directory) > 0:
    team_directory["team_name"] = team_directory["team_id"].map(lambda x: canonical_team_name(x, x))
    team_directory["team_display_name"] = team_directory["team_id"].map(TEAM_DISPLAY_NAME_LOOKUP).fillna(team_directory["team_name"])

team_directory = team_directory.drop_duplicates().sort_values("team_id").reset_index(drop=True)

# -----------------------------
# Player directory
# -----------------------------
if len(player_game_stats) > 0:
    player_directory = (
        player_game_stats
        .sort_values(["season", "game_number", "game_id"])
        .groupby("player_id", dropna=False)
        .agg({
            "full_name": "last",
            "first_name": "last",
            "last_name": "last",
            "player_name_key": "last",
            "player_slug": "last",
            "profile_url": "last",
            "position": mode_or_first,
            "position_name": mode_or_first,
            "jersey_number": mode_or_first,
            "team_id": lambda s: "|".join(sorted(set([str(x) for x in s.dropna()]))),
            "team_name": lambda s: "|".join(sorted(set([str(x) for x in s.dropna()]))),
            "game_id": "nunique",
            "season": "nunique",
        })
        .reset_index()
        .rename(columns={
            "game_id": "career_games_in_database",
            "season": "seasons_in_database",
        })
        .sort_values(["full_name", "player_id"])
        .reset_index(drop=True)
    )
else:
    player_directory = pd.DataFrame()

# -----------------------------
# Player season by team
# -----------------------------
if len(player_game_stats) > 0:
    player_season_stats_by_team = (
        player_game_stats
        .groupby(["season", "player_id", "full_name", "team_id", "team_name"], dropna=False)
        .agg(
            games=("game_id", "nunique"),
            position=("position", mode_or_first),
            position_name=("position_name", mode_or_first),
            first_game_date=("game_date_utc", "min"),
            last_game_date=("game_date_utc", "max"),
            **{c: (c, "sum") for c in PLAYER_SUM_COLS}
        )
        .reset_index()
    )
    player_season_stats_by_team = add_player_rate_columns(player_season_stats_by_team)
else:
    player_season_stats_by_team = pd.DataFrame()

# -----------------------------
# Player season, one row per player-season
# -----------------------------
player_season_rows = []

if len(player_game_stats) > 0:
    for keys, g in player_game_stats.groupby(["season", "player_id"], dropna=False):
        season, player_id = keys

        row = {
            "season": season,
            "player_id": player_id,
            "full_name": latest_non_null_by_game(g, "full_name"),
            "first_name": latest_non_null_by_game(g, "first_name"),
            "last_name": latest_non_null_by_game(g, "last_name"),
            "position": mode_or_first(g["position"]) if "position" in g.columns else pd.NA,
            "position_name": mode_or_first(g["position_name"]) if "position_name" in g.columns else pd.NA,
            "games": g["game_id"].nunique(),
            "teams": "|".join(sorted(set([str(x) for x in g["team_id"].dropna()]))),
            "team_names": "|".join(sorted(set([str(x) for x in g["team_name"].dropna()]))),
            "first_game_date": g["game_date_utc"].min(),
            "last_game_date": g["game_date_utc"].max(),
        }

        for c in PLAYER_SUM_COLS:
            row[c] = pd.to_numeric(g[c], errors="coerce").sum()

        player_season_rows.append(row)

player_season_stats = pd.DataFrame(player_season_rows)
player_season_stats = add_player_rate_columns(player_season_stats) if len(player_season_stats) > 0 else player_season_stats

# -----------------------------
# Player career, one row per player
# -----------------------------
player_career_rows = []

if len(player_game_stats) > 0:
    for player_id, g in player_game_stats.groupby("player_id", dropna=False):
        row = {
            "player_id": player_id,
            "full_name": latest_non_null_by_game(g, "full_name"),
            "first_name": latest_non_null_by_game(g, "first_name"),
            "last_name": latest_non_null_by_game(g, "last_name"),
            "position": mode_or_first(g["position"]) if "position" in g.columns else pd.NA,
            "position_name": mode_or_first(g["position_name"]) if "position_name" in g.columns else pd.NA,
            "games": g["game_id"].nunique(),
            "seasons": g["season"].nunique(),
            "teams": "|".join(sorted(set([str(x) for x in g["team_id"].dropna()]))),
            "team_names": "|".join(sorted(set([str(x) for x in g["team_name"].dropna()]))),
            "first_game_date": g["game_date_utc"].min(),
            "last_game_date": g["game_date_utc"].max(),
        }

        for c in PLAYER_SUM_COLS:
            row[c] = pd.to_numeric(g[c], errors="coerce").sum()

        player_career_rows.append(row)

player_career_stats = pd.DataFrame(player_career_rows)
player_career_stats = add_player_rate_columns(player_career_stats) if len(player_career_stats) > 0 else player_career_stats

# -----------------------------
# Score-based team game results
# -----------------------------
def add_score_based_team_result_flags(team_games):
    """
    Adds authoritative team-game win/loss/tie flags from the team-game score itself.

    This avoids using game_manifest winner_team_id / loser_team_id, which can be stale,
    missing, or inconsistent for newly completed games.
    """
    if team_games is None or len(team_games) == 0:
        return pd.DataFrame() if team_games is None else team_games.copy()

    out = team_games.copy()

    if "scores" not in out.columns or "scores_against" not in out.columns:
        raise KeyError("team_game_stats must contain 'scores' and 'scores_against' before record flags can be created.")

    out["scores"] = pd.to_numeric(out["scores"], errors="coerce")
    out["scores_against"] = pd.to_numeric(out["scores_against"], errors="coerce")

    valid_score = out["scores"].notna() & out["scores_against"].notna()

    out["win_flag"] = np.where(
        valid_score & (out["scores"] > out["scores_against"]),
        1,
        np.where(valid_score, 0, np.nan)
    )

    out["loss_flag"] = np.where(
        valid_score & (out["scores"] < out["scores_against"]),
        1,
        np.where(valid_score, 0, np.nan)
    )

    out["tie_flag"] = np.where(
        valid_score & (out["scores"] == out["scores_against"]),
        1,
        np.where(valid_score, 0, np.nan)
    )

    out["score_margin"] = np.where(
        valid_score,
        out["scores"] - out["scores_against"],
        np.nan
    )

    out["result"] = np.select(
        [
            out["win_flag"].eq(1),
            out["loss_flag"].eq(1),
            out["tie_flag"].eq(1),
        ],
        ["W", "L", "T"],
        default=pd.NA
    )

    return out


team_game_stats = add_score_based_team_result_flags(team_game_stats)

print("Score-based team result flags added to team_game_stats.")

if len(team_game_stats) > 0:
    display(
        team_game_stats[
            [
                "season",
                "game_number",
                "team_id",
                "team_name",
                "opponent_team_id",
                "opponent_team_name",
                "scores",
                "scores_against",
                "win_flag",
                "loss_flag",
                "tie_flag",
                "score_margin",
                "result",
            ]
        ]
        .sort_values(["season", "game_number", "team_name"])
        .tail(30)
    )


# -----------------------------
# Team season
# -----------------------------
if len(team_game_stats) > 0:
    team_season_stats = (
        team_game_stats
        .groupby(["season", "team_id", "team_name"], dropna=False)
        .agg(
            games=("game_id", "nunique"),
            wins=("win_flag", "sum"),
            losses=("loss_flag", "sum"),
            ties=("tie_flag", "sum"),
            score_margin=("score_margin", "sum"),
            first_game_date=("game_date_utc", "min"),
            last_game_date=("game_date_utc", "max"),
            **{c: (c, "sum") for c in TEAM_SUM_COLS}
        )
        .reset_index()
    )

    team_season_stats["win_pct"] = np.where(
        team_season_stats["games"] > 0,
        team_season_stats["wins"] / team_season_stats["games"],
        np.nan
    )

    team_season_stats["score_margin_per_game"] = np.where(
        team_season_stats["games"] > 0,
        team_season_stats["score_margin"] / team_season_stats["games"],
        np.nan
    )

    team_season_stats = add_team_rate_columns(team_season_stats)

    # Keep record fields in a clean numeric format.
    for c in ["wins", "losses", "ties"]:
        if c in team_season_stats.columns:
            team_season_stats[c] = pd.to_numeric(team_season_stats[c], errors="coerce")

else:
    team_season_stats = pd.DataFrame()


# -----------------------------
# Team career/franchise
# -----------------------------
if len(team_game_stats) > 0:
    team_career_stats = (
        team_game_stats
        .groupby(["team_id", "team_name"], dropna=False)
        .agg(
            games=("game_id", "nunique"),
            seasons=("season", "nunique"),
            wins=("win_flag", "sum"),
            losses=("loss_flag", "sum"),
            ties=("tie_flag", "sum"),
            score_margin=("score_margin", "sum"),
            first_game_date=("game_date_utc", "min"),
            last_game_date=("game_date_utc", "max"),
            **{c: (c, "sum") for c in TEAM_SUM_COLS}
        )
        .reset_index()
    )

    team_career_stats["win_pct"] = np.where(
        team_career_stats["games"] > 0,
        team_career_stats["wins"] / team_career_stats["games"],
        np.nan
    )

    team_career_stats["score_margin_per_game"] = np.where(
        team_career_stats["games"] > 0,
        team_career_stats["score_margin"] / team_career_stats["games"],
        np.nan
    )

    team_career_stats = add_team_rate_columns(team_career_stats)

    for c in ["wins", "losses", "ties"]:
        if c in team_career_stats.columns:
            team_career_stats[c] = pd.to_numeric(team_career_stats[c], errors="coerce")

else:
    team_career_stats = pd.DataFrame()


# -----------------------------
# Opponent splits
# -----------------------------
if len(player_game_stats) > 0:
    player_vs_opponent_stats = (
        player_game_stats
        .groupby(["player_id", "full_name", "opponent_team_id", "opponent_team_name"], dropna=False)
        .agg(
            games=("game_id", "nunique"),
            position=("position", mode_or_first),
            position_name=("position_name", mode_or_first),
            first_game_date=("game_date_utc", "min"),
            last_game_date=("game_date_utc", "max"),
            **{c: (c, "sum") for c in PLAYER_SUM_COLS}
        )
        .reset_index()
    )

    player_vs_opponent_stats = add_player_rate_columns(player_vs_opponent_stats)

else:
    player_vs_opponent_stats = pd.DataFrame()


if len(team_game_stats) > 0:
    team_vs_opponent_stats = (
        team_game_stats
        .groupby(["team_id", "team_name", "opponent_team_id", "opponent_team_name"], dropna=False)
        .agg(
            games=("game_id", "nunique"),
            wins=("win_flag", "sum"),
            losses=("loss_flag", "sum"),
            ties=("tie_flag", "sum"),
            score_margin=("score_margin", "sum"),
            first_game_date=("game_date_utc", "min"),
            last_game_date=("game_date_utc", "max"),
            **{c: (c, "sum") for c in TEAM_SUM_COLS}
        )
        .reset_index()
    )

    team_vs_opponent_stats["win_pct"] = np.where(
        team_vs_opponent_stats["games"] > 0,
        team_vs_opponent_stats["wins"] / team_vs_opponent_stats["games"],
        np.nan
    )

    team_vs_opponent_stats["score_margin_per_game"] = np.where(
        team_vs_opponent_stats["games"] > 0,
        team_vs_opponent_stats["score_margin"] / team_vs_opponent_stats["games"],
        np.nan
    )

    team_vs_opponent_stats = add_team_rate_columns(team_vs_opponent_stats)

else:
    team_vs_opponent_stats = pd.DataFrame()


print("Curated base tables created.")
print("player_directory:", player_directory.shape)
print("player_season_stats:", player_season_stats.shape)
print("player_career_stats:", player_career_stats.shape)
print("team_season_stats:", team_season_stats.shape)
print("team_career_stats:", team_career_stats.shape)
print("player_vs_opponent_stats:", player_vs_opponent_stats.shape)
print("team_vs_opponent_stats:", team_vs_opponent_stats.shape)

if len(team_season_stats) > 0:
    print("\nScore-based team records check:")
    display(
        team_season_stats[
            ["season", "team_id", "team_name", "games", "wins", "losses", "ties", "win_pct", "scores", "scores_against"]
        ]
        .sort_values(["season", "team_name"])
        .tail(40)
    )

# ============================================================
# BLOCK 8 — LAST 5 / LAST 10 SPLIT TABLES
# ============================================================

def build_player_last_n_stats(player_games, n=5, by_season=False):
    if len(player_games) == 0:
        return pd.DataFrame()

    base = player_games.sort_values(["season", "game_date_utc", "game_number", "game_id"]).copy()

    group_cols = ["player_id"]
    if by_season:
        group_cols = ["season", "player_id"]

    rows = []

    for keys, g in base.groupby(group_cols, dropna=False):
        g_last = g.tail(n).copy()

        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row["full_name"] = latest_non_null_by_game(g_last, "full_name")
        row["position"] = mode_or_first(g_last["position"]) if "position" in g_last.columns else pd.NA
        row["position_name"] = mode_or_first(g_last["position_name"]) if "position_name" in g_last.columns else pd.NA
        row["split_type"] = f"last_{n}"
        row["games"] = g_last["game_id"].nunique()
        row["first_game_date"] = g_last["game_date_utc"].min()
        row["last_game_date"] = g_last["game_date_utc"].max()
        row["opponents"] = "|".join(sorted(set([str(x) for x in g_last["opponent_team_id"].dropna()])))
        row["teams"] = "|".join(sorted(set([str(x) for x in g_last["team_id"].dropna()])))

        for c in PLAYER_SUM_COLS:
            row[c] = pd.to_numeric(g_last[c], errors="coerce").sum()

        rows.append(row)

    out = pd.DataFrame(rows)
    out = add_player_rate_columns(out)

    return out

def build_team_last_n_stats(team_games, n=5, by_season=False):
    if len(team_games) == 0:
        return pd.DataFrame()

    base = team_games.sort_values(["season", "game_date_utc", "game_number", "game_id"]).copy()

    group_cols = ["team_id"]
    if by_season:
        group_cols = ["season", "team_id"]

    rows = []

    for keys, g in base.groupby(group_cols, dropna=False):
        g_last = g.tail(n).copy()

        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row["team_name"] = latest_non_null_by_game(g_last, "team_name")
        row["split_type"] = f"last_{n}"
        row["games"] = g_last["game_id"].nunique()
        row["first_game_date"] = g_last["game_date_utc"].min()
        row["last_game_date"] = g_last["game_date_utc"].max()
        row["opponents"] = "|".join(sorted(set([str(x) for x in g_last["opponent_team_id"].dropna()])))

        for c in TEAM_SUM_COLS:
            row[c] = pd.to_numeric(g_last[c], errors="coerce").sum()

        rows.append(row)

    out = pd.DataFrame(rows)
    out = add_team_rate_columns(out)

    return out

player_last5_stats = build_player_last_n_stats(player_game_stats, n=5, by_season=False)
player_last10_stats = build_player_last_n_stats(player_game_stats, n=10, by_season=False)

player_season_last5_stats = build_player_last_n_stats(player_game_stats, n=5, by_season=True)
player_season_last10_stats = build_player_last_n_stats(player_game_stats, n=10, by_season=True)

team_last5_stats = build_team_last_n_stats(team_game_stats, n=5, by_season=False)
team_last10_stats = build_team_last_n_stats(team_game_stats, n=10, by_season=False)

team_season_last5_stats = build_team_last_n_stats(team_game_stats, n=5, by_season=True)
team_season_last10_stats = build_team_last_n_stats(team_game_stats, n=10, by_season=True)

print("Rolling split tables created.")
print("player_last5_stats:", player_last5_stats.shape)
print("player_last10_stats:", player_last10_stats.shape)
print("player_season_last5_stats:", player_season_last5_stats.shape)
print("player_season_last10_stats:", player_season_last10_stats.shape)
print("team_last5_stats:", team_last5_stats.shape)
print("team_last10_stats:", team_last10_stats.shape)
print("team_season_last5_stats:", team_season_last5_stats.shape)
print("team_season_last10_stats:", team_season_last10_stats.shape)

display(player_last5_stats.head())
display(team_last5_stats.head())

# ============================================================
# BLOCK 9 — CLEAN SCHEDULE TABLES
# ============================================================

def build_clean_schedule_table(schedule_inventory):
    if len(schedule_inventory) == 0:
        return pd.DataFrame()

    out = schedule_inventory.copy()

    for c in ["event_status_num", "event_status", "away_score", "home_score"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "event_status_label" not in out.columns:
        out["event_status_label"] = np.select(
            [
                out.get("event_status_num", pd.Series(index=out.index, dtype=float)) == 3,
                out.get("event_status_num", pd.Series(index=out.index, dtype=float)) == 0,
            ],
            ["final", "scheduled"],
            default="unknown"
        )

    out["away_team_id"] = out["away_team_id_raw"].map(canonical_team_id) if "away_team_id_raw" in out.columns else pd.NA
    out["home_team_id"] = out["home_team_id_raw"].map(canonical_team_id) if "home_team_id_raw" in out.columns else pd.NA

    out["away_team_name"] = out.apply(
        lambda r: canonical_team_name(r.get("away_team_id_raw"), r.get("away_team_name_raw")),
        axis=1
    )

    out["home_team_name"] = out.apply(
        lambda r: canonical_team_name(r.get("home_team_id_raw"), r.get("home_team_name_raw")),
        axis=1
    )

    keep_cols = [
        "season",
        "game_number",
        "slug",
        "event_id",
        "event_numeric_id",
        "game_date_guess",
        "away_team_id_raw",
        "away_team_name_raw",
        "home_team_id_raw",
        "home_team_name_raw",
        "away_team_id",
        "away_team_name",
        "home_team_id",
        "home_team_name",
        "away_score",
        "home_score",
        "event_status",
        "event_status_num",
        "event_status_label",
        "discovery_source",
        "source",
    ]

    keep_cols = [c for c in keep_cols if c in out.columns]

    out = out[keep_cols].copy()
    out = out.sort_values(["season", "game_number", "game_date_guess", "slug"], na_position="last").reset_index(drop=True)

    return out

game_schedule_all = build_clean_schedule_table(season_schedule_inventory)

game_schedule_2026 = game_schedule_all[
    pd.to_numeric(game_schedule_all["season"], errors="coerce") == 2026
].copy()

print("game_schedule_all:", game_schedule_all.shape)
print("game_schedule_2026:", game_schedule_2026.shape)

display(game_schedule_2026.head(25))

# ============================================================
# BLOCK 10 — QUALITY CHECKS
# ============================================================

quality_rows = []

def add_qc_check(check_name, status, actual=None, expected=None, notes=None):
    quality_rows.append({
        "check_name": check_name,
        "status": status,
        "actual": actual,
        "expected": expected,
        "notes": notes,
    })

add_qc_check("game_manifest_rows", "info", len(game_manifest), None, "Number of completed/stat-available regular-season games parsed.")
add_qc_check("team_game_stats_rows", "info", len(team_game_stats), None, "Should usually be 2x game_manifest rows.")
add_qc_check("player_game_stats_rows", "info", len(player_game_stats), None, "One row per player-game.")
add_qc_check("skipped_games_rows", "info", len(skipped_games), None, "Games skipped due to missing surfaces or invalid stats.")
add_qc_check("full_schedule_inventory_rows", "info", len(season_schedule_inventory), None, "Full schedule, including future games.")
add_qc_check("stat_slug_inventory_rows", "info", len(stat_slug_inventory), None, "Completed/stat-available games only.")

# Expected completed games by season.
if len(game_manifest) > 0:
    games_by_season = game_manifest.groupby("season")["game_id"].nunique().reset_index(name="games")

    for _, r in games_by_season.iterrows():
        season = int(r["season"])
        actual_games = int(r["games"])
        expected_games = EXPECTED_REGULAR_GAMES.get(season)

        if expected_games is None:
            status = "info" if actual_games > 0 else "warning"
            notes = "Ongoing season; only completed games are expected."
        else:
            status = "pass" if actual_games == expected_games else "warning"
            notes = "Completed regular season expected count."

        add_qc_check(
            f"expected_stat_game_count_{season}",
            status,
            actual_games,
            expected_games,
            notes
        )

# Full schedule by season.
if len(season_schedule_inventory) > 0:
    sched_by_season = season_schedule_inventory.groupby("season")["slug"].nunique().reset_index(name="schedule_games")

    for _, r in sched_by_season.iterrows():
        season = int(r["season"])
        add_qc_check(
            f"full_schedule_game_count_{season}",
            "info",
            int(r["schedule_games"]),
            None,
            "Full schedule count, including scheduled/future games."
        )

# Team rows per game.
if len(team_game_stats) > 0:
    team_rows_per_game = team_game_stats.groupby("game_id").size().reset_index(name="team_rows")
    bad_team_row_games = team_rows_per_game[team_rows_per_game["team_rows"] != 2].copy()

    add_qc_check(
        "exactly_two_team_rows_per_game",
        "pass" if len(bad_team_row_games) == 0 else "warning",
        len(bad_team_row_games),
        0,
        "Each completed game should have exactly two team-game rows."
    )

    bad_team_row_games.to_csv(RUN_CHECK_DIR / "bad_team_row_games.csv", index=False)

# Duplicate keys.
if len(team_game_stats) > 0:
    team_dupes = team_game_stats.duplicated(subset=["game_id", "team_id"], keep=False).sum()
    add_qc_check(
        "duplicate_team_game_keys",
        "pass" if team_dupes == 0 else "fail",
        int(team_dupes),
        0,
        "Duplicate game_id/team_id rows."
    )

if len(player_game_stats) > 0:
    player_dupes = player_game_stats.duplicated(subset=["game_id", "player_id", "team_id"], keep=False).sum()
    add_qc_check(
        "duplicate_player_game_keys",
        "pass" if player_dupes == 0 else "fail",
        int(player_dupes),
        0,
        "Duplicate game_id/player_id/team_id rows."
    )

# Team scoring formula.
if len(team_game_stats) > 0:
    scoring_check = team_game_stats.copy()
    scoring_check["scores_calc"] = scoring_check["one_point_goals"] + 2 * scoring_check["two_point_goals"]
    scoring_check["score_diff"] = scoring_check["scores"] - scoring_check["scores_calc"]

    bad_score_formula = scoring_check[
        scoring_check["score_diff"].notna() &
        (scoring_check["score_diff"].abs() > 0.001)
    ].copy()

    add_qc_check(
        "team_scores_formula",
        "pass" if len(bad_score_formula) == 0 else "warning",
        len(bad_score_formula),
        0,
        "scores should equal one_point_goals + 2 * two_point_goals."
    )

    bad_score_formula.to_csv(RUN_CHECK_DIR / "bad_team_score_formula_rows.csv", index=False)

# Player scoring formulas.
if len(player_game_stats) > 0:
    player_scoring_check = player_game_stats.copy()
    player_scoring_check["scoring_points_calc"] = player_scoring_check["one_point_goals"] + 2 * player_scoring_check["two_point_goals"]
    player_scoring_check["scoring_points_diff"] = player_scoring_check["scoring_points"] - player_scoring_check["scoring_points_calc"]

    bad_player_scoring = player_scoring_check[
        player_scoring_check["scoring_points_diff"].notna() &
        (player_scoring_check["scoring_points_diff"].abs() > 0.001)
    ].copy()

    player_scoring_check["points_calc"] = player_scoring_check["scoring_points"] + player_scoring_check["assists"]
    player_scoring_check["points_diff"] = player_scoring_check["points"] - player_scoring_check["points_calc"]

    bad_player_points = player_scoring_check[
        player_scoring_check["points_diff"].notna() &
        (player_scoring_check["points_diff"].abs() > 0.001)
    ].copy()

    add_qc_check(
        "player_scoring_points_formula",
        "pass" if len(bad_player_scoring) == 0 else "warning",
        len(bad_player_scoring),
        0,
        "scoring_points should equal one_point_goals + 2 * two_point_goals."
    )

    add_qc_check(
        "player_total_points_formula",
        "pass" if len(bad_player_points) == 0 else "warning",
        len(bad_player_points),
        0,
        "points should equal scoring_points + assists."
    )

    bad_player_scoring.to_csv(RUN_CHECK_DIR / "bad_player_scoring_points_formula_rows.csv", index=False)
    bad_player_points.to_csv(RUN_CHECK_DIR / "bad_player_total_points_formula_rows.csv", index=False)

# Team score vs manifest.
if len(game_manifest) > 0 and len(team_game_stats) > 0:
    team_scores_compare = team_game_stats.merge(
        game_manifest[["game_id", "home_team_id", "away_team_id", "home_score", "away_score"]],
        on="game_id",
        how="left"
    )

    team_scores_compare["manifest_score"] = np.where(
        team_scores_compare["team_id"] == team_scores_compare["home_team_id"],
        team_scores_compare["home_score"],
        np.where(
            team_scores_compare["team_id"] == team_scores_compare["away_team_id"],
            team_scores_compare["away_score"],
            np.nan
        )
    )

    team_scores_compare["manifest_score_diff"] = team_scores_compare["scores"] - team_scores_compare["manifest_score"]

    bad_manifest_score = team_scores_compare[
        team_scores_compare["manifest_score_diff"].notna() &
        (team_scores_compare["manifest_score_diff"].abs() > 0.001)
    ].copy()

    add_qc_check(
        "team_scores_match_game_manifest",
        "pass" if len(bad_manifest_score) == 0 else "warning",
        len(bad_manifest_score),
        0,
        "Team stats score should match home/away score in game manifest."
    )

    bad_manifest_score.to_csv(RUN_CHECK_DIR / "bad_manifest_score_match_rows.csv", index=False)

# Possession field quality.
if len(possession_field_quality) > 0:
    for _, r in possession_field_quality.iterrows():
        season = int(r["season"])
        total_nonzero = int(r["total_possessions_nonzero"])

        status = "info" if total_nonzero > 0 else "warning"

        add_qc_check(
            f"official_total_possessions_populated_{season}",
            status,
            total_nonzero,
            None,
            "Raw totalPossessions is only reliable in seasons where non-zero values exist."
        )

quality_summary = pd.DataFrame(quality_rows)
quality_summary.to_csv(RUN_CHECK_DIR / "quality_summary.csv", index=False)

display(quality_summary)

print("QC files saved to:", RUN_CHECK_DIR)

# ============================================================
# BLOCK 10.5 — DEFENSIVE / OPPONENT METRICS MARTS
# ============================================================

def safe_divide(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")

    return np.where(
        denominator.notna() & (denominator != 0),
        numerator / denominator,
        np.nan
    )


def build_team_game_opponent_context(team_df):
    """
    Builds one defensive/opponent-context row per team-game by self-joining
    each team row to the opposing team row from the same game.

    This table is the foundation for defensive rankings, matchup previews,
    opponent allowances, and team style/profile work.
    """
    if team_df is None or len(team_df) == 0:
        return pd.DataFrame()

    left = team_df.copy()
    right = team_df.copy()

    merged = left.merge(
        right,
        on="game_id",
        how="left",
        suffixes=("", "_opp_row")
    )

    merged = merged[
        merged["team_id"].astype(str) != merged["team_id_opp_row"].astype(str)
    ].copy()

    # Safety: one opponent row per team-game.
    merged = merged.drop_duplicates(subset=["game_id", "team_id"], keep="first").copy()

    ctx = pd.DataFrame()

    base_cols = [
        "season", "competition_type", "game_id", "schedule_slug", "game_slug",
        "game_number", "week", "game_date_utc", "team_id", "team_name",
        "opponent_team_id", "opponent_team_name", "is_home"
    ]

    for c in base_cols:
        ctx[c] = merged[c] if c in merged.columns else pd.NA

    # -----------------------------
    # Team/offensive values
    # -----------------------------
    team_value_map = {
        "team_scores": "scores",
        "team_goals": "goals",
        "team_one_point_goals": "one_point_goals",
        "team_two_point_goals": "two_point_goals",
        "team_assists": "assists",
        "team_shots": "shots",
        "team_shots_on_goal": "shots_on_goal",
        "team_ground_balls": "ground_balls",
        "team_turnovers": "turnovers",
        "caused_turnovers_for": "caused_turnovers",
        "team_faceoffs": "faceoffs",
        "team_faceoffs_won": "faceoffs_won",
        "team_faceoffs_lost": "faceoffs_lost",
        "saves_for": "saves",
        "team_clears": "clears",
        "team_clear_attempts": "clear_attempts",
        "team_touches": "touches",
        "team_total_passes": "total_passes",
        "team_time_in_possession": "time_in_possession",
        "team_offensive_sequence_proxy": "offensive_sequence_proxy",
    }

    for new_col, old_col in team_value_map.items():
        ctx[new_col] = pd.to_numeric(merged[old_col], errors="coerce") if old_col in merged.columns else np.nan

    # -----------------------------
    # Opponent/offense allowed values
    # -----------------------------
    opponent_value_map = {
        "scores_allowed": "scores_opp_row",
        "goals_allowed": "goals_opp_row",
        "one_point_goals_allowed": "one_point_goals_opp_row",
        "two_point_goals_allowed": "two_point_goals_opp_row",
        "assists_allowed": "assists_opp_row",
        "opponent_shots": "shots_opp_row",
        "opponent_shots_on_goal": "shots_on_goal_opp_row",
        "opponent_two_point_shots": "two_point_shots_opp_row",
        "opponent_two_point_shots_on_goal": "two_point_shots_on_goal_opp_row",
        "opponent_ground_balls": "ground_balls_opp_row",
        "opponent_turnovers": "turnovers_opp_row",
        "opponent_caused_turnovers": "caused_turnovers_opp_row",
        "opponent_faceoffs": "faceoffs_opp_row",
        "opponent_faceoffs_won": "faceoffs_won_opp_row",
        "opponent_faceoffs_lost": "faceoffs_lost_opp_row",
        "opponent_saves": "saves_opp_row",
        "opponent_clears": "clears_opp_row",
        "opponent_clear_attempts": "clear_attempts_opp_row",
        "opponent_touches": "touches_opp_row",
        "opponent_total_passes": "total_passes_opp_row",
        "opponent_time_in_possession": "time_in_possession_opp_row",
        "opponent_offensive_sequence_proxy": "offensive_sequence_proxy_opp_row",
    }

    for new_col, old_col in opponent_value_map.items():
        ctx[new_col] = pd.to_numeric(merged[old_col], errors="coerce") if old_col in merged.columns else np.nan

    # Explicit opponent checks used by QC.
    ctx["opponent_scores_check"] = ctx["scores_allowed"]
    ctx["opponent_goals_check"] = ctx["goals_allowed"]

    # -----------------------------
    # Game-level defensive metrics
    # -----------------------------
    ctx["score_margin"] = ctx["team_scores"] - ctx["scores_allowed"]
    ctx["win_flag"] = np.where(ctx["score_margin"] > 0, 1, np.where(ctx["score_margin"] < 0, 0, np.nan))
    ctx["loss_flag"] = np.where(ctx["score_margin"] < 0, 1, np.where(ctx["score_margin"] > 0, 0, np.nan))

    ctx["opponent_goal_pct"] = safe_divide(ctx["goals_allowed"], ctx["opponent_shots"])
    ctx["opponent_sog_rate"] = safe_divide(ctx["opponent_shots_on_goal"], ctx["opponent_shots"])
    ctx["opponent_sog_goal_pct"] = safe_divide(ctx["goals_allowed"], ctx["opponent_shots_on_goal"])

    # Save percentage proxy uses goals allowed, not PLL "scores" allowed, because 2PT goals count as one goal but two score units.
    ctx["save_pct_proxy"] = safe_divide(ctx["saves_for"], ctx["saves_for"] + ctx["goals_allowed"])

    ctx["ct_per_opponent_turnover"] = safe_divide(ctx["caused_turnovers_for"], ctx["opponent_turnovers"])
    ctx["opponent_scores_per_offensive_sequence_proxy"] = safe_divide(
        ctx["scores_allowed"],
        ctx["opponent_offensive_sequence_proxy"]
    )
    ctx["opponent_goals_per_shot"] = safe_divide(ctx["goals_allowed"], ctx["opponent_shots"])

    return ctx.sort_values(["season", "game_number", "team_name"]).reset_index(drop=True)


TEAM_DEFENSE_SUM_COLS = [
    "team_scores", "scores_allowed", "team_goals", "goals_allowed",
    "team_one_point_goals", "one_point_goals_allowed",
    "team_two_point_goals", "two_point_goals_allowed",
    "team_assists", "assists_allowed",
    "team_shots", "opponent_shots",
    "team_shots_on_goal", "opponent_shots_on_goal",
    "opponent_two_point_shots", "opponent_two_point_shots_on_goal",
    "team_ground_balls", "opponent_ground_balls",
    "team_turnovers", "opponent_turnovers",
    "caused_turnovers_for", "opponent_caused_turnovers",
    "team_faceoffs", "opponent_faceoffs",
    "team_faceoffs_won", "opponent_faceoffs_won",
    "team_faceoffs_lost", "opponent_faceoffs_lost",
    "saves_for", "opponent_saves",
    "team_clears", "opponent_clears",
    "team_clear_attempts", "opponent_clear_attempts",
    "team_touches", "opponent_touches",
    "team_total_passes", "opponent_total_passes",
    "team_time_in_possession", "opponent_time_in_possession",
    "team_offensive_sequence_proxy", "opponent_offensive_sequence_proxy",
    "score_margin",
]


def add_team_defense_rate_columns(df):
    out = df.copy()

    if "games" in out.columns:
        for c in TEAM_DEFENSE_SUM_COLS:
            if c in out.columns:
                out[f"{c}_per_game"] = safe_divide(out[c], out["games"])

    out["win_pct"] = safe_divide(out["wins"], out["games"]) if {"wins", "games"}.issubset(out.columns) else np.nan

    out["opponent_goal_pct"] = safe_divide(out["goals_allowed"], out["opponent_shots"])
    out["opponent_sog_rate"] = safe_divide(out["opponent_shots_on_goal"], out["opponent_shots"])
    out["opponent_sog_goal_pct"] = safe_divide(out["goals_allowed"], out["opponent_shots_on_goal"])
    out["save_pct_proxy"] = safe_divide(out["saves_for"], out["saves_for"] + out["goals_allowed"])
    out["ct_per_opponent_turnover"] = safe_divide(out["caused_turnovers_for"], out["opponent_turnovers"])
    out["opponent_scores_per_offensive_sequence_proxy"] = safe_divide(
        out["scores_allowed"],
        out["opponent_offensive_sequence_proxy"]
    )
    out["opponent_goals_per_shot"] = safe_divide(out["goals_allowed"], out["opponent_shots"])

    return out


def build_team_defense_agg(context_df, group_cols):
    if context_df is None or len(context_df) == 0:
        return pd.DataFrame()

    sum_cols = [c for c in TEAM_DEFENSE_SUM_COLS if c in context_df.columns]

    agg_dict = {
        "games": ("game_id", "nunique"),
        "first_game_date": ("game_date_utc", "min"),
        "last_game_date": ("game_date_utc", "max"),
        "wins": ("win_flag", "sum"),
        "losses": ("loss_flag", "sum"),
    }

    for c in sum_cols:
        agg_dict[c] = (c, "sum")

    out = (
        context_df
        .groupby(group_cols, dropna=False)
        .agg(**agg_dict)
        .reset_index()
    )

    out = add_team_defense_rate_columns(out)

    sort_cols = [c for c in ["season", "scores_allowed_per_game", "team_name"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[True, True, True][:len(sort_cols)]).reset_index(drop=True)

    return out


team_game_opponent_context = build_team_game_opponent_context(team_game_stats)

team_defense_season_stats = build_team_defense_agg(
    team_game_opponent_context,
    ["season", "team_id", "team_name"]
)

team_defense_career_stats = build_team_defense_agg(
    team_game_opponent_context,
    ["team_id", "team_name"]
)

# -----------------------------
# Defensive/opponent QC
# -----------------------------
def add_def_qc(check_name, status, actual=None, expected=None, notes=None):
    return {
        "check_name": check_name,
        "status": status,
        "actual": actual,
        "expected": expected,
        "notes": notes,
    }


defensive_qc_rows = []

expected_context_rows = len(team_game_stats)
actual_context_rows = len(team_game_opponent_context)

defensive_qc_rows.append(add_def_qc(
    "context_rows_match_team_game_rows",
    "pass" if actual_context_rows == expected_context_rows else "fail",
    actual_context_rows,
    expected_context_rows,
    "Opponent context should have one row per team-game."
))

defensive_qc_rows.append(add_def_qc(
    "season_defense_rows",
    "info",
    len(team_defense_season_stats),
    None,
    "One row per season/team with completed stat games."
))

defensive_qc_rows.append(add_def_qc(
    "career_defense_rows",
    "info",
    len(team_defense_career_stats),
    None,
    "One row per team across all completed stat games."
))

missing_opponent_join_rows = max(expected_context_rows - actual_context_rows, 0)

defensive_qc_rows.append(add_def_qc(
    "missing_opponent_join_rows",
    "pass" if missing_opponent_join_rows == 0 else "fail",
    missing_opponent_join_rows,
    0,
    "Every team-game row should find opponent row from same game."
))

duplicate_context_keys = (
    team_game_opponent_context.duplicated(subset=["game_id", "team_id"], keep=False).sum()
    if len(team_game_opponent_context) > 0 and {"game_id", "team_id"}.issubset(team_game_opponent_context.columns)
    else 0
)

defensive_qc_rows.append(add_def_qc(
    "duplicate_team_game_context_keys",
    "pass" if duplicate_context_keys == 0 else "fail",
    int(duplicate_context_keys),
    0,
    "No duplicate game_id/team_id rows in opponent context."
))

context_rows_per_game = (
    team_game_opponent_context.groupby("game_id").size().reset_index(name="rows")
    if len(team_game_opponent_context) > 0
    else pd.DataFrame(columns=["game_id", "rows"])
)

bad_context_game_count = int((context_rows_per_game["rows"] != 2).sum()) if len(context_rows_per_game) > 0 else 0

defensive_qc_rows.append(add_def_qc(
    "exactly_two_context_rows_per_game",
    "pass" if bad_context_game_count == 0 else "fail",
    bad_context_game_count,
    0,
    "Each completed/stat-available game should have two context rows."
))

if len(team_game_opponent_context) > 0:
    bad_scores_allowed = team_game_opponent_context[
        team_game_opponent_context["scores_allowed"].notna()
        & team_game_opponent_context["opponent_scores_check"].notna()
        & ((team_game_opponent_context["scores_allowed"] - team_game_opponent_context["opponent_scores_check"]).abs() > 0.001)
    ].copy()

    bad_goals_allowed = team_game_opponent_context[
        team_game_opponent_context["goals_allowed"].notna()
        & team_game_opponent_context["opponent_goals_check"].notna()
        & ((team_game_opponent_context["goals_allowed"] - team_game_opponent_context["opponent_goals_check"]).abs() > 0.001)
    ].copy()

else:
    bad_scores_allowed = pd.DataFrame()
    bad_goals_allowed = pd.DataFrame()

defensive_qc_rows.append(add_def_qc(
    "scores_allowed_matches_opponent_scores",
    "pass" if len(bad_scores_allowed) == 0 else "fail",
    len(bad_scores_allowed),
    0,
    "scores_allowed should equal opponent team scores."
))

defensive_qc_rows.append(add_def_qc(
    "goals_allowed_matches_opponent_goals",
    "pass" if len(bad_goals_allowed) == 0 else "fail",
    len(bad_goals_allowed),
    0,
    "goals_allowed should equal opponent team goals."
))

season_coverage_created = (
    team_defense_season_stats["season"].nunique()
    if len(team_defense_season_stats) > 0 and "season" in team_defense_season_stats.columns
    else 0
)

defensive_qc_rows.append(add_def_qc(
    "season_coverage_created",
    "info",
    int(season_coverage_created),
    None,
    "Number of seasons with defensive/opponent data."
))

defensive_opponent_build_quality = pd.DataFrame(defensive_qc_rows)

# Save early QC copies to run directory.
team_game_opponent_context.to_csv(RUN_CHECK_DIR / "team_game_opponent_context_preview.csv", index=False)
team_defense_season_stats.to_csv(RUN_CHECK_DIR / "team_defense_season_stats_preview.csv", index=False)
team_defense_career_stats.to_csv(RUN_CHECK_DIR / "team_defense_career_stats_preview.csv", index=False)
defensive_opponent_build_quality.to_csv(RUN_CHECK_DIR / "defensive_opponent_build_quality.csv", index=False)

# Append defensive QC into existing quality_summary so Data Quality shows it with the rest.
if "quality_summary" in globals() and isinstance(quality_summary, pd.DataFrame):
    existing_checks = set(quality_summary["check_name"].astype(str)) if "check_name" in quality_summary.columns else set()
    add_rows = defensive_opponent_build_quality[
        ~defensive_opponent_build_quality["check_name"].astype(str).isin(existing_checks)
    ].copy()

    quality_summary = pd.concat([quality_summary, add_rows], ignore_index=True)
    quality_summary.to_csv(RUN_CHECK_DIR / "quality_summary.csv", index=False)

print("Defensive/opponent marts created.")
print("team_game_opponent_context:", team_game_opponent_context.shape)
print("team_defense_season_stats:", team_defense_season_stats.shape)
print("team_defense_career_stats:", team_defense_career_stats.shape)

print("\nDefensive/opponent QC:")
display(defensive_opponent_build_quality)

print("\nBest defensive seasons by Scores Allowed/G:")
display(
    team_defense_season_stats[
        [
            c for c in [
                "season", "team_name", "games", "scores_allowed_per_game",
                "goals_allowed_per_game", "opponent_shots_per_game",
                "opponent_goal_pct", "save_pct_proxy", "caused_turnovers_for_per_game"
            ] if c in team_defense_season_stats.columns
        ]
    ]
    .sort_values(["scores_allowed_per_game", "opponent_goal_pct"], ascending=[True, True])
    .head(25)
)

print("\nRecent team-game opponent context:")
display(
    team_game_opponent_context[
        [
            c for c in [
                "season", "game_number", "game_date_utc", "team_name", "opponent_team_name",
                "team_scores", "scores_allowed", "opponent_shots",
                "opponent_shots_on_goal", "opponent_turnovers",
                "caused_turnovers_for", "save_pct_proxy"
            ] if c in team_game_opponent_context.columns
        ]
    ]
    .sort_values(["season", "game_number"], ascending=[False, False])
    .head(30)
)


# ============================================================
# GITHUB PORT ADD-ON — PLAYER RANKING + TEAM STYLE MARTS
# Mirrors the tested Streamlit pages' expected marts/columns.
# ============================================================

def _rank_pct(series, higher_is_better=True):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    return s.rank(pct=True, ascending=not higher_is_better) * 100


def _minmax_score(series, higher_is_better=True):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    lo = s.min(skipna=True)
    hi = s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or np.isclose(lo, hi):
        return pd.Series(50.0, index=series.index).where(s.notna(), np.nan)
    score = (s - lo) / (hi - lo) * 100
    if not higher_is_better:
        score = 100 - score
    return score.clip(0, 100)


def _robust_z(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(0.0, index=series.index).where(s.notna(), np.nan)
    med = s.median(skipna=True)
    mad = (s - med).abs().median(skipna=True)
    if pd.isna(mad) or np.isclose(mad, 0):
        std = s.std(skipna=True)
        if pd.isna(std) or np.isclose(std, 0):
            return pd.Series(0.0, index=series.index).where(s.notna(), np.nan)
        return ((s - s.mean(skipna=True)) / std).clip(-4, 4)
    return (0.6745 * (s - med) / mad).clip(-4, 4)


def _z_to_score(z):
    return (50 + 12.5 * pd.to_numeric(z, errors="coerce")).clip(0, 100)


def _value_tier_from_z(z):
    try:
        z = float(z)
    except Exception:
        return "Unrated"
    if pd.isna(z):
        return "Unrated"
    if z >= 2.0:
        return "Elite Separator"
    if z >= 1.25:
        return "High-End Separator"
    if z >= 0.50:
        return "Above-Average"
    if z >= -0.50:
        return "Average Range"
    if z >= -1.25:
        return "Below-Average"
    return "Low-End"


def _label_from_score(score, labels):
    try:
        s = float(score)
    except Exception:
        return "Unrated"
    if pd.isna(s):
        return "Unrated"
    if s >= 80:
        return labels[0]
    if s >= 65:
        return labels[1]
    if s >= 45:
        return labels[2]
    if s >= 30:
        return labels[3]
    return labels[4]


def _ensure_cols(df, cols):
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _ranking_context_min_games(context_type, max_games):
    if context_type in {"Last 5", "Last 10"}:
        return 1
    if context_type == "Career":
        return 5 if max_games >= 5 else 1
    if max_games <= 2:
        return 1
    if max_games <= 5:
        return 2
    return 3


def _add_player_ranking_scores(df):
    out = df.copy()

    needed = [
        "points_per_game", "scoring_points_per_game", "one_point_goals_per_game",
        "two_point_goals_per_game", "goals_per_game", "assists_per_game",
        "shots_per_game", "ground_balls_per_game", "turnovers_per_game",
        "caused_turnovers_per_game", "touches_per_game", "faceoff_pct_calc",
        "faceoffs_per_game", "faceoffs_won_per_game", "saves_per_game",
        "goals_against_per_game", "scores_against_per_game", "save_pct_calc",
        "games"
    ]
    out = _ensure_cols(out, needed)

    pos = out.get("position", pd.Series("", index=out.index)).astype(str).str.upper()
    out["role_group"] = np.select(
        [pos.eq("G"), pos.eq("FO"), pos.isin(["D", "LSM", "SSDM"])],
        ["Goalie", "Faceoff", "Defense"],
        default="Offense"
    )

    out["goal_value_score"] = (
        0.40 * _minmax_score(out["scoring_points_per_game"], True).fillna(50)
        + 0.20 * _minmax_score(out["one_point_goals_per_game"], True).fillna(50)
        + 0.25 * _minmax_score(out["two_point_goals_per_game"], True).fillna(50)
        + 0.15 * _minmax_score(out["goals_per_game"], True).fillna(50)
    ).clip(0, 100)

    out["offensive_score"] = (
        0.32 * _minmax_score(out["points_per_game"], True).fillna(50)
        + 0.22 * _minmax_score(out["scoring_points_per_game"], True).fillna(50)
        + 0.18 * _minmax_score(out["assists_per_game"], True).fillna(50)
        + 0.14 * _minmax_score(out["shots_per_game"], True).fillna(50)
        + 0.14 * out["goal_value_score"].fillna(50)
    ).clip(0, 100)

    out["usage_score"] = (
        0.55 * _minmax_score(out["touches_per_game"], True).fillna(50)
        + 0.25 * _minmax_score(out["shots_per_game"], True).fillna(50)
        + 0.20 * _minmax_score(out["ground_balls_per_game"], True).fillna(50)
    ).clip(0, 100)

    out["defensive_score"] = (
        0.45 * _minmax_score(out["caused_turnovers_per_game"], True).fillna(50)
        + 0.35 * _minmax_score(out["ground_balls_per_game"], True).fillna(50)
        + 0.20 * _minmax_score(out["turnovers_per_game"], False).fillna(50)
    ).clip(0, 100)

    out["faceoff_score"] = (
        0.55 * _minmax_score(out["faceoff_pct_calc"], True).fillna(50)
        + 0.25 * _minmax_score(out["faceoffs_won_per_game"], True).fillna(50)
        + 0.20 * _minmax_score(out["ground_balls_per_game"], True).fillna(50)
    ).clip(0, 100)

    out["goalie_score"] = (
        0.50 * _minmax_score(out["save_pct_calc"], True).fillna(50)
        + 0.25 * _minmax_score(out["saves_per_game"], True).fillna(50)
        + 0.25 * _minmax_score(out["goals_against_per_game"], False).fillna(50)
    ).clip(0, 100)

    out["role_primary_score"] = np.select(
        [
            out["role_group"].eq("Goalie"),
            out["role_group"].eq("Faceoff"),
            out["role_group"].eq("Defense"),
            out["role_group"].eq("Offense"),
        ],
        [
            out["goalie_score"],
            out["faceoff_score"],
            out["defensive_score"],
            out["offensive_score"],
        ],
        default=out["offensive_score"]
    )

    out["role_primary_percentile"] = np.nan
    out["role_robust_z"] = np.nan
    out["role_adjusted_z"] = np.nan
    out["role_separation_score"] = np.nan
    out["role_group_size"] = np.nan

    for _, idx in out.groupby("role_group", dropna=False).groups.items():
        idx = list(idx)
        role_scores = pd.to_numeric(out.loc[idx, "role_primary_score"], errors="coerce")
        z = _robust_z(role_scores)

        # Shrink peer-separation signal when group size is tiny.
        n = len(idx)
        shrink = min(1.0, np.sqrt(max(n, 1) / 8.0))
        z_adj = z * shrink

        out.loc[idx, "role_primary_percentile"] = _rank_pct(role_scores, True).values
        out.loc[idx, "role_robust_z"] = z.values
        out.loc[idx, "role_adjusted_z"] = z_adj.values
        out.loc[idx, "role_separation_score"] = _z_to_score(z_adj).values
        out.loc[idx, "role_group_size"] = n

    out["role_value_tier"] = out["role_adjusted_z"].apply(_value_tier_from_z)

    out["role_context_value_score"] = (
        0.50 * pd.to_numeric(out["role_primary_score"], errors="coerce").fillna(50)
        + 0.25 * pd.to_numeric(out["role_primary_percentile"], errors="coerce").fillna(50)
        + 0.25 * pd.to_numeric(out["role_separation_score"], errors="coerce").fillna(50)
    ).clip(0, 100)

    out["base_impact_score"] = (
        0.42 * out["offensive_score"].fillna(50)
        + 0.18 * out["usage_score"].fillna(50)
        + 0.16 * out["defensive_score"].fillna(50)
        + 0.12 * out["faceoff_score"].fillna(50)
        + 0.12 * out["goalie_score"].fillna(50)
    ).clip(0, 100)

    out["v22_overall_score"] = np.select(
        [
            out["role_group"].eq("Offense"),
            out["role_group"].eq("Defense"),
            out["role_group"].eq("Faceoff"),
            out["role_group"].eq("Goalie"),
        ],
        [
            0.62 * out["base_impact_score"] + 0.20 * out["role_context_value_score"] + 0.10 * out["usage_score"] + 0.08 * out["goal_value_score"],
            0.60 * out["base_impact_score"] + 0.30 * out["role_context_value_score"] + 0.10 * out["usage_score"],
            0.65 * out["base_impact_score"] + 0.25 * out["role_context_value_score"] + 0.10 * _minmax_score(out["ground_balls_per_game"], True).fillna(50),
            0.62 * out["base_impact_score"] + 0.38 * out["role_context_value_score"],
        ],
        default=out["base_impact_score"]
    )

    out["v22_overall_score"] = pd.to_numeric(out["v22_overall_score"], errors="coerce").clip(0, 100)
    out["overall_score"] = out["v22_overall_score"]
    out["overall_impact_score"] = out["v22_overall_score"]
    out["usage_possession_score"] = out["usage_score"]
    out["role_context_percentile"] = out["role_primary_percentile"]

    return out


def _build_player_ranking_context(df, context_type, context_label, sort_order):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    out = df.copy()
    if "games" not in out.columns:
        out["games"] = 0

    max_games = int(pd.to_numeric(out["games"], errors="coerce").max()) if len(out) else 0
    min_games = _ranking_context_min_games(context_type, max_games)

    out["ranking_context_type"] = context_type
    out["ranking_context"] = context_label
    out["ranking_sort_order"] = sort_order
    out["ranking_context_sort"] = sort_order
    out["max_games_in_context"] = max_games
    out["default_min_games_used"] = min_games
    out["is_ranking_eligible"] = (pd.to_numeric(out["games"], errors="coerce").fillna(0) >= min_games).astype(int)
    out["sample_size_note"] = np.where(max_games <= 2, "Early-season sample.", "")

    out = _add_player_ranking_scores(out)

    eligible = out["is_ranking_eligible"].eq(1)

    out["v22_overall_rank"] = np.nan
    out["v22_position_rank"] = np.nan
    out["offensive_rank"] = np.nan
    out["defensive_rank"] = np.nan
    out["faceoff_rank"] = np.nan
    out["goalie_rank"] = np.nan

    if eligible.any():
        out.loc[eligible, "v22_overall_rank"] = out.loc[eligible, "v22_overall_score"].rank(method="min", ascending=False)
        out.loc[eligible, "v22_overall_percentile"] = _rank_pct(out.loc[eligible, "v22_overall_score"], True).values

        for _, idx in out.loc[eligible].groupby("position", dropna=False).groups.items():
            idx = list(idx)
            out.loc[idx, "v22_position_rank"] = out.loc[idx, "v22_overall_score"].rank(method="min", ascending=False)
            out.loc[idx, "v22_position_percentile"] = _rank_pct(out.loc[idx, "v22_overall_score"], True).values

        out.loc[eligible, "offensive_rank"] = out.loc[eligible, "offensive_score"].rank(method="min", ascending=False)

        mask = eligible & out["role_group"].eq("Defense")
        if mask.any():
            out.loc[mask, "defensive_rank"] = out.loc[mask, "defensive_score"].rank(method="min", ascending=False)

        mask = eligible & out["role_group"].eq("Faceoff")
        if mask.any():
            out.loc[mask, "faceoff_rank"] = out.loc[mask, "faceoff_score"].rank(method="min", ascending=False)

        mask = eligible & out["role_group"].eq("Goalie")
        if mask.any():
            out.loc[mask, "goalie_rank"] = out.loc[mask, "goalie_score"].rank(method="min", ascending=False)

    out["overall_rank"] = out["v22_overall_rank"]
    out["overall_percentile"] = out.get("v22_overall_percentile", np.nan)
    out["position_rank"] = out["v22_position_rank"]
    out["position_percentile"] = out.get("v22_position_percentile", np.nan)

    if "save_pct_calc" in out.columns:
        out["save_pct"] = out["save_pct_calc"]
    if "faceoff_pct_calc" in out.columns:
        out["faceoff_pct"] = out["faceoff_pct_calc"]

    score_cols = [c for c in out.columns if c.endswith("_score") or c.endswith("_percentile") or c.endswith("_rank")]
    for c in score_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

    return out.sort_values(["ranking_sort_order", "v22_overall_rank"], na_position="last").reset_index(drop=True)


ranking_contexts = []
if "player_career_stats" in globals() and len(player_career_stats) > 0:
    ranking_contexts.append(_build_player_ranking_context(player_career_stats, "Career", "Career", 0))
if "player_last10_stats" in globals() and len(player_last10_stats) > 0:
    ranking_contexts.append(_build_player_ranking_context(player_last10_stats, "Last 10", "Last 10", 1))
if "player_last5_stats" in globals() and len(player_last5_stats) > 0:
    ranking_contexts.append(_build_player_ranking_context(player_last5_stats, "Last 5", "Last 5", 2))
if "player_season_stats" in globals() and len(player_season_stats) > 0 and "season" in player_season_stats.columns:
    for i, season in enumerate(sorted(pd.to_numeric(player_season_stats["season"], errors="coerce").dropna().astype(int).unique())):
        sdf = player_season_stats[pd.to_numeric(player_season_stats["season"], errors="coerce").eq(season)].copy()
        ranking_contexts.append(_build_player_ranking_context(sdf, "Season", f"{season} Season", 100 + i))

player_ranking_profiles = pd.concat(ranking_contexts, ignore_index=True, sort=False) if ranking_contexts else pd.DataFrame()


def _build_team_style_context(team_stats, defense_stats, context_type, context_label, sort_order):
    if team_stats is None or len(team_stats) == 0:
        return pd.DataFrame()

    teams = team_stats.copy()
    defense = defense_stats.copy() if defense_stats is not None else pd.DataFrame()

    if len(defense) > 0:
        merge_keys = ["team_id"]
        if "season" in teams.columns and "season" in defense.columns:
            merge_keys = ["season", "team_id"]

        keep_cols = [
            c for c in [
                *merge_keys,
                "scores_allowed_per_game",
                "goals_allowed_per_game",
                "opponent_shots_per_game",
                "def_opponent_shots_per_game",
                "opponent_goal_pct",
                "def_opponent_goal_pct",
                "opponent_sog_rate",
                "save_pct_proxy",
                "def_save_pct_proxy",
            ]
            if c in defense.columns
        ]

        if all(k in defense.columns for k in merge_keys):
            teams = teams.merge(defense[keep_cols].drop_duplicates(merge_keys), on=merge_keys, how="left", suffixes=("", "_def"))

    alias_pairs = {
        "scores_allowed_per_game": ["scores_allowed_per_game_def", "scores_against_per_game"],
        "goals_allowed_per_game": ["goals_allowed_per_game_def", "goals_against_per_game"],
        "opponent_shots_per_game": ["opponent_shots_per_game_def", "def_opponent_shots_per_game", "shots_against_per_game"],
        "opponent_goal_pct": ["opponent_goal_pct_def", "def_opponent_goal_pct"],
        "save_pct_proxy": ["save_pct_proxy_def", "def_save_pct_proxy"],
    }

    for primary, fallbacks in alias_pairs.items():
        if primary not in teams.columns:
            teams[primary] = np.nan
        for fallback in fallbacks:
            if fallback in teams.columns:
                teams[primary] = teams[primary].fillna(teams[fallback])

    numeric_cols = [
        "scores_per_game", "shots_per_game", "touches_per_game",
        "time_in_possession_per_game", "offensive_sequence_proxy_per_game",
        "turnovers_per_game", "assists_per_game", "total_passes_per_game",
        "scores_allowed_per_game", "opponent_shots_per_game", "opponent_goal_pct",
        "save_pct_proxy", "faceoff_pct_calc", "score_margin_per_game", "shot_pct_calc"
    ]
    teams = _ensure_cols(teams, numeric_cols)

    if teams["opponent_goal_pct"].isna().all() and "goals_against" in teams.columns and "shots_against" in teams.columns:
        teams["opponent_goal_pct"] = pd.to_numeric(teams["goals_against"], errors="coerce") / pd.to_numeric(teams["shots_against"], errors="coerce").replace(0, np.nan)

    if teams["save_pct_proxy"].isna().all() and "saves" in teams.columns and "goals_against" in teams.columns:
        saves = pd.to_numeric(teams["saves"], errors="coerce")
        ga = pd.to_numeric(teams["goals_against"], errors="coerce")
        teams["save_pct_proxy"] = (saves / (saves + ga).replace(0, np.nan)).clip(0, 1)

    teams["profile_context_type"] = context_type
    teams["profile_context"] = context_label
    teams["profile_sort_order"] = sort_order
    teams["profile_context_sort"] = sort_order

    teams["offensive_volume_score"] = (
        0.35 * _minmax_score(teams["scores_per_game"], True).fillna(50)
        + 0.25 * _minmax_score(teams["shots_per_game"], True).fillna(50)
        + 0.20 * _minmax_score(teams["touches_per_game"], True).fillna(50)
        + 0.20 * _minmax_score(teams["offensive_sequence_proxy_per_game"], True).fillna(50)
    ).clip(0, 100)

    teams["offensive_efficiency_score"] = (
        0.45 * _minmax_score(teams["scores_per_game"], True).fillna(50)
        + 0.25 * _minmax_score(teams["shot_pct_calc"], True).fillna(50)
        + 0.20 * _minmax_score(teams["turnovers_per_game"], False).fillna(50)
        + 0.10 * _minmax_score(teams["score_margin_per_game"], True).fillna(50)
    ).clip(0, 100)

    teams["ball_movement_score"] = (
        0.55 * _minmax_score(teams["assists_per_game"], True).fillna(50)
        + 0.25 * _minmax_score(teams["total_passes_per_game"], True).fillna(50)
        + 0.20 * _minmax_score(teams["touches_per_game"], True).fillna(50)
    ).clip(0, 100)

    teams["possession_control_score"] = (
        0.45 * _minmax_score(teams["touches_per_game"], True).fillna(50)
        + 0.35 * _minmax_score(teams["time_in_possession_per_game"], True).fillna(50)
        + 0.20 * _minmax_score(teams["faceoff_pct_calc"], True).fillna(50)
    ).clip(0, 100)

    teams["defensive_suppression_score"] = (
        0.40 * _minmax_score(teams["scores_allowed_per_game"], False).fillna(50)
        + 0.25 * _minmax_score(teams["opponent_shots_per_game"], False).fillna(50)
        + 0.20 * _minmax_score(teams["opponent_goal_pct"], False).fillna(50)
        + 0.15 * _minmax_score(teams["save_pct_proxy"], True).fillna(50)
    ).clip(0, 100)

    teams["pace_tempo_score"] = (
        0.35 * _minmax_score(teams["shots_per_game"], True).fillna(50)
        + 0.30 * _minmax_score(teams["touches_per_game"], True).fillna(50)
        + 0.20 * _minmax_score(teams["offensive_sequence_proxy_per_game"], True).fillna(50)
        + 0.15 * _minmax_score(teams["time_in_possession_per_game"], True).fillna(50)
    ).clip(0, 100)

    teams["team_style_overall_score"] = (
        0.22 * teams["offensive_volume_score"]
        + 0.20 * teams["offensive_efficiency_score"]
        + 0.16 * teams["ball_movement_score"]
        + 0.18 * teams["possession_control_score"]
        + 0.18 * teams["defensive_suppression_score"]
        + 0.06 * teams["pace_tempo_score"]
    ).clip(0, 100)

    teams["overall_score"] = teams["team_style_overall_score"]
    teams["overall_style"] = teams["team_style_overall_score"]
    teams["profile_rank"] = teams["team_style_overall_score"].rank(method="min", ascending=False)
    teams["def_scores_allowed_per_game"] = teams["scores_allowed_per_game"]
    teams["def_opponent_shots_per_game"] = teams["opponent_shots_per_game"]
    teams["def_opponent_goal_pct"] = teams["opponent_goal_pct"]
    teams["def_save_pct_proxy"] = teams["save_pct_proxy"]
    teams["net_scores_per_game"] = teams["scores_per_game"] - teams["scores_allowed_per_game"]
    teams["time_in_possession_per_game_mmss"] = teams["time_in_possession_per_game"].apply(seconds_to_mmss_safe)
    teams["possession_pg"] = teams["time_in_possession_per_game_mmss"]

    teams["pace_label"] = teams["pace_tempo_score"].apply(lambda x: _label_from_score(x, ("High Tempo", "Above-Average Tempo", "Balanced Tempo", "Slower Tempo", "Very Slow Tempo")))
    teams["offensive_profile_label"] = teams["offensive_efficiency_score"].apply(lambda x: _label_from_score(x, ("Elite Offense", "Above-Average Offense", "Middle Tier", "Low-Output Offense", "Poor Offense")))
    teams["defensive_profile_label"] = teams["defensive_suppression_score"].apply(lambda x: _label_from_score(x, ("Elite Defense", "Above-Average Defense", "Middle Tier", "Below-Average Defense", "Vulnerable Defense")))
    teams["possession_profile_label"] = teams["possession_control_score"].apply(lambda x: _label_from_score(x, ("Elite Possession", "Above-Average Possession", "Middle Tier", "Below-Average Possession", "Poor Possession")))
    teams["style_summary"] = teams["pace_label"].astype(str) + " | " + teams["offensive_profile_label"].astype(str) + " | " + teams["defensive_profile_label"].astype(str) + " | " + teams["possession_profile_label"].astype(str)
    teams["sample_size_note"] = np.where(pd.to_numeric(teams.get("games", pd.Series(0, index=teams.index)), errors="coerce").fillna(0) <= 2, "Early-season sample.", "")

    for c in [c for c in teams.columns if c.endswith("_score") or c in ["profile_rank"]]:
        teams[c] = pd.to_numeric(teams[c], errors="coerce").round(2)

    return teams.sort_values(["profile_sort_order", "profile_rank"], na_position="last").reset_index(drop=True)


style_contexts = []
if "team_career_stats" in globals() and len(team_career_stats) > 0:
    style_contexts.append(_build_team_style_context(team_career_stats, team_defense_career_stats if "team_defense_career_stats" in globals() else pd.DataFrame(), "Career", "Career", 0))
if "team_season_stats" in globals() and len(team_season_stats) > 0 and "season" in team_season_stats.columns:
    for i, season in enumerate(sorted(pd.to_numeric(team_season_stats["season"], errors="coerce").dropna().astype(int).unique())):
        sdf = team_season_stats[pd.to_numeric(team_season_stats["season"], errors="coerce").eq(season)].copy()
        if "team_defense_season_stats" in globals() and len(team_defense_season_stats) > 0 and "season" in team_defense_season_stats.columns:
            ddf = team_defense_season_stats[pd.to_numeric(team_defense_season_stats["season"], errors="coerce").eq(season)].copy()
        else:
            ddf = pd.DataFrame()
        style_contexts.append(_build_team_style_context(sdf, ddf, "Season", f"{season} Season", 100 + i))

team_style_profiles = pd.concat(style_contexts, ignore_index=True, sort=False) if style_contexts else pd.DataFrame()

print("Player ranking and team style profile marts created.")
print("player_ranking_profiles:", player_ranking_profiles.shape)
print("team_style_profiles:", team_style_profiles.shape)



# ============================================================
# GITHUB PORT ADD-ON — PARQUET-SAFE STORAGE
# ============================================================

def _make_unique_storage_columns(columns):
    seen = {}
    clean_cols = []
    for col in columns:
        base = str(col)
        if base not in seen:
            seen[base] = 0
            clean_cols.append(base)
        else:
            seen[base] += 1
            clean_cols.append(f"{base}_{seen[base] + 1}")
    return clean_cols


def sanitize_dataframe_for_storage(df):
    if df is None:
        return pd.DataFrame({"_empty_placeholder": pd.Series(dtype="string")})

    out = df.copy()

    if len(out.columns) == 0:
        return pd.DataFrame({"_empty_placeholder": pd.Series(dtype="string")})

    out.columns = _make_unique_storage_columns(list(out.columns))

    def storage_scalar(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, (dict, list, tuple, set)):
            try:
                return json.dumps(value, default=str, sort_keys=True)
            except Exception:
                return str(value)
        if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
            return value.isoformat()
        return value

    for col in out.columns:
        s = out[col]

        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_bool_dtype(s):
            out[col] = s.astype("boolean")
            continue
        if pd.api.types.is_integer_dtype(s):
            out[col] = s.astype("Int64")
            continue
        if pd.api.types.is_float_dtype(s):
            out[col] = pd.to_numeric(s, errors="coerce")
            continue

        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            mapped = s.map(storage_scalar)
            non_null = mapped.dropna()

            if len(non_null) == 0:
                out[col] = mapped.astype("string")
                continue

            numeric_types = (int, float, np.integer, np.floating)
            if non_null.map(lambda x: isinstance(x, numeric_types) and not isinstance(x, bool)).all():
                numeric = pd.to_numeric(mapped, errors="coerce")
                if len(numeric.dropna()) > 0 and numeric.dropna().map(lambda x: float(x).is_integer()).all():
                    out[col] = numeric.astype("Int64")
                else:
                    out[col] = numeric
                continue

            if non_null.map(lambda x: isinstance(x, (bool, np.bool_))).all():
                out[col] = mapped.astype("boolean")
                continue

            out[col] = mapped.astype("string")
            continue

        try:
            out[col] = s.map(storage_scalar).astype("string")
        except Exception:
            out[col] = s.astype("string")

    return out


# ============================================================
# BLOCK 11 — SAVE CURATED TABLES + DUCKDB WAREHOUSE
# DEFENSIVE / OPPONENT + POSSESSION QC INCLUDED
# ============================================================

def ensure_non_empty_schema(df, table_name):
    """
    DuckDB cannot read Parquet files with zero columns.
    This guarantees every exported table has at least one column.
    """
    if df is None:
        return pd.DataFrame({
            "_empty_table_name": [table_name],
            "_note": ["table_was_none_or_not_created"]
        })

    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame({
            "_empty_table_name": [table_name],
            "_note": [f"not_a_dataframe__type={type(df)}"]
        })

    if len(df.columns) == 0:
        return pd.DataFrame(columns=[
            "_empty_table_name",
            "_note",
            "season",
            "game_id",
            "game_slug",
            "reason",
            "error",
        ])

    return df.copy()


def get_table_var(var_name, required=False):
    """
    Safely pulls a dataframe variable from notebook globals.
    If required=True, raises a clear error if missing.
    If required=False, creates an empty note dataframe if missing.
    """
    if var_name in globals():
        return globals()[var_name]

    msg = f"Variable `{var_name}` was not found when saving curated tables."

    if required:
        raise NameError(msg)

    print("WARNING:", msg)

    return pd.DataFrame({
        "_empty_table_name": [var_name],
        "_note": ["variable_missing_when_block_11_ran"]
    })


# ============================================================
# CURATED TABLE REGISTRY
# ============================================================

curated_tables = {
    # ------------------------------------------------------------
    # Clean/base game-level data
    # ------------------------------------------------------------
    "game_manifest": get_table_var("game_manifest", required=True),
    "team_game_stats": get_table_var("team_game_stats", required=True),
    "player_game_stats": get_table_var("player_game_stats", required=True),

    # ------------------------------------------------------------
    # Reference/directories
    # ------------------------------------------------------------
    "team_alias_mapping": get_table_var("team_alias_mapping", required=True),
    "team_directory": get_table_var("team_directory", required=True),
    "player_directory": get_table_var("player_directory", required=True),

    # ------------------------------------------------------------
    # Player marts
    # ------------------------------------------------------------
    "player_season_stats_by_team": get_table_var("player_season_stats_by_team", required=True),
    "player_season_stats": get_table_var("player_season_stats", required=True),
    "player_career_stats": get_table_var("player_career_stats", required=True),
    "player_vs_opponent_stats": get_table_var("player_vs_opponent_stats", required=True),
    "player_last5_stats": get_table_var("player_last5_stats", required=True),
    "player_last10_stats": get_table_var("player_last10_stats", required=True),
    "player_season_last5_stats": get_table_var("player_season_last5_stats", required=True),
    "player_season_last10_stats": get_table_var("player_season_last10_stats", required=True),
    "player_ranking_profiles": get_table_var("player_ranking_profiles", required=False),

    # ------------------------------------------------------------
    # Team offensive / existing marts
    # ------------------------------------------------------------
    "team_season_stats": get_table_var("team_season_stats", required=True),
    "team_career_stats": get_table_var("team_career_stats", required=True),
    "team_vs_opponent_stats": get_table_var("team_vs_opponent_stats", required=True),
    "team_last5_stats": get_table_var("team_last5_stats", required=True),
    "team_last10_stats": get_table_var("team_last10_stats", required=True),
    "team_season_last5_stats": get_table_var("team_season_last5_stats", required=True),
    "team_season_last10_stats": get_table_var("team_season_last10_stats", required=True),
    "team_style_profiles": get_table_var("team_style_profiles", required=False),

    # ------------------------------------------------------------
    # Team defensive / opponent marts
    # Created in defensive/opponent build block
    # ------------------------------------------------------------
    "team_game_opponent_context": get_table_var("team_game_opponent_context", required=True),
    "team_defense_season_stats": get_table_var("team_defense_season_stats", required=True),
    "team_defense_career_stats": get_table_var("team_defense_career_stats", required=True),

    # ------------------------------------------------------------
    # Possession marts / QC
    # Created in updated Block 6
    # ------------------------------------------------------------
    "team_game_possession_quality": get_table_var("team_game_possession_quality", required=False),
    "game_possession_quality": get_table_var("game_possession_quality", required=False),
    "possession_field_quality": get_table_var("possession_field_quality", required=False),

    # ------------------------------------------------------------
    # Schedule/discovery
    # ------------------------------------------------------------
    "season_schedule_inventory": get_table_var("season_schedule_inventory", required=True),
    "stat_slug_inventory": get_table_var("stat_slug_inventory", required=True),
    "game_schedule_all": get_table_var("game_schedule_all", required=True),
    "game_schedule_2026": get_table_var("game_schedule_2026", required=True),

    # ------------------------------------------------------------
    # QC/logs
    # ------------------------------------------------------------
    "event_list_probe_summary": get_table_var("event_list_probe_summary", required=False),
    "game_discovery_log": get_table_var("game_discovery_log", required=False),
    "season_slug_inventory": get_table_var("season_slug_inventory", required=False),
    "api_collection_log": get_table_var("api_collection_log", required=False),
    "quality_summary": get_table_var("quality_summary", required=False),
    "defensive_opponent_build_quality": get_table_var("defensive_opponent_build_quality", required=False),
    "skipped_games": get_table_var("skipped_games", required=False),
}


# ============================================================
# SAVE CURATED TABLES TO PARQUET + CSV
# ============================================================

artifact_rows = []

for name, df in curated_tables.items():
    df_safe = sanitize_dataframe_for_storage(ensure_non_empty_schema(df, name))

    parquet_path = CURATED_ALL_DIR / f"{name}.parquet"
    csv_path = CURATED_ALL_DIR / f"{name}.csv"

    df_safe.to_parquet(parquet_path, index=False)
    df_safe.to_csv(csv_path, index=False)

    artifact_rows.append({
        "table_name": name,
        "rows": len(df_safe),
        "columns": len(df_safe.columns),
        "parquet_path": str(parquet_path),
        "csv_path": str(csv_path),
    })

artifact_index = (
    pd.DataFrame(artifact_rows)
    .sort_values("table_name")
    .reset_index(drop=True)
)

artifact_index.to_csv(CURATED_ALL_DIR / "artifact_index.csv", index=False)

print("Saved curated table artifacts:")
display(artifact_index)


# ============================================================
# DUCKDB WAREHOUSE BUILD
# ============================================================

DB_PATH = ANALYTICS_DATABASE_DIR / "pll_warehouse.duckdb"

# Close any existing notebook connection named con if it exists.
try:
    con.close()
except Exception:
    pass

con = duckdb.connect(str(DB_PATH))

con.execute("CREATE SCHEMA IF NOT EXISTS clean;")
con.execute("CREATE SCHEMA IF NOT EXISTS marts;")
con.execute("CREATE SCHEMA IF NOT EXISTS qc;")


# ------------------------------------------------------------
# Clean schema tables
# ------------------------------------------------------------

clean_table_names = [
    "game_manifest",
    "team_game_stats",
    "player_game_stats",
    "team_alias_mapping",
    "team_directory",
    "player_directory",
    "game_schedule_all",
    "game_schedule_2026",
]


# ------------------------------------------------------------
# Marts schema tables
# ------------------------------------------------------------

mart_table_names = [
    # Player marts
    "player_season_stats_by_team",
    "player_season_stats",
    "player_career_stats",
    "player_vs_opponent_stats",
    "player_last5_stats",
    "player_last10_stats",
    "player_season_last5_stats",
    "player_season_last10_stats",
    "player_ranking_profiles",

    # Team offensive / existing marts
    "team_season_stats",
    "team_career_stats",
    "team_vs_opponent_stats",
    "team_last5_stats",
    "team_last10_stats",
    "team_season_last5_stats",
    "team_season_last10_stats",
    "team_style_profiles",

    # Possession marts
    "team_game_possession_quality",

    # Defensive / opponent marts
    "team_game_opponent_context",
    "team_defense_season_stats",
    "team_defense_career_stats",
]


# ------------------------------------------------------------
# QC schema tables
# ------------------------------------------------------------

qc_table_names = [
    "season_schedule_inventory",
    "stat_slug_inventory",
    "event_list_probe_summary",
    "game_discovery_log",
    "season_slug_inventory",
    "api_collection_log",
    "possession_field_quality",
    "game_possession_quality",
    "quality_summary",
    "defensive_opponent_build_quality",
    "skipped_games",
]


def duckdb_load_parquet(con, schema_name, table_name):
    fp = CURATED_ALL_DIR / f"{table_name}.parquet"

    if not fp.exists():
        print(f"Skipping missing file: {fp}")
        return False

    try:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {schema_name}.{table_name} AS
            SELECT *
            FROM read_parquet(?);
            """,
            [str(fp)]
        )
        return True

    except Exception as e:
        print(f"FAILED loading {schema_name}.{table_name}: {e}")
        return False


load_rows = []

for table_name in clean_table_names:
    loaded = duckdb_load_parquet(con, "clean", table_name)
    load_rows.append({
        "schema": "clean",
        "table_name": table_name,
        "loaded": loaded,
    })

for table_name in mart_table_names:
    loaded = duckdb_load_parquet(con, "marts", table_name)
    load_rows.append({
        "schema": "marts",
        "table_name": table_name,
        "loaded": loaded,
    })

for table_name in qc_table_names:
    loaded = duckdb_load_parquet(con, "qc", table_name)
    load_rows.append({
        "schema": "qc",
        "table_name": table_name,
        "loaded": loaded,
    })

duckdb_load_summary = pd.DataFrame(load_rows)


# ============================================================
# WAREHOUSE INDEX + BASIC VALIDATION
# ============================================================

warehouse_tables = con.execute("""
SELECT
    table_schema,
    table_name
FROM information_schema.tables
WHERE table_schema IN ('clean', 'marts', 'qc')
ORDER BY table_schema, table_name
""").df()

warehouse_tables.to_csv(CURATED_ALL_DIR / "duckdb_table_index.csv", index=False)
duckdb_load_summary.to_csv(CURATED_ALL_DIR / "duckdb_load_summary.csv", index=False)


print("DuckDB load summary:")
display(duckdb_load_summary)

print("DuckDB warehouse tables:")
display(warehouse_tables)


# ------------------------------------------------------------
# Quick row-count validation
# ------------------------------------------------------------

validation_queries = {
    "clean.game_manifest": "SELECT COUNT(*) AS rows FROM clean.game_manifest",
    "clean.team_game_stats": "SELECT COUNT(*) AS rows FROM clean.team_game_stats",
    "clean.player_game_stats": "SELECT COUNT(*) AS rows FROM clean.player_game_stats",
    "marts.team_season_stats": "SELECT COUNT(*) AS rows FROM marts.team_season_stats",
    "marts.team_defense_season_stats": "SELECT COUNT(*) AS rows FROM marts.team_defense_season_stats",
    "marts.team_game_opponent_context": "SELECT COUNT(*) AS rows FROM marts.team_game_opponent_context",
    "marts.team_game_possession_quality": "SELECT COUNT(*) AS rows FROM marts.team_game_possession_quality",
    "marts.player_ranking_profiles": "SELECT COUNT(*) AS rows FROM marts.player_ranking_profiles",
    "marts.team_style_profiles": "SELECT COUNT(*) AS rows FROM marts.team_style_profiles",
    "qc.game_possession_quality": "SELECT COUNT(*) AS rows FROM qc.game_possession_quality",
    "qc.defensive_opponent_build_quality": "SELECT COUNT(*) AS rows FROM qc.defensive_opponent_build_quality",
}

validation_rows = []

for label, sql in validation_queries.items():
    try:
        n = con.execute(sql).df()["rows"].iloc[0]
        validation_rows.append({
            "table": label,
            "rows": int(n),
            "status": "ok",
        })
    except Exception as e:
        validation_rows.append({
            "table": label,
            "rows": None,
            "status": f"error: {e}",
        })

warehouse_validation = pd.DataFrame(validation_rows)
warehouse_validation.to_csv(CURATED_ALL_DIR / "warehouse_validation_summary.csv", index=False)

print("Warehouse validation summary:")
display(warehouse_validation)


# ------------------------------------------------------------
# Useful final checks
# ------------------------------------------------------------

try:
    completed_games_by_season = con.execute("""
        SELECT
            season,
            COUNT(DISTINCT game_id) AS completed_stat_games
        FROM clean.game_manifest
        GROUP BY season
        ORDER BY season
    """).df()

    print("Completed stat games by season:")
    display(completed_games_by_season)

except Exception as e:
    print("Could not display completed games by season:", e)


try:
    defensive_check = con.execute("""
        SELECT
            season,
            COUNT(*) AS team_defense_rows,
            COUNT(DISTINCT team_id) AS teams
        FROM marts.team_defense_season_stats
        GROUP BY season
        ORDER BY season
    """).df()

    print("Defensive season rows by season:")
    display(defensive_check)

except Exception as e:
    print("Could not display defensive season rows:", e)


try:
    possession_check = con.execute("""
        SELECT
            possession_data_status,
            COUNT(*) AS games
        FROM qc.game_possession_quality
        GROUP BY possession_data_status
        ORDER BY games DESC
    """).df()

    print("Game possession QC status counts:")
    display(possession_check)

except Exception as e:
    print("Could not display possession QC status counts:", e)


con.close()

print("Curated tables saved to:", CURATED_ALL_DIR)
print("DuckDB warehouse saved to:", DB_PATH)
