"""
Retrieval Agent — fetches property supply data from Supabase.

Calls get_property_suggestionsv2 with progressive fallback:
  Pass 1 — all filters
  Pass 2 — drop room_type
  Pass 3 — drop move_in
  Pass 4 — drop lease
"""
import logging
from datetime import datetime
from typing import Optional

from app.core.database import get_supabase

logger = logging.getLogger(__name__)

# Correct RPC function name in Supabase
_RPC_FUNCTION = "get_property_suggestionsv2"

# University short-code → canonical DB name
_UNIVERSITY_MAP = {
    "ucl":        "UCL",
    "imperial":   "Imperial College London",
    "kcl":        "King's College London",
    "oxford":     "University of Oxford",
    "cambridge":  "University of Cambridge",
    "manchester": "University of Manchester",
    "birmingham": "University of Birmingham",
    "leeds":      "University of Leeds",
    "sheffield":  "University of Sheffield",
    "edinburgh":  "University of Edinburgh",
    "bristol":    "University of Bristol",
    "nottingham": "University of Nottingham",
    "newcastle":  "Newcastle University",
    "cardiff":    "Cardiff University",
    "liverpool":  "University of Liverpool",
    "glasgow":    "University of Glasgow",
    "coventry":   "Coventry University",
    "dmu":        "De Montfort University",
}


def _parse_date(raw: str) -> Optional[str]:
    """
    Normalize any of these formats to YYYY-MM-DD for Supabase DATE params:
      DD-MM-YYYY  e.g. 09-09-2026
      DD/MM/YYYY  e.g. 09/09/2026
      YYYY-MM-DD  e.g. 2026-09-09  (already correct, pass through)
    Returns None if unparseable.
    """
    if not raw:
        return None
    raw = raw.strip()
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    logger.warning(f"[RETRIEVAL] Could not parse date: '{raw}'")
    return None


def _normalize_city(city: str) -> str:
    """Title-case the city so 'nottingham' → 'Nottingham'."""
    return city.strip().title() if city else ""


def _normalize_university(uni: str) -> str:
    """Map short codes ('nottingham') to full DB names ('University of Nottingham')."""
    if not uni:
        return ""
    key = uni.strip().lower()
    return _UNIVERSITY_MAP.get(key, uni.strip())


def _normalize_room_type(rt: str) -> str:
    """Upper-case room type and map UI labels to DB values."""
    if not rt:
        return ""
    rt = rt.strip().upper()
    aliases = {
        "PREMIUM_ENSUITE": "PREMIUM ENSUITE",
        "STANDARD":        "STANDARD ROOM",
        "SHARED":          "SHARED ROOM",
    }
    return aliases.get(rt, rt)


def _call_rpc(supabase, payload: dict) -> list[dict]:
    """Execute the RPC and return the data list (may be empty)."""
    logger.info(f"[RETRIEVAL DEBUG] → {_RPC_FUNCTION}({payload})")
    resp = supabase.rpc(_RPC_FUNCTION, payload).execute()
    rows = resp.data or []
    logger.info(f"[RETRIEVAL DEBUG] ← {len(rows)} rows")
    return rows


def _format_properties(properties: list[dict]) -> str:
    """Format property rows into the text block injected into the LLM system prompt."""
    lines = ["AVAILABLE PROPERTIES (ranked by match score):\n"]
    for i, p in enumerate(properties, 1):
        # v2 function returns prop_name (not property_name)
        name = p.get("prop_name") or p.get("property_name") or "N/A"
        lines.append(
            f"{i}. {name}\n"
            f"   Room: {p.get('room_type', 'N/A')} ({p.get('room_name', '')}) — "
            f"£{p.get('rent_pw', 'N/A')}/week | {p.get('lease_weeks', 'N/A')} weeks\n"
            f"   Move-in: {p.get('move_in', 'N/A')} | Score: {p.get('match_score', 'N/A')}/100\n"
            f"   Walk: {p.get('walk_time_mins', 'N/A')} min "
            f"({p.get('walk_dist_km', 'N/A')} km) | "
            f"Car: {p.get('car_time_mins', 'N/A')} min\n"
            f"   Amenities: {p.get('amenities', 'N/A')}\n"
            f"   Recon: {p.get('recon_conf', 'N/A')} | "
            f"Commission: {p.get('avg_commission', 'N/A')}\n"
        )
    return "\n".join(lines)


def fetch_properties(filters: dict) -> tuple[str, bool]:
    """
    Call get_property_suggestionsv2 with progressive filter relaxation.

    Fallback order (each step only runs if the previous returned 0 rows):
      Pass 1 — all filters (city, university, movein, lease, budget, room_type)
      Pass 2 — drop room_type
      Pass 3 — drop move_in
      Pass 4 — drop lease
      Final  — return "no results" message

    Returns:
        (property_data_text: str, data_fetched: bool)
    """
    supabase = get_supabase()

    # ── Normalize inputs ──────────────────────────────────────────────────────
    city       = _normalize_city(filters.get("city") or "")
    university = _normalize_university(filters.get("university") or "")
    budget     = float(filters.get("budget") or 300)
    lease      = float(filters.get("lease") or 44)
    room_type  = _normalize_room_type(filters.get("room_type") or "")
    move_in    = _parse_date(filters.get("intake") or "")

    logger.info(
        f"[RETRIEVAL] Normalized inputs — city='{city}' university='{university}' "
        f"budget={budget} lease={lease} room_type='{room_type}' move_in={move_in}"
    )

    if not city:
        logger.error("[RETRIEVAL] city is required but empty — aborting")
        return "No city provided. Cannot fetch properties.", False

    # ── Base payload (always included) ───────────────────────────────────────
    base = {
        "p_city":       city,
        "p_university": university,
        "p_min_budget": 0,
        "p_max_budget": budget,
    }

    # ── Pass 1: all filters ───────────────────────────────────────────────────
    try:
        payload = {
            **base,
            "p_movein":    move_in,
            "p_lease":     lease,
            "p_room_type": room_type or None,
        }
        props = _call_rpc(supabase, payload)

        # ── Pass 2: drop room_type ────────────────────────────────────────────
        if not props and room_type:
            logger.info("[RETRIEVAL] Pass 2 — relaxing room_type filter")
            payload = {**base, "p_movein": move_in, "p_lease": lease, "p_room_type": None}
            props = _call_rpc(supabase, payload)

        # ── Pass 3: drop move_in ──────────────────────────────────────────────
        if not props and move_in:
            logger.info("[RETRIEVAL] Pass 3 — relaxing move_in filter")
            payload = {**base, "p_movein": None, "p_lease": lease, "p_room_type": None}
            props = _call_rpc(supabase, payload)

        # ── Pass 4: drop lease ────────────────────────────────────────────────
        if not props:
            logger.info("[RETRIEVAL] Pass 4 — relaxing lease filter")
            payload = {**base, "p_movein": None, "p_lease": None, "p_room_type": None}
            props = _call_rpc(supabase, payload)

        if not props:
            logger.warning("[RETRIEVAL] All passes returned 0 results")
            return (
                "No properties found matching the criteria even after broadening the search. "
                "Inform the agent to try a different city or adjust the budget.",
                False,
            )

        text = _format_properties(props)
        logger.info(f"[RETRIEVAL] Returning {len(props)} properties ({len(text)} chars)")
        return text, True

    except Exception as e:
        logger.error(f"[RETRIEVAL] RPC call failed: {e}", exc_info=True)
        return "", False
