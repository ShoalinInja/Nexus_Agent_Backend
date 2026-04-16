from app.core.database import get_supabase
from app.schemas.requests import PropertyRecommendationRequest
from app.schemas.responses import (
    PropertyRecommendationResponse,
    PropertyResult,
    PropertyConfigResult,
)

MAX_RESULTS = 20


def get_property_recommendations(
    params: PropertyRecommendationRequest,
) -> PropertyRecommendationResponse:
    client = get_supabase()

    # -------------------------------------------------------------------------
    # STEP A — Fetch all matching properties then their configs
    # (Two separate queries: no FK declared so embedded select is unavailable)
    # -------------------------------------------------------------------------
    props_query = (
        client.table("Prop_table")
        .select("*")
        .eq("city", params.city)
        .eq("is_active", True)
    )
    if not params.include_soldout:
        props_query = props_query.eq("sold_out", False)

    raw_properties = props_query.execute().data or []

    # Build (property, filtered_configs, min_rent_pw) tuples
    candidates: list[tuple[dict, list[PropertyConfigResult], float]] = []

    for prop in raw_properties:
        configs_query = (
            client.table("prop_config_table")
            .select("*")
            .eq("property_id", prop["id"])
            .gte("lease_weeks", params.min_lease_weeks)
            .lte("rent_pw", params.max_rent_pw)
        )
        if not params.include_soldout:
            configs_query = configs_query.eq("is_soldout", False)

        raw_configs = configs_query.execute().data or []

        filtered_configs: list[PropertyConfigResult] = []
        for cfg in raw_configs:
            if cfg.get("lease_weeks") is None or cfg.get("rent_pw") is None:
                continue
            filtered_configs.append(
                PropertyConfigResult(
                    config_id=cfg["config_id"],
                    property_id=cfg["property_id"],
                    room_type=cfg.get("room_type") or "",
                    room_name=cfg.get("room_name") or "",
                    rent_pw=float(cfg["rent_pw"]),
                    lease_weeks=float(cfg["lease_weeks"]),
                    move_in=str(cfg.get("move_in") or ""),
                )
            )

        # STEP B — Skip properties with no matching configs; compute min rent
        if not filtered_configs:
            continue

        min_rent = min(c.rent_pw for c in filtered_configs)
        candidates.append((prop, filtered_configs, min_rent))

    # -------------------------------------------------------------------------
    # STEP C — Sort by cheapest first, cap at 20
    # -------------------------------------------------------------------------
    candidates.sort(key=lambda t: t[2])
    top = candidates[:MAX_RESULTS]

    # -------------------------------------------------------------------------
    # STEP D — Build final results with ALL matching configs per property
    # -------------------------------------------------------------------------
    results: list[PropertyResult] = [
        PropertyResult(
            id=prop["id"],
            property=prop.get("property") or "",
            manager=prop.get("manager") or "",
            city=prop.get("city") or "",
            country=prop.get("country") or "",
            configs=configs,
        )
        for prop, configs, _ in top
    ]

    # -------------------------------------------------------------------------
    # STEP E — Response
    # -------------------------------------------------------------------------
    return PropertyRecommendationResponse(
        city=params.city,
        total_properties=len(results),
        total_configs=sum(len(r.configs) for r in results),
        sort_basis="min_rent_pw_asc",
        results=results,
    )
