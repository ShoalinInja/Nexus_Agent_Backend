from app.core.database import get_supabase


def fetch_city_data(city: str) -> dict:
    client = get_supabase()

    props_resp = (
        client.table("Prop_table")
        .select("*")
        .eq("city", city)
        .eq("is_active", True)
        .eq("sold_out", False)
        .execute()
    )
    properties = props_resp.data or []

    configs = []
    for prop in properties:
        configs_resp = (
            client.table("prop_config_table")
            .select("*")
            .eq("property_id", prop["id"])
            .eq("is_soldout", False)
            .execute()
        )
        configs.extend(configs_resp.data or [])

    return {"properties": properties, "configs": configs}
