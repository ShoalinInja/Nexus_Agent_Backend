"""
Universities endpoint — no auth required.

Returns all rows from public."Uni_table" with:
  place_id, university, city_id, city
"""
import logging

from fastapi import APIRouter, HTTPException

from app.core.database import get_supabase

router = APIRouter(prefix="/universities", tags=["universities"])
logger = logging.getLogger(__name__)


@router.get("")
async def get_universities():
    """
    Return all universities from Uni_table.
    No authentication required.
    """
    try:
        supabase = get_supabase()
        result = (
            supabase.table("Uni_table")
            .select("place_id, university, city_id, city")
            .execute()
        )
        rows = result.data or []
        print(f"[UNIVERSITIES] Fetched {len(rows)} entries from Uni_table")
        logger.info(f"[UNIVERSITIES] Returned {len(rows)} rows")
        return {"universities": rows, "count": len(rows)}

    except Exception as e:
        logger.error(f"[UNIVERSITIES] Supabase query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch universities: {str(e)}",
        )
