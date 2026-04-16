from fastapi import APIRouter

from app.schemas.enquiry_schemas import PropertyEnquiryRequest, PropertyEnquiryResponse
from app.services.property_enquiry_service import handle_property_enquiry

router = APIRouter()


@router.post("/property_enquiry", response_model=PropertyEnquiryResponse)
async def property_enquiry(req: PropertyEnquiryRequest):
    return await handle_property_enquiry(req)
