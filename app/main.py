import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.database import get_supabase
from app.api.routes import router
from app.api.v0.router import v0_router

from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: test Supabase connection
    try:
        get_supabase()
        logger.info("Supabase connection: OK")
    except Exception as e:
        logger.error(f"Supabase connection failed: {e}")
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# app.include_router(router)
app.include_router(v0_router, prefix="/v0/api")


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def home():
    return {"status": "server is running"}


@app.get("/health/db")
async def health_db():
    try:
        client = get_supabase()
        return {"supabase": "connected", "status": "ok"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"supabase": "error", "status": "failed", "detail": str(e)},
        )
