from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import api_router
from app.ai.model_manager import BACKEND_DIR
from app.database.init_db import initialize_database


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="Smart Parking Management System API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Accept", "Content-Type"],
)
app.include_router(api_router)
app.mount("/uploads", StaticFiles(directory=BACKEND_DIR / "uploads", check_dir=False), name="uploads")


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
