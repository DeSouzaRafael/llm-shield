from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from .routes import router
from .state import AppState


POLICIES_DIR = Path(__file__).parent.parent.parent / "policies"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.shield = AppState(policies_dir=POLICIES_DIR)
    yield


app = FastAPI(title="llm-shield", version="0.1.0", lifespan=lifespan)
app.include_router(router)
