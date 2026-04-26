from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

import src.guardrails.input  # noqa: F401 — register all input guardrails
import src.guardrails.output  # noqa: F401 — register all output guardrails

from .routes import router
from .state import AppState
from ..observability.metrics import start_metrics_server


POLICIES_DIR = Path(__file__).parent.parent.parent / "policies"


@asynccontextmanager
async def lifespan(app: FastAPI):
    metrics_port = int(os.environ.get("METRICS_PORT", 9090))
    start_metrics_server(metrics_port)

    policy = os.environ.get("DEFAULT_POLICY", "balanced")
    app.state.shield = AppState(policies_dir=POLICIES_DIR, default_policy=policy)
    yield


app = FastAPI(title="llm-shield", version="0.1.0", lifespan=lifespan)
app.include_router(router)
