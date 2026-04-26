from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..guardrails.base import CheckContext
from ..observability.logger import log_request


router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    policy: str = "balanced"
    system: str | None = None
    context: str | None = None


class ChatResponse(BaseModel):
    reply: str
    blocked: bool = False
    block_reason: str | None = None
    request_id: str
    input_results: list[dict[str, Any]]
    output_results: list[dict[str, Any]]


class PolicySwitch(BaseModel):
    name: str


def _serialize_results(results) -> list[dict]:
    return [
        {
            "name": r.name,
            "severity": r.severity,
            "reason": r.reason,
            "latency_ms": round(r.latency_ms, 2),
        }
        for r in results
    ]


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    shield = request.app.state.shield

    if req.policy != shield.active_policy:
        try:
            shield.switch_policy(req.policy)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    request_id = str(uuid.uuid4())
    ctx = CheckContext(
        policy=req.policy,
        request_id=request_id,
        grounding=req.context,
    )

    input_result = await shield.input_pipeline.run(req.message, ctx)

    audit: dict[str, Any] = {
        "request_id": request_id,
        "policy": req.policy,
        "input_results": _serialize_results(input_result.results),
        "output_results": [],
        "llm_called": False,
        "final_decision": None,
    }

    if input_result.blocked:
        audit["final_decision"] = "block"
        log_request(audit)
        return ChatResponse(
            reply="",
            blocked=True,
            block_reason=input_result.block_reason,
            request_id=request_id,
            input_results=audit["input_results"],
            output_results=[],
        )

    reply = await shield.llm.complete(req.message, system=req.system)

    output_result = await shield.output_pipeline.run(reply, ctx)
    audit["llm_called"] = True
    audit["output_results"] = _serialize_results(output_result.results)

    if output_result.blocked:
        audit["final_decision"] = "block_output"
        log_request(audit)
        return ChatResponse(
            reply="",
            blocked=True,
            block_reason=output_result.block_reason,
            request_id=request_id,
            input_results=audit["input_results"],
            output_results=audit["output_results"],
        )

    audit["final_decision"] = "pass"
    log_request(audit)

    return ChatResponse(
        reply=reply,
        request_id=request_id,
        input_results=audit["input_results"],
        output_results=audit["output_results"],
    )


@router.put("/v1/policy")
async def set_policy(body: PolicySwitch, request: Request):
    shield = request.app.state.shield
    try:
        shield.switch_policy(body.name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"active": shield.active_policy}


@router.get("/v1/health")
async def health(request: Request):
    shield = request.app.state.shield
    return {"status": "ok", "policy": shield.active_policy}
