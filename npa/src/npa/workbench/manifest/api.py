"""API surface: a thin waiter over Runtime.invoke (FastAPI)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .runtime import Runtime


class InvokeRequest(BaseModel):
    tool: str
    version: str
    command: str
    inputs: dict[str, Any] = {}
    backend: str = "local"


def create_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="manifest-runtime")

    @app.post("/invoke")
    def invoke_endpoint(req: InvokeRequest):
        try:
            res = runtime.invoke(
                tool=req.tool,
                version=req.version,
                command=req.command,
                inputs=req.inputs,
                surface="api",
                backend=req.backend,
            )
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        return res.verification_record()

    @app.get("/tools")
    def list_tools():
        return {"tools": runtime.catalog.list()}

    return app
