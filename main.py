import logging
import time

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from app.case4 import Config, ContractError, Service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("case4")
for noisy in ("httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case: StrictInt
    dataset_code: StrictStr = Field(min_length=1, max_length=128)


def create_app(config: Config | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    cfg = config or Config.from_env()
    app = FastAPI(title="SAIN case 4", docs_url=None, redoc_url=None)
    service = Service(cfg)
    if transport is not None:
        service.client = httpx.AsyncClient(transport=transport, follow_redirects=False, trust_env=False)
        service.gateway.client = service.client
    app.state.service = service

    @app.get("/")
    async def root():
        return {"team": "team22", "service": "case4", "status": "ok"}

    @app.get("/health")
    async def health(request: Request):
        current: Service = request.app.state.service
        return {
            "status": "ok",
            "llm_configured": cfg.llm_ready,
            "dataset_ready": current.index is not None,
            "tokens_spent_estimate": current.gateway.budget.used,
        }

    @app.post("/run")
    async def run(body: RunRequest, request: Request, authorization: str | None = Header(default=None)):
        if not authorization or not authorization.startswith("Bearer ") or not authorization[7:].strip():
            return JSONResponse({"error": "bearer_required"}, status_code=401)
        if body.case != 4:
            return JSONResponse({"error": "unsupported_case"}, status_code=422)
        current: Service = request.app.state.service
        started = time.monotonic()
        try:
            payload = await current.run(body.dataset_code, authorization)
        except ContractError as exc:
            logger.warning("run_failed code=%s seconds=%.1f", exc.code, time.monotonic() - started)
            return JSONResponse({"error": exc.code}, status_code=502)
        except Exception:  # never leak internal details or dataset content
            logger.exception("run_crashed")
            return JSONResponse({"error": "internal_error"}, status_code=500)
        logger.info(
            "run_complete answers=%d tokens=%d seconds=%.1f",
            len(payload.get("requests", [])),
            current.gateway.budget.used,
            time.monotonic() - started,
        )
        return JSONResponse(payload)

    return app


app = create_app()
