import logging
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from app.config import Settings
from app.contracts import ContractAdapter
from app.errors import NotConfigured, ResourceExhausted, ServiceError
from app.llm import LLMGateway
from app.network import DatasetClient
from app.registry import RunRegistry
from app.service import RunService
from app.tokens import TokenCounter

logger = logging.getLogger("sain")
# HTTP clients otherwise expose dataset_code in INFO logs.
for name in ("httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.WARNING)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case: StrictInt
    dataset_code: StrictStr = Field(min_length=1)


def create_app(
    settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    cfg = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        registry = RunRegistry(cfg.cache_entries, cfg.cache_bytes, cfg.cache_ttl_seconds)
        application.state.registry = registry
        application.state.service = None
        application.state.config_error = None
        async with httpx.AsyncClient(transport=transport, follow_redirects=False, trust_env=False) as client:
            try:
                adapter = ContractAdapter(cfg.contract_profile, cfg.max_sections, cfg.max_requests)
                counter = TokenCounter(cfg.token_cache, cfg.token_encoding)
                application.state.service = RunService(
                    cfg, adapter, DatasetClient(cfg, client), LLMGateway(cfg, client, counter), counter
                )
            except ServiceError as exc:
                application.state.config_error = exc.code
            yield
            await registry.close()

    application = FastAPI(title="SAIN case 4", lifespan=lifespan, docs_url=None, redoc_url=None)

    @application.exception_handler(ServiceError)
    async def service_error(request: Request, error: ServiceError):
        # Never return exception messages: upstream bodies and tokens are untrusted.
        status = 409 if isinstance(error, ResourceExhausted) else error.status
        return JSONResponse({"error": error.code}, status_code=status)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        return JSONResponse({"error": "invalid_request"}, status_code=422)

    @application.get("/")
    async def root():
        return {"service": "sain-case4", "status": "ok"}

    @application.get("/health")
    async def health(request: Request):
        service = getattr(request.app.state, "service", None)
        official = bool(service and service.adapter.profile.verified_official)
        return {
            "status": "ok",
            "configured": service is not None,
            "official_contract": official,
            "network_enabled": cfg.allow_network,
            "mode": "local" if cfg.dataset_path else "stand",
            "ready_for_online": bool(
                service
                and official
                and cfg.allow_network
                and not cfg.dataset_path
                and cfg.llm_model
                and cfg.llm_api_key
                and cfg.llm_base_url
                and not service.gateway.authentication_failed
            ),
        }

    @application.post("/run")
    async def run(body: RunRequest, request: Request, authorization: str | None = Header(default=None)):
        if not authorization or not authorization.startswith("Bearer ") or not authorization[7:].strip():
            return JSONResponse({"error": "bearer_required"}, status_code=401)
        if body.case != 4:
            return JSONResponse({"error": "unsupported_case"}, status_code=422)
        service = request.app.state.service
        if service is None:
            raise NotConfigured()
        key = request.app.state.registry.key(body.dataset_code, authorization)
        run_id = uuid.uuid4().hex

        async def compute():
            payload, report = await service.execute(body.dataset_code, authorization)
            # Explicit safe allowlist, no dataset, auth or generated texts.
            logger.info(
                "run_complete id=%s groups=%d fallbacks=%d tokens=%d seconds=%.3f",
                run_id,
                report["metrics"]["groups"],
                report["metrics"]["fallback_groups"],
                report["llm"]["charged_estimate"],
                report["elapsed_seconds"],
            )
            return payload

        payload = await request.app.state.registry.run(key, compute)
        return Response(content=payload, media_type="application/json")

    return application


app = create_app()
