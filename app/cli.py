"""Local runner. Network is disabled unless explicitly requested."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

from app.config import Settings
from app.contracts import ContractAdapter
from app.errors import ServiceError
from app.llm import LLMGateway
from app.network import DatasetClient
from app.service import RunService
from app.tokens import TokenCounter


async def run(args):
    cfg = Settings.from_env().model_copy(
        update={"dataset_path": args.dataset.resolve(), "allow_network": args.allow_network}
    )
    adapter = ContractAdapter(cfg.contract_profile, cfg.max_sections, cfg.max_requests)
    if args.allow_network:
        adapter.require_official()
    counter = TokenCounter(cfg.token_cache, cfg.token_encoding)
    # Dataset is local; this header is never used for outbound authorization.
    async with httpx.AsyncClient(follow_redirects=False, trust_env=False) as client:
        service = RunService(
            cfg, adapter, DatasetClient(cfg, client), LLMGateway(cfg, client, counter), counter
        )
        payload, report = await service.execute(args.code, "Bearer local-only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Run the Sain pipeline on a LOCAL dataset")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--code", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Explicitly permit proxy LLM calls; requires a verified contract profile",
    )
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except ServiceError as error:
        parser.exit(1, f"error: {error.code}\n")


if __name__ == "__main__":
    main()
