"""Local dev run: accepts the organizer getData envelope, never uses `reference`."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.case4 import Config, Service, decode_dataset


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--groups", type=int, default=0)
    parser.add_argument("--group-ids", default="")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    payload = json.loads(Path(args.dataset).read_text())
    if args.group_ids:
        payload = json.loads(json.dumps(payload))
        keep = {item.strip() for item in args.group_ids.split(",") if item.strip()}
        payload["dataset"]["requests"] = [r for r in payload["dataset"]["requests"] if r.get("group_id") in keep]
    if args.groups and not args.group_ids:
        payload = json.loads(json.dumps(payload))
        seen, keep = [], set()
        for request in payload["dataset"]["requests"]:
            group = request.get("group_id")
            if group not in seen:
                if len(seen) >= args.groups:
                    continue
                seen.append(group)
                keep.add(group)
        payload["dataset"]["requests"] = [r for r in payload["dataset"]["requests"] if r.get("group_id") in keep]
    dataset = decode_dataset(payload)
    config = Config.from_env()
    if args.no_llm:
        config = Config(**{**config.__dict__, "llm_api_key": ""})
    service = Service(config)
    try:
        result = await service.generate(dataset)
    finally:
        await service.close()
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(
        json.dumps(
            {
                "answers": len(result["requests"]),
                "refused": sum(1 for item in result["requests"] if item["refused"]),
                "tokens_estimate": service.gateway.budget.used,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
