"""Local structural evaluation. Never posts to validate or contacts a judge."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT
from app.contracts import ContractAdapter, pointer
from app.models import identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=ROOT / "contracts/profiles/local_example.json")
    args = parser.parse_args()
    adapter = ContractAdapter(args.profile)
    dataset = adapter.decode(json.loads(args.dataset.read_text()))
    payload = json.loads(args.answers.read_text())
    p = adapter.profile
    schema_valid = bool(adapter.answers_validator and adapter.answers_validator.is_valid(payload))
    records = pointer(payload, p.output_answers)
    seen = set()
    refs_valid = True
    actual = {}
    corpus = {s.pair_key for s in dataset.sections}
    for record in records:
        key = identity(pointer(record, p.output_request_id))
        if key in seen:
            raise SystemExit("Duplicate answer ID")
        seen.add(key)
        refs = pointer(record, p.output_references)
        pairs = tuple(
            (identity(pointer(r, p.reference_article)), identity(pointer(r, p.reference_section)))
            for r in refs
        )
        refs_valid &= bool(pairs) and all(pair in corpus for pair in pairs)
        actual[key] = pairs
    from app.grouping import group_requests

    groups = group_requests(dataset)
    shared = all(len({actual.get(identity(r.request_id)) for r in g.records}) == 1 for g in groups)
    coverage = seen == {identity(r.request_id) for r in dataset.requests}
    report = {
        "local_schema_valid": schema_valid,
        "official_contract": p.verified_official,
        "answer_coverage": coverage,
        "references_exist": refs_valid,
        "shared_triplet_references": shared,
        "answer_count": len(records),
        "medical_safety_verified": False,
        "official_score": None,
    }
    print(json.dumps(report, indent=2))
    if not all((schema_valid, coverage, refs_valid, shared)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
