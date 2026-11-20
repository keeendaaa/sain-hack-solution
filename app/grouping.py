import unicodedata
from collections import OrderedDict

from app.models import Dataset, Group, RequestRecord, identity


def group_requests(dataset: Dataset) -> tuple[Group, ...]:
    experts = {identity(e.expert_id): e for e in dataset.experts}
    buckets: dict[tuple, list[RequestRecord]] = OrderedDict()
    for record in dataset.requests:
        # Do not semantic-deduplicate, reorder dialogue or merge distinct users.
        text = unicodedata.normalize("NFC", record.text).strip()
        key = identity(record.user_id), text, record.context_json
        buckets.setdefault(key, []).append(record)
    result = []
    for index, records in enumerate(buckets.values()):
        keys = {identity(r.expert_id) for r in records}
        selected = tuple(e for eid, e in experts.items() if eid in keys)
        result.append(
            Group(f"G{index:04d}", records[0].text, records[0].context_json, tuple(records), selected)
        )
    return tuple(result)
