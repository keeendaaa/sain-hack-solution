import asyncio
from collections import Counter

from pydantic import ValidationError

from app.config import Settings
from app.errors import InvalidModelOutput, ServiceError
from app.llm import LLMGateway
from app.models import Group, Plan, RunMetrics
from app.prompts import planning_messages
from app.resources import RunResources
from app.retrieval import Retriever


class QueryPlanner:
    def __init__(self, settings: Settings, gateway: LLMGateway):
        self.settings, self.gateway = settings, gateway

    def decode(self, raw: dict, groups: tuple[Group, ...], index: Retriever) -> dict[str, Plan]:
        rows = raw.get("plans")
        if not isinstance(rows, list):
            return {}
        allowed = {g.key for g in groups}
        occurrences = Counter(
            r.get("group_key") for r in rows if isinstance(r, dict) and isinstance(r.get("group_key"), str)
        )
        plans = {}
        for row in rows:
            try:
                plan = Plan.model_validate(row)
            except ValidationError:
                continue
            if plan.group_key not in allowed or occurrences[plan.group_key] != 1:
                continue
            valid = [k for k in plan.candidate_sections if k in index.by_key or k in index.articles]
            plans[plan.group_key] = plan.model_copy(update={"candidate_sections": list(dict.fromkeys(valid))})
        return plans

    async def plan(
        self, groups: tuple[Group, ...], index: Retriever, resources: RunResources, metrics: RunMetrics
    ) -> dict[str, Plan]:
        if not groups or not self.settings.allow_network:
            metrics.planner_degraded += len(groups)
            return {}
        counter = self.gateway.counter
        packets = None
        for article_level in (False, True):
            catalog = index.catalog(article_level)
            for batches in ((groups,), (groups[: (len(groups) + 1) // 2], groups[(len(groups) + 1) // 2 :])):
                batches = tuple(batch for batch in batches if batch)
                candidates = []
                for batch in batches:
                    messages = planning_messages(batch, catalog)
                    output = min(self.settings.planner_output, max(1500, 1000 + len(batch) * 80))
                    size = counter.messages(messages)
                    if size > self.settings.planner_input:
                        break
                    candidates.append((batch, messages, output, size))
                if (
                    len(candidates) == len(batches)
                    and sum(c[2] + c[3] for c in candidates) <= self.settings.planner_budget
                ):
                    packets = candidates
                    break
            if packets:
                break
        if not packets:
            metrics.planner_degraded += len(groups)
            metrics.error("planner_catalog_does_not_fit")
            return {}

        repairable: set[str] = set()

        async def invoke(packet):
            batch, messages, output, _ = packet
            try:
                raw = await self.gateway.call(
                    messages,
                    resources,
                    stage="planner",
                    input_cap=self.settings.planner_input,
                    output_cap=output,
                    timeout=self.settings.planner_seconds,
                )
                decoded = self.decode(raw, batch, index)
                results.update(decoded)
                repairable.update(g.key for g in batch if g.key not in decoded)
                return decoded
            except InvalidModelOutput:
                repairable.update(g.key for g in batch)
                return {}
            except ServiceError:
                return {}

        results = {}
        try:
            async with asyncio.timeout(min(self.settings.planner_seconds, resources.remaining())):
                await asyncio.gather(*(invoke(p) for p in packets))
        except TimeoutError:
            metrics.error("planner_timeout")
        missing = tuple(g for g in groups if g.key not in results and g.key in repairable)
        # Partial repair is intentionally small and has its own 4k stage budget.
        if missing:
            repair_groups = missing[:2]
            messages = planning_messages(repair_groups, index.catalog(article_level=True))
            if counter.messages(messages) <= self.settings.planner_repair_input:
                try:
                    raw = await self.gateway.call(
                        messages,
                        resources,
                        stage="planner_repair",
                        input_cap=self.settings.planner_repair_input,
                        output_cap=self.settings.planner_repair_output,
                        timeout=self.settings.planner_repair_seconds,
                    )
                    results.update(self.decode(raw, repair_groups, index))
                except ServiceError:
                    pass
        metrics.planner_valid += len(results)
        metrics.planner_degraded += len(groups) - len(results)
        return results
