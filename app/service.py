import asyncio
import resource
import sys
import time
from dataclasses import asdict

from pydantic import ValidationError

from app.config import ROOT, Settings
from app.contracts import ContractAdapter
from app.errors import ContractError, InvalidModelOutput, ResourceExhausted, ServiceError, TransientLLMError
from app.fallback import fallback_answers
from app.grouping import group_requests
from app.llm import LLMGateway
from app.models import Answer, Candidate, Dataset, Group, RunMetrics, canonical, digest
from app.network import DatasetClient
from app.planner import QueryPlanner
from app.policy import assess
from app.prompts import answer_messages
from app.resources import RunResources
from app.retrieval import Retriever
from app.tokens import TokenCounter
from app.validation import assert_group_invariants, validate_candidate


class RunService:
    def __init__(
        self,
        settings: Settings,
        adapter: ContractAdapter,
        loader: DatasetClient,
        gateway: LLMGateway,
        counter: TokenCounter,
    ):
        self.settings, self.adapter, self.loader = settings, adapter, loader
        self.gateway, self.counter = gateway, counter
        self._index_key = ""
        self._index: Retriever | None = None

    def index(self, dataset: Dataset) -> Retriever:
        key = digest([(s.pair_key, s.article_title, s.heading, s.text) for s in dataset.sections])
        if self._index is None or key != self._index_key:
            self._index = Retriever(dataset, self.counter, ROOT / "config/sleep_terms.json")
            self._index_key = key
        return self._index

    async def execute(self, code: str, authorization: str) -> tuple[bytes, dict]:
        started = time.monotonic()
        resources = RunResources(
            self.settings.run_seconds, self.settings.token_budget, self.settings.generation_extras
        )
        if self.settings.allow_network or not self.settings.dataset_path:
            self.adapter.require_official()
        envelope = await self.loader.fetch(code, authorization, resources)
        dataset = await asyncio.to_thread(self.adapter.decode, envelope)
        del envelope  # In particular, reference is never passed into the pipeline.
        if self.settings.dataset_path and str(dataset.dataset_id) != code:
            raise ContractError("local_dataset_code_mismatch")
        # CPU preparation off the event loop; readiness/health remains available.
        index = await asyncio.to_thread(self.index, dataset)
        groups = group_requests(dataset)
        metrics = RunMetrics(groups=len(groups))
        planner = QueryPlanner(self.settings, self.gateway)
        plans = await planner.plan(groups, index, resources, metrics)
        workers = asyncio.Semaphore(self.settings.concurrency)

        async def answer_group(group: Group) -> list[Answer]:
            # Bounded workers: do not create simultaneous prompts for every record.
            async with workers:
                risk = assess(group)
                evidence = await asyncio.to_thread(index.evidence, group, plans.get(group.key))
                known_refuse = False
                errors: tuple[str, ...] = ()
                if self.settings.allow_network and evidence:
                    for attempt in range(2):
                        extra = attempt == 1
                        if extra and resources.ledger.extras >= self.settings.generation_extras:
                            break
                        messages = answer_messages(group, evidence, risk, errors)
                        # Drop a complete lowest-priority fragment, never crop its meaning.
                        while (
                            len(evidence) > 1
                            and self.counter.messages(messages) > self.settings.generation_input
                        ):
                            evidence = evidence[:-1]
                            messages = answer_messages(group, evidence, risk, errors)
                        try:
                            raw = await self.gateway.call(
                                messages,
                                resources,
                                stage="generation",
                                input_cap=self.settings.generation_input,
                                output_cap=self.settings.generation_output,
                                timeout=self.settings.generation_seconds,
                                extra=extra,
                            )
                            known_refuse |= raw.get("refused") is True
                            try:
                                candidate = Candidate.model_validate(raw)
                            except ValidationError:
                                raise InvalidModelOutput("candidate_schema") from None
                            errors, result = validate_candidate(candidate, group, evidence, risk)
                            if not errors:
                                return result
                            for error in errors:
                                metrics.error(error)
                        except ResourceExhausted:
                            break
                        except TransientLLMError as exc:
                            metrics.error("transient_llm")
                            errors = ("regenerate_after_transient_failure",)
                            if not extra:
                                try:
                                    await resources.backoff(exc.retry_after)
                                except ResourceExhausted:
                                    break
                        except InvalidModelOutput as exc:
                            metrics.error(exc.code)
                            errors = (exc.code,)
                        except ServiceError as exc:
                            metrics.error(exc.code)
                            break  # Authentication/configuration/4xx are not repairable JSON.
                metrics.fallback_groups += 1
                return fallback_answers(group, evidence, risk, index.fallback_section, known_refuse)

        answers: list[Answer] = []
        tasks = {asyncio.create_task(answer_group(g)): g for g in groups}
        try:
            done, pending = (
                await asyncio.wait(tasks, timeout=resources.remaining()) if tasks else (set(), set())
            )
            for task in done:
                group = tasks[task]
                try:
                    answers.extend(task.result())
                except Exception:
                    # Isolate a single worker's unexpected failure. No raw error text is logged.
                    metrics.error("group_worker_failed")
                    metrics.fallback_groups += 1
                    answers.extend(fallback_answers(group, (), assess(group), index.fallback_section))
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            for task in pending:
                group = tasks[task]
                metrics.fallback_groups += 1
                answers.extend(fallback_answers(group, (), assess(group), index.fallback_section))
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        assert_group_invariants(groups, answers)
        body = self.adapter.encode(dataset, answers)
        payload = canonical(body).encode()
        if len(payload) > self.settings.max_response_bytes:
            raise ContractError("output_too_large")
        report = {
            "profile": self.adapter.profile.name,
            "official_contract": self.adapter.profile.verified_official,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "metrics": asdict(metrics),
            "llm": resources.ledger.report(),
            "output_bytes": len(payload),
            "answer_count": len(answers),
            "index_hash": index.hash,
            "network_allowed": self.settings.allow_network,
            "peak_rss_mib": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 * 1024 if sys.platform == "darwin" else 1024),
                2,
            ),
        }
        report["metrics"]["extra_generation_calls"] = resources.ledger.extras
        return payload, report
