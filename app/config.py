import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    allow_network: bool = False
    contract_profile: Path = ROOT / "contracts/profiles/local_example.json"
    dataset_path: Path | None = None
    hack_api_url: str = "https://hack-api.be-healthy.tech"
    llm_base_url: str = ""
    llm_api_key: str = Field(default="", repr=False)
    llm_model: str = ""
    token_encoding: str = "o200k_base"
    token_cache: Path = ROOT / "app/assets/tiktoken"
    run_seconds: float = Field(default=2520, gt=0, le=2520)
    fetch_seconds: float = Field(default=120, gt=0, le=120)
    generation_seconds: float = Field(default=60, gt=0, le=100)
    planner_seconds: float = Field(default=110, gt=0, le=110)
    planner_repair_seconds: float = Field(default=60, gt=0, le=60)
    token_budget: int = Field(default=145000, gt=0, le=145000)
    requests_per_minute: int = Field(default=24, gt=0, le=24)
    concurrency: int = Field(default=2, gt=0, le=2)
    generation_input: int = Field(default=2000, gt=0, le=2000)
    generation_output: int = Field(default=1600, ge=1100, le=1600)
    planner_input: int = Field(default=7000, gt=0, le=7000)
    planner_output: int = Field(default=4000, ge=1100, le=4000)
    planner_budget: int = Field(default=11000, gt=0, le=11000)
    planner_repair_input: int = Field(default=2500, gt=0, le=2500)
    planner_repair_output: int = Field(default=1500, ge=1100, le=1500)
    generation_extras: int = Field(default=3, ge=0, le=3)
    max_dataset_bytes: int = Field(default=32 * 1024 * 1024, gt=0)
    max_response_bytes: int = 64 * 1024 * 1024
    max_sections: int = 5000
    max_requests: int = 300
    cache_entries: int = 2
    cache_ttl_seconds: float = 3600
    cache_bytes: int = 64 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        values: dict = {}
        mapping = {
            "HACK_API_URL": "hack_api_url",
            "LLM_BASE_URL": "llm_base_url",
            "LLM_API_KEY": "llm_api_key",
            "LLM_MODEL": "llm_model",
            "SAIN_CONTRACT_PROFILE": "contract_profile",
            "SAIN_DATASET_PATH": "dataset_path",
            "SAIN_TOKEN_ENCODING": "token_encoding",
            "SAIN_TOKEN_CACHE": "token_cache",
            "SAIN_GENERATION_TIMEOUT": "generation_seconds",
        }
        for env, field in mapping.items():
            if os.environ.get(env):
                values[field] = os.environ[env]
        # Deliberately opt-in: importing/starting the application sends nothing.
        values["allow_network"] = os.environ.get("SAIN_ALLOW_NETWORK", "0") == "1"
        return cls(**values)
