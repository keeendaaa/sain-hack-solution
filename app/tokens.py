"""Token counting with a bundled vocabulary, with no runtime download path.

o200k_base is an explicit encoding assumption, not a verified tokenizer for
the organizer's currently configured model. Provider usage must be calibrated.
"""

import hashlib
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe

from app.errors import NotConfigured

SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
PATTERN = "|".join(
    [
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)


class TokenCounter:
    def __init__(self, cache: Path, encoding: str = "o200k_base"):
        path = cache / "o200k_base.tiktoken"
        if encoding != "o200k_base" or not path.is_file():
            raise NotConfigured("bundled_tokenizer_required")
        if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
            raise NotConfigured("tokenizer_checksum_failed")
        self.encoding = tiktoken.Encoding(
            name="sain_o200k_base",
            pat_str=PATTERN,
            mergeable_ranks=load_tiktoken_bpe(str(path), expected_hash=SHA256),
            special_tokens={"<|endoftext|>": 199999, "<|endofprompt|>": 200018},
        )

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))

    def messages(self, messages: list[dict[str, str]]) -> int:
        # Conservative framing reserve; actual provider usage is reconciled.
        return 32 + sum(32 + self.count(m["role"]) + self.count(m["content"]) for m in messages)
