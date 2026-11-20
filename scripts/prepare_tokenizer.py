"""Download public tokenizer vocabulary only; never contacts the hackathon API."""

import hashlib
import urllib.request
from pathlib import Path

URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
target = Path(__file__).resolve().parent.parent / "app/assets/tiktoken/o200k_base.tiktoken"
if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == SHA256:
    print("Tokenizer already prepared")
else:
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read(5 * 1024 * 1024)
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise SystemExit("Tokenizer checksum mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print("Public tokenizer vocabulary prepared locally")
