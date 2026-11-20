The bundled `o200k_base.tiktoken` is a tokenizer vocabulary, not model weights.

Source: https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken

SHA-256: `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`

The encoding pattern is from the installed OpenAI tiktoken distribution.
Its MIT license is reproduced in LICENSE.tiktoken.txt. The application loads the
vocabulary from disk and never calls tiktoken's network-backed registry.

`o200k_base` is an explicit tokenizer assumption. Matching it to the configured
proxy model requires real usage calibration, which has not been performed.
