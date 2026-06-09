# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=3.0",
#   "transformers>=4.45",
#   "torch>=2.4",
#   "huggingface-hub>=0.25",
# ]
# ///
"""hello-gpu.py — smoke-test recipe for HF + Colab CLI.

Reads N rows from a Hugging Face dataset, runs sentiment classification with a
small distilbert pipeline, and prints results plus GPU info. Designed to
confirm the workflow end-to-end before building real recipes.

Config (env vars first, then positional args for local `uv run` use):
    DATASET     HF dataset id      [stanfordnlp/sst2]
    N_ROWS      Rows to embed      [10]

Usage:
    # Local with uv (positional args ok):
    uv run hello-gpu.py [DATASET] [N_ROWS]

    # Via colab-hf-run (env vars):
    DATASET=stanfordnlp/sst2 N_ROWS=20 \\
        bin/colab-hf-run recipes/hello-gpu.py

    # Or `colab run` directly (no env-var forwarding, positional args work):
    colab run --gpu T4 recipes/hello-gpu.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time


def _arg(env: str, pos: int, default: str) -> str:
    """Read config from env var first, then positional arg, then default.

    When this script is piped into `colab exec` (e.g. by colab-hf-run),
    sys.argv is the *kernel's* argv (its runtime json path), not user args
    — so we always check env vars first.
    """
    if env in os.environ and os.environ[env]:
        return os.environ[env]
    if pos < len(sys.argv) and not sys.argv[pos].endswith(".json"):
        return sys.argv[pos]
    return default


def _ensure_deps() -> None:
    """Colab base image has most of this — install only what's missing."""
    try:
        import datasets  # noqa: F401
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "datasets>=3.0",
                "transformers>=4.45",
                "torch>=2.4",
            ]
        )


def main() -> None:
    dataset_id = _arg("DATASET", 1, "stanfordnlp/sst2")
    n_rows = int(_arg("N_ROWS", 2, "10"))

    _ensure_deps()

    import torch
    from datasets import load_dataset
    from transformers import pipeline

    print("=== Environment ===", flush=True)
    print(f"torch:           {torch.__version__}", flush=True)
    print(f"cuda available:  {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"device:          {torch.cuda.get_device_name(0)}", flush=True)
        print(
            f"cuda capability: {torch.cuda.get_device_capability(0)}",
            flush=True,
        )

    print(
        f"\n=== Loading dataset: {dataset_id} (first {n_rows} rows) ===",
        flush=True,
    )
    t0 = time.time()
    ds = load_dataset(dataset_id, split=f"train[:{n_rows}]")
    print(
        f"loaded {len(ds)} rows in {time.time() - t0:.1f}s, "
        f"columns: {ds.column_names}",
        flush=True,
    )

    # Pick a text column — sst2 uses "sentence", most others use "text".
    text_col = next(
        (c for c in ("sentence", "text", "review", "content") if c in ds.column_names),
        ds.column_names[0],
    )
    texts = [str(x) for x in ds[text_col]]

    print("\n=== Running sentiment classification ===", flush=True)
    t0 = time.time()
    clf = pipeline(
        "sentiment-analysis",
        device=0 if torch.cuda.is_available() else -1,
    )
    results = clf(texts)
    print(f"inference: {time.time() - t0:.2f}s for {len(texts)} rows", flush=True)

    print("\n=== Results ===", flush=True)
    for text, result in zip(texts, results):
        snippet = text[:80].replace("\n", " ")
        print(
            f"[{result['label']:>8}  {result['score']:.3f}]  {snippet}",
            flush=True,
        )

    print("\n=== Done ===", flush=True)


if __name__ == "__main__":
    main()
