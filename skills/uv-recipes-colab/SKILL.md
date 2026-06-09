---
name: uv-recipes-colab
description: "Run self-contained UV-script recipes over Hugging Face datasets on a managed Google Colab GPU using `bin/colab-hf-run`. Each recipe reads a Hub dataset and writes a new one, so recipes chain into pipelines: text embeddings, image embeddings (SigLIP/CLIP), zero-shot PII extraction (GLiNER), and smoke tests. Use when the user wants to batch-process a dataset at scale on a Colab T4/L4/A100/H100, pre-label data for review, or build a small HF data pipeline without writing notebooks or installing CUDA locally. Inspired by `davanstrien/uv-scripts-for-ai` but pointed at Google's Colab CLI instead of Hugging Face Jobs. Recipes live in `recipes/` and run with one terminal command — local file path or raw http(s):// URL."
---

# uv-recipes-colab

A recipe is one self-contained Python file (a [PEP 723](https://peps.python.org/pep-0723/) [UV script](https://docs.astral.sh/uv/guides/scripts/)) that reads a Hugging Face dataset and writes a new one. Recipes take config via env vars and run on a managed Colab GPU through a thin bash wrapper — no clone, no virtualenv, no CUDA install.

The repo: [`tcondello/uv-scripts-colab`](https://github.com/tcondello/uv-scripts-colab).

## Requires

- **`uv`** — `curl -LsSf https://astral.sh/uv/install.sh | sh`. The recipes themselves use PEP 723 metadata blocks; locally `uv run` honors them. On Colab, recipes self-install missing deps via a `pip` shim because Colab's Jupyter kernel doesn't read PEP 723.
- **Google Colab CLI** — `uv tool install google-colab-cli`. Provides `colab new`, `colab exec`, `colab run`, `colab stop`. First invocation opens a browser for Google OAuth; subsequent calls are silent. Requires a Google account with Colab compute units (`colab pay` shows plans).
- **Hugging Face auth** — `hf auth login` with a **write-scoped** token from https://huggingface.co/settings/tokens. The wrapper reads it from `~/.cache/huggingface/token` and injects it into the Colab kernel for you.
- **The wrapper** — `bin/colab-hf-run`. Either clone the repo and use the script directly, or save it to your `PATH` so you can run recipes from any directory.

## Run a recipe

The wrapper accepts either a local file path **or** an `http(s)://` URL. URL recipes are `curl`'d to a temp file and then piped into `colab exec`:

```bash
# From a clone of the repo:
INPUT_DATASET=stanfordnlp/sst2 \
OUTPUT_DATASET=you/sst2-MiniLM-embeddings \
LIMIT=1000 \
bin/colab-hf-run recipes/embed-dataset.py

# Directly from the published raw URL — no clone needed:
INPUT_DATASET=stanfordnlp/sst2 \
OUTPUT_DATASET=you/sst2-MiniLM-embeddings \
LIMIT=1000 \
colab-hf-run https://raw.githubusercontent.com/tcondello/uv-scripts-colab/main/recipes/embed-dataset.py
```

### How the wrapper works

1. Reads `HF_TOKEN` from `~/.cache/huggingface/token` (or the env).
2. `colab new -s <name> --gpu $COLAB_GPU` provisions a session (default T4).
3. Builds a kernel preamble in Python (using `json.dumps` for safe quoting) that sets `os.environ[...]` for `HF_TOKEN` plus a whitelist of recipe-config vars.
4. `cat preamble + recipe | colab exec -s <name>` — kernel runs Python directly.
5. Trap on `EXIT`/`INT`/`TERM` calls `colab stop -s <name>` so sessions don't leak.

Forwarded env vars by default: `INPUT_DATASET OUTPUT_DATASET DATASET TEXT_COLUMN IMAGE_COLUMN MODEL_ID BATCH_SIZE LIMIT N_ROWS SPLIT`. Add more with `FORWARD_ENV="VAR1 VAR2"`. Override GPU with `COLAB_GPU=A100` (also accepts `T4`, `L4`, `G4`, `H100`). Keep the session alive after a failure for debugging with `KEEP_SESSION=1`.

## Recipe inventory

```
recipes/
  hello-gpu.py          smoke test: GPU info + distilbert sentiment over N rows
  embed-dataset.py      sentence-transformers text embeddings → HF dataset
  clip-embed.py         SigLIP/CLIP image embeddings (parallel URL fetch) → HF dataset
  gliner-pii.py         zero-shot PII extraction with GLiNER → HF dataset
  whisper-transcribe.py Whisper-large-v3 audio → text transcripts → HF dataset
  dataset-stats.py      per-column profiling + markdown report (CPU-only)
  vlm-detect.py         VLM object detection (Qwen2.5-VL-3B default) → HF dataset
```

Each file's docstring lists its env vars and an example invocation. To inspect a recipe before running, just `curl` and `head` the URL — or open it in your editor. The PEP 723 block at the top tells you the deps.

## Compose a pipeline

Each recipe's output dataset is the next recipe's input (handoff through the Hub):

```
text dataset                 →  embed-dataset.py    →  text-embeddings dataset
image dataset (URLs ok)      →  clip-embed.py       →  image-embeddings dataset
text dataset (free-form)     →  gliner-pii.py       →  PII-tagged dataset
```

A pipeline can also fork: embed once with `embed-dataset.py`, then send the embedding dataset to a vector DB (Pinecone, lancedb, Weaviate) with a separate local script.

## Pre-label, then review

GLiNER and other prediction recipes produce **bootstrap labels, not ground truth.** For anything that affects users — masking, deletion, redaction — put a human in the loop. The simplest viable reviewer is a single-file Gradio app: load the predictions dataset, show each row + its predicted spans with highlighted overlays, write corrections back to a new Hub dataset. Order the queue by score (low-confidence first) or by diversity (sample across labels) so review effort lands where it matters.

## Known quirks (live with these)

- **`HF_TOKEN` is not auto-forwarded by Colab CLI.** Colab's secret-vault only works from the Notebook UI, not `colab exec`. The wrapper handles this for recipes it launches; if you write your own kernel code, set the env var explicitly.
- **`colab exec` exits non-zero on success.** After a recipe finishes cleanly, the CLI's Jupyter-kernel shutdown handshake races and raises `TimeoutError`. **The recipe's work completes** (output dataset lands on the Hub) — don't gate CI on the exit code. Check the Hub or grep stdout for "Done.".
- **`sys.argv` is the kernel's argv, not yours.** When the wrapper pipes a recipe into `colab exec` via stdin, `sys.argv[0]` is `ipykernel_launcher` and `sys.argv[1+]` is a `kernel-uuid.json` path. **Recipes for the wrapper should read config from env vars**, not positional args. (Positional args work with `colab run <file>` directly.)
- **GitHub raw URLs are CDN-cached.** Expect 2–10 min of lag between `git push` and the new content showing up at `raw.githubusercontent.com`. Cache-bust during iteration with `?t=$(date +%s)`.
- **Colab T4 assignment is bursty.** Sustained churn can return `503 Service Unavailable` on session create. Wait and retry; try `COLAB_GPU=L4` if it persists.
- **One Colab session at a time.** Launching `bin/colab-hf-run` twice in parallel raises `TooManyAssignmentsError`. Run sequentially.
- **POSIX-standard env var names collide with the Colab kernel.** Vars like `LANG`, `LANGUAGE`, `LC_ALL`, `USER`, `HOME`, `PATH` are pre-set in the kernel environment. A recipe that reads them via `os.environ.get("LANGUAGE")` will get the system locale (`en_US:en`), not an unset value. Use namespaced names (`WHISPER_LANGUAGE`, `OCR_PROMPT`, etc.) for any recipe config.
- **Long runs need bandwidth discipline.** A 30-minute run with default Rich update rates (10 Hz) can saturate the kernel WebSocket and drop the connection — losing in-memory progress. The image recipe throttles Rich to 2 Hz and parallelizes URL fetching; copy that pattern for any recipe doing heavy IO.
- **`load_dataset(id, split="train[:N]")` then `push_to_hub` fails.** Split names must match `^\w+(\.\w+)*$`. Load full, then `ds.select(range(N))`.

The repo's `README.md` has the same list with more detail.

## Write a new recipe

Use this template, then drop into `recipes/`:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=3.0",
#   "torch>=2.4",
#   "huggingface-hub>=0.25",
#   "rich>=13.0",
#   # ... your model-specific deps
# ]
# ///
"""docstring documenting env vars + example invocation."""
import os, subprocess, sys

def _ensure_deps():
    try:
        import datasets, torch, rich  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "datasets>=3.0", "torch>=2.4", "rich>=13.0"])

def _cfg(name, *, default=None, pos=None):
    """env var → positional arg (skip kernel json paths) → default."""
    if name in os.environ and os.environ[name]:
        return os.environ[name]
    if pos is not None and pos < len(sys.argv) and not sys.argv[pos].endswith(".json"):
        return sys.argv[pos]
    return default

def main():
    _ensure_deps()
    # ...
```

To make a new env var pass through the wrapper, add it to `FORWARD_ENV="MY_VAR"` or PR it into `DEFAULT_FORWARD` in `bin/colab-hf-run`.

## If a recipe fails

Triage in this order — most failures are environment or usage, not recipe bugs:

- **No HF token** → `hf auth login` (write-scoped).
- **403 Forbidden on push** → your HF token's namespace doesn't match `OUTPUT_DATASET`. Use your actual HF username (check with `hf auth whoami`).
- **503 from Colab session create** → bursty T4 availability. Wait + retry, or try `COLAB_GPU=L4`.
- **`Connection was lost` mid-run** → WebSocket overload. Reduce stdout volume (throttle `rich.Progress(refresh_per_second=2)`), parallelize IO so the run is shorter, or checkpoint partial results back to the Hub.
- **Exit code 1, but the dataset is on the Hub** → kernel-shutdown race. Not a real failure; trust the Hub.
- **`OOM`** → reduce `BATCH_SIZE`, or `COLAB_GPU=A100`.
- **`ValueError: Split name should match …`** → don't use `split="train[:N]"`. Use `ds = ds.select(range(N))`.

If a recipe is genuinely broken after triage, open an issue at https://github.com/tcondello/uv-scripts-colab/issues with: the recipe URL, the exact env-var invocation (with **tokens redacted**), a public input dataset that reproduces it, and the stderr tail.
