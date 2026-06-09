# uv-scripts-colab

> Self-contained Python recipes for **Hugging Face + Google Colab CLI**. Each recipe is one file you can run on your own GPU with `uv run`, or push to a managed Colab GPU with `colab run` — no virtualenv, no `requirements.txt`, no setup.

Inspired by [`davanstrien/uv-scripts-for-ai`](https://github.com/davanstrien/uv-scripts-for-ai), which targets [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) as the remote-GPU backend. This repo is the same shape, but pointed at the [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) instead — so the managed-GPU side runs on Colab's T4 / A100 fleet.

## Quickstart

**1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and the [Colab CLI](https://github.com/googlecolab/google-colab-cli):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install google-colab-cli
```

**2. Run a recipe on a managed Colab T4:**

```bash
colab run --gpu T4 recipes/hello-gpu.py
```

First invocation opens a browser for Google OAuth. Colab CLI uses your existing Colab compute units — see `colab pay` for plan details.

**3. Or run it locally** (any box with a CUDA GPU, or CPU if you're patient):

```bash
uv run recipes/hello-gpu.py
```

## What's a recipe?

A single Python file with a [PEP 723](https://peps.python.org/pep-0723/) inline-metadata block at the top declaring its dependencies:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "transformers", "torch"]
# ///
```

`uv` reads that block and installs the pinned deps into a throwaway env before running. The Colab CLI doesn't read PEP 723 yet, so recipes here also self-install missing deps via `pip` at runtime — same script works on both.

Each recipe reads from and/or writes to the [Hugging Face Hub](https://huggingface.co/datasets) so they compose: one recipe's output dataset becomes the next one's input.

## Recipes

| Recipe | What it does |
|---|---|
| `recipes/hello-gpu.py` | Smoke test: prints GPU info, runs distilbert sentiment over N rows of a HF dataset. Use this to confirm your Colab CLI setup works end-to-end. |
| `recipes/embed-dataset.py` | Embed any HF dataset on GPU with `sentence-transformers` and push the embedded dataset back to the Hub. Run via the wrapper below. |

*(More to come — OCR, fine-tuning, reranker training. PRs welcome.)*

## The `colab-hf-run` wrapper

`colab run` can't pass env vars into the kernel, and the Colab UI's secret-vault mechanism doesn't work from `colab exec`. So `huggingface_hub` calls inside a recipe can't see your `HF_TOKEN` — fine for public reads, fatal for any write-back.

`bin/colab-hf-run` is a thin bash wrapper that fixes this. It reads your local `~/.cache/huggingface/token`, creates a Colab session, injects `HF_TOKEN` plus a whitelist of config env vars into the kernel as a preamble, then streams the recipe's output back to your terminal — `rich` progress bars and all. Session is stopped on exit, even on Ctrl-C.

**One-time setup:**

```bash
hf auth login          # paste a write-scoped token
```

**Run a recipe that writes back to the Hub:**

```bash
INPUT_DATASET=stanfordnlp/sst2 \
OUTPUT_DATASET=your-username/sst2-MiniLM-embeddings \
LIMIT=1000 \
bin/colab-hf-run recipes/embed-dataset.py
```

Forwarded env vars: `INPUT_DATASET`, `OUTPUT_DATASET`, `TEXT_COLUMN`, `MODEL_ID`, `BATCH_SIZE`, `LIMIT`, `SPLIT`. Add more with `FORWARD_ENV="VAR1 VAR2"`. Override GPU flavor with `COLAB_GPU=A100`. Keep the session alive for debugging with `KEEP_SESSION=1`.

## How it compares

|  | `uv-scripts-for-ai` (HF Jobs) | `uv-scripts-colab` (this repo) |
|---|---|---|
| Remote runner | `hf jobs uv run <url>` | `colab run --gpu T4 <file>` |
| Hardware | NVIDIA L4 / A10 / A100 / H100 | NVIDIA T4 / L4 / A100 (Colab fleet) |
| Billing | Pay-per-second, HF pricing | Colab compute units |
| Reads URL directly? | Yes | No — runs local file (for now) |
| PEP 723 native? | Yes (uv-aware) | No — recipes self-install deps |
| Auth | `HF_TOKEN` forwarded via `--secrets` | Google OAuth, ADC fallback |

The two are complementary — use whichever managed-GPU platform you're already paying for.

## Known quirks (Colab CLI v0.5.9)

These bit us writing the first recipe; capturing them so you don't trip on them too:

- **`HF_TOKEN` is not auto-forwarded.** Inside `colab exec` / `colab run`, `huggingface_hub` tries to read Colab's secret vault, times out (only the Notebook UI can read it), and falls back to unauthenticated requests. Public Hub resources work fine; private ones need you to pass the token explicitly (e.g. `colab install` a dotenv loader, or write the token into the session via `colab exec` before invoking the recipe).
- **`colab run` exits non-zero on success.** After a recipe finishes cleanly, the CLI's Jupyter-kernel shutdown handshake races and raises `TimeoutError: Timeout waiting for reply`. The recipe's output is fine; exit code is not. Don't gate CI on it — grep the output for a sentinel like `=== Done ===` instead.
- **OAuth on first run.** Default `--auth oauth2` opens a browser; the URL only shows in stderr. If you're driving the CLI from a wrapped terminal (Claude Code, tmux logging, etc.) and don't see the URL, check `~/.config/colab-cli/colab.log`.

## License

[Apache 2.0](LICENSE).
