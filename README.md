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

*(More to come — OCR, embeddings, fine-tuning. PRs welcome once the pattern is settled.)*

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
