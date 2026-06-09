# uv-scripts-colab

> One-file Python recipes that **read a dataset from the Hugging Face Hub, run it on a managed Colab GPU, and push the results back to the Hub** — without writing a notebook, without installing anything beyond `uv` and the Colab CLI.

Inspired by [`davanstrien/uv-scripts-for-ai`](https://github.com/davanstrien/uv-scripts-for-ai), which targets [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) as the remote-GPU backend. This repo is the same idea, but pointed at the [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) instead — so the managed-GPU side runs on Colab's T4 / L4 / A100 / H100 fleet.

## The marquee example

Take a Hugging Face dataset of [10,269 movie posters](https://huggingface.co/datasets/pinecone/movie-posters), embed every one of them with SigLIP on a Colab T4, and write a new dataset back to the Hub — with one terminal command:

```bash
INPUT_DATASET=pinecone/movie-posters \
OUTPUT_DATASET=your-username/movie-posters-siglip-embeddings \
bin/colab-hf-run recipes/clip-embed.py
```

Real result on a T4 (June 2026): **6,034 posters × 768-dim embeddings, pushed to the Hub in 5.8 minutes wall-clock** — including session provisioning, model download, image fetch, GPU inference, and the push back. The output dataset preserves the original schema (`imdbId`, `poster` URL) and adds an `embedding` column, so it composes with whatever you do next (visualize, search, build an atlas, fine-tune a reranker).

Output dataset: [`Tim-Pinecone/movie-posters-siglip-embeddings`](https://huggingface.co/datasets/Tim-Pinecone/movie-posters-siglip-embeddings)

## Quickstart

**1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and the [Colab CLI](https://github.com/googlecolab/google-colab-cli):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install google-colab-cli
```

**2. Log into Hugging Face once** (write-scoped token from https://huggingface.co/settings/tokens):

```bash
hf auth login
```

The wrapper reads the cached token from `~/.cache/huggingface/token` and injects it into the Colab kernel for you — no copy-paste, no envs to manage.

**3. Smoke-test the Colab side:**

```bash
colab run --gpu T4 recipes/hello-gpu.py
```

First invocation opens a browser for Google OAuth; subsequent runs are silent. Colab CLI uses your existing Colab compute units — see `colab pay`.

**4. Run a real recipe** that reads + writes the Hub:

```bash
INPUT_DATASET=stanfordnlp/sst2 \
OUTPUT_DATASET=your-username/sst2-MiniLM-embeddings \
LIMIT=1000 \
bin/colab-hf-run recipes/embed-dataset.py
```

That's it — same shape for every recipe.

### You don't even need to clone

`bin/colab-hf-run` accepts an http(s):// URL as the recipe argument. If you have the wrapper on your `PATH` (or save it as `~/bin/colab-hf-run`), you can run any recipe from this repo (or any raw URL) without cloning:

```bash
INPUT_DATASET=stanfordnlp/sst2 \
OUTPUT_DATASET=your-username/sst2-MiniLM-embeddings \
LIMIT=1000 \
colab-hf-run https://raw.githubusercontent.com/tcondello/uv-scripts-colab/main/recipes/embed-dataset.py
```

The same works for recipes hosted as raw files on a HF dataset repo, a gist, or anywhere else — the wrapper just `curl`s the file, then pipes it (plus the kernel preamble) into `colab exec`.

## What's a recipe?

A single Python file with a [PEP 723](https://peps.python.org/pep-0723/) metadata block at the top declaring its dependencies:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "transformers", "torch"]
# ///
```

`uv` reads that block and installs the pinned deps into a throwaway env. The Colab CLI doesn't read PEP 723 yet, so recipes here also self-install missing deps via `pip` at runtime — same script works locally on a CUDA box (`uv run recipes/foo.py`) and on a managed Colab GPU (`bin/colab-hf-run recipes/foo.py`).

Every recipe reads from and/or writes to the [Hugging Face Hub](https://huggingface.co/datasets), which is what makes them composable: one recipe's output dataset is the next one's input.

## Recipes

| Recipe | What it does |
|---|---|
| `recipes/hello-gpu.py` | Smoke test: prints GPU info + sentiment over N rows of an HF dataset. Use this to confirm your Colab CLI setup works end-to-end. |
| `recipes/embed-dataset.py` | Embed any HF text dataset on GPU with `sentence-transformers` and push the embedded dataset back to the Hub. |
| `recipes/clip-embed.py` | Embed an HF image dataset (PIL images **or** URLs) on GPU with SigLIP / CLIP and push back. Parallel image fetching (8 workers, configurable). |

PRs welcome — fine-tuning recipes (QLoRA producing an adapter back on the Hub), CLIP-text recipes for cross-modal search, and OCR recipes would all fit.

## The `colab-hf-run` wrapper

`colab run` is a one-shot "spin up a kernel, execute a file, tear down" — but it has no way to forward env vars to the kernel, and `huggingface_hub` inside a Colab session can't read your local HF token. That means recipes can read public HF datasets but **can't push back**, which kills the composability story.

`bin/colab-hf-run` is a thin bash wrapper that fixes this. It:

1. Accepts either a local file path **or** an `http(s)://` URL for the recipe — URL recipes are `curl`'d to a temp file
2. Reads your HF token from `~/.cache/huggingface/token`
3. Creates a fresh Colab session (`colab new --gpu T4`)
4. Injects `HF_TOKEN` + a whitelist of config env vars into the kernel as a preamble (`os.environ[...] = "..."`)
5. Streams the recipe's stdout/stderr back to your terminal — Rich progress bars and all
6. Stops the session on exit, even on Ctrl-C

Whitelisted env vars (set in your shell, the wrapper forwards them):
`INPUT_DATASET`, `OUTPUT_DATASET`, `TEXT_COLUMN`, `IMAGE_COLUMN`, `MODEL_ID`, `BATCH_SIZE`, `LIMIT`, `SPLIT`. Add more with `FORWARD_ENV="VAR1 VAR2"`. Override GPU flavor with `COLAB_GPU=A100`. Debug with `KEEP_SESSION=1`.

## Known quirks (Colab CLI v0.5.9)

These bit us building the recipes; capturing them so they don't surprise you:

- **`HF_TOKEN` is not auto-forwarded.** Colab's secret-vault mechanism only works from the Notebook UI, not `colab exec`. `colab-hf-run` works around this by injecting the token into the kernel.
- **`colab run` / `colab exec` exit non-zero on success.** After a recipe finishes cleanly, the CLI's Jupyter-kernel shutdown handshake races and raises `TimeoutError: Timeout waiting for reply` (or `Connection was lost`). The recipe's output is fine; the exit code is not. Don't gate CI on it — check the HF Hub for your output dataset, or grep stdout for "Done.".
- **Long-running runs need bandwidth discipline.** A 30-minute run with default Rich update rates can saturate the kernel WebSocket and drop the connection (losing all in-memory progress). The image-embed recipe throttles Rich to 2Hz and parallelizes image downloads (8 threads) — full 10k poster run fits in ~6 min. If you write a longer recipe, consider periodic checkpointing back to the Hub.
- **`load_dataset("foo", split="train[:N]")` then `push_to_hub` fails.** The split name `train[:N]` doesn't match `^\w+(\.\w+)*$`. Load full, then `.select(range(N))`.
- **OAuth on first run.** Default `--auth oauth2` opens a browser; the URL only shows in stderr. If you're driving the CLI from a wrapped terminal and don't see the URL, check `~/.config/colab-cli/colab.log`.

## How it compares to HF Jobs

|  | `uv-scripts-for-ai` (HF Jobs) | `uv-scripts-colab` (this repo) |
|---|---|---|
| Remote runner | `hf jobs uv run <url>` | `bin/colab-hf-run recipes/foo.py` |
| Hardware | NVIDIA L4 / A10 / A100 / H100 | NVIDIA T4 / L4 / A100 / H100 (Colab fleet) |
| Billing | Pay-per-second, HF pricing | Colab compute units |
| Reads URL directly? | Yes | Yes — local file **or** http(s):// URL (e.g. raw GitHub / HF dataset URL) |
| PEP 723 native? | Yes (uv-aware) | No — recipes self-install deps |
| Auth | `HF_TOKEN` forwarded via `--secrets` | Wrapper reads `~/.cache/huggingface/token` |

The two are complementary — use whichever managed-GPU platform you're already paying for.

## Where this goes next

Things this scaffolding makes easy that aren't here yet:

- **Fine-tuning recipes.** QLoRA fine-tune of a 7-8B model on T4, push the adapter back to the Hub. T4 is genuinely a fine-tuning workhorse for small models.
- **Cross-modal search.** Embed images and text in the same SigLIP space, query one with the other.
- **Vector-DB integrations.** Embeddings on the Hub are halfway to a working retrieval app — wire them into Pinecone / Weaviate / lancedb with one more local script.
- **Atlases.** Pipe the output dataset of `clip-embed.py` into davanstrien's `build-atlas` recipe for an interactive UMAP of poster space.

## License

[Apache 2.0](LICENSE).
