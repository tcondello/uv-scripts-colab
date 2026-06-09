# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=3.0",
#   "transformers>=4.45",
#   "torch>=2.4",
#   "huggingface-hub>=0.25",
#   "Pillow>=10.0",
#   "rich>=13.0",
#   "requests>=2.31",
# ]
# ///
"""clip-embed.py — embed an image dataset on GPU and push results back.

Image analog of `embed-dataset.py`. Reads a Hugging Face dataset whose
image column is either PIL.Image (HF Image feature) or a URL string,
runs a SigLIP / CLIP vision encoder over it on GPU, and pushes a new
dataset back with an `embedding` column added (one fixed-dim vector per row).

If the image column is a string URL, this script casts it to the HF Image
feature so loading is lazy + concurrent. Failed image fetches are skipped
with a warning rather than crashing the whole run.

Env vars (or positional CLI args, in order):
    INPUT_DATASET       HF dataset id (e.g. pinecone/movie-posters)
    OUTPUT_DATASET      HF dataset id to push to
    IMAGE_COLUMN        Image column                 [auto-detect: poster|image|img|url]
    MODEL_ID            Vision encoder               [google/siglip-base-patch16-224]
    BATCH_SIZE          Per-batch images on GPU      [32]
    LIMIT               Max rows to embed            [no limit]
    SPLIT               Dataset split                [train]
    HF_TOKEN            For push                     (required)

Usage:
    # Via the wrapper (recommended):
    INPUT_DATASET=pinecone/movie-posters \\
    OUTPUT_DATASET=you/movie-posters-siglip-embeddings \\
        bin/colab-hf-run recipes/clip-embed.py

    # Locally with uv:
    INPUT_DATASET=... OUTPUT_DATASET=... uv run recipes/clip-embed.py
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def _ensure_deps() -> None:
    """Install deps not in the Colab base image."""
    try:
        import datasets  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import PIL  # noqa: F401
        import rich  # noqa: F401
    except ImportError:
        print("[deps] installing transformers + rich + Pillow...", flush=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "datasets>=3.0",
                "transformers>=4.45",
                "Pillow>=10.0",
                "rich>=13.0",
            ]
        )


def _cfg(name: str, *, default: str | None = None, pos: int | None = None) -> str | None:
    if name in os.environ and os.environ[name]:
        return os.environ[name]
    if pos is not None and len(sys.argv) > pos:
        return sys.argv[pos]
    return default


def main() -> None:
    _ensure_deps()

    import torch
    from datasets import Image as HfImage
    from datasets import load_dataset
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from transformers import AutoModel, AutoProcessor

    console = Console(force_terminal=True)

    input_id = _cfg("INPUT_DATASET", pos=1)
    output_id = _cfg("OUTPUT_DATASET", pos=2)
    image_col = _cfg("IMAGE_COLUMN", pos=3)
    model_id = _cfg(
        "MODEL_ID",
        pos=4,
        default="google/siglip-base-patch16-224",
    )
    batch_size = int(_cfg("BATCH_SIZE", default="32"))
    fetch_workers = int(_cfg("FETCH_WORKERS", default="8"))
    fetch_timeout = float(_cfg("FETCH_TIMEOUT", default="5.0"))
    limit_raw = _cfg("LIMIT")
    limit = int(limit_raw) if limit_raw else None
    split = _cfg("SPLIT", default="train")
    hf_token = _cfg("HF_TOKEN")

    if not input_id or not output_id:
        console.print(
            "[red]Missing required config.[/red] Set INPUT_DATASET and "
            "OUTPUT_DATASET (env vars or positional args)."
        )
        sys.exit(2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"

    console.print(
        Panel.fit(
            f"[bold]clip-embed[/bold]\n"
            f"input:   [cyan]{input_id}[/cyan] (split={split}"
            + (f", limit={limit:,}" if limit else "")
            + ")\n"
            f"output:  [green]{output_id}[/green]\n"
            f"model:   [magenta]{model_id}[/magenta]\n"
            f"device:  [yellow]{gpu_name}[/yellow]\n"
            f"batch:   {batch_size}  fetch_workers: {fetch_workers}",
            title="config",
            border_style="blue",
        )
    )

    # --- Load dataset --------------------------------------------------------
    with console.status(f"[cyan]Loading {input_id} (split={split})...", spinner="dots"):
        t0 = time.time()
        ds = load_dataset(input_id, split=split)
    if limit and len(ds) > limit:
        ds = ds.select(range(limit))
    console.print(
        f"[green]✓[/green] loaded [bold]{len(ds):,}[/bold] rows "
        f"({time.time() - t0:.1f}s), columns: {ds.column_names}"
    )

    # --- Pick image column + cast URL strings to Image feature ---------------
    if not image_col:
        for candidate in ("image", "img", "poster", "url", "image_url"):
            if candidate in ds.column_names:
                image_col = candidate
                break
        else:
            raise SystemExit(
                f"No image column found in {ds.column_names}. Set IMAGE_COLUMN."
            )

    # If the column is currently strings, cast a *separate copy* to HF Image
    # so datasets lazily downloads URLs at iteration time. We embed against
    # the cast copy but build the output from the original — that way the
    # pushed dataset preserves the source schema (URL strings stay URLs, not
    # blown up into parquet-embedded image bytes) and stays composable.
    feat = ds.features.get(image_col)
    if feat.__class__.__name__ != "Image":
        console.print(
            f"[dim]Casting [bold]{image_col}[/bold] to Image feature for "
            f"lazy URL download (original schema preserved on push).[/dim]"
        )
        ds_for_embed = ds.cast_column(image_col, HfImage())
    else:
        ds_for_embed = ds
    console.print(f"[dim]image column:[/dim] [bold]{image_col}[/bold]")

    # --- Load model + processor ---------------------------------------------
    with console.status(f"[cyan]Loading {model_id}...", spinner="dots"):
        t0 = time.time()
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
    console.print(f"[green]✓[/green] loaded model ({time.time() - t0:.1f}s)")

    def _extract_image_features(model, inputs):
        """Portable image-embedding extraction.

        Different vision models surface the embedding differently:
          - CLIP/SigLIP (full model)   → .get_image_features() returns tensor
          - Some versions / variants   → .get_image_features() returns
                                         BaseModelOutputWithPooling
          - Vision-only model          → fall back to .vision_model(...).pooler_output
          - Bare vision encoder        → model(**inputs).pooler_output
        """
        if hasattr(model, "get_image_features"):
            out = model.get_image_features(**inputs)
            if isinstance(out, torch.Tensor):
                return out
            for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
                t = getattr(out, attr, None)
                if isinstance(t, torch.Tensor):
                    return t.mean(dim=1) if attr == "last_hidden_state" else t
        if hasattr(model, "vision_model"):
            return model.vision_model(**inputs).pooler_output
        return model(**inputs).pooler_output

    # Defer dim discovery to the first real batch (no dummy probe needed).
    out_dim: int | None = None

    # --- Embed in batches with progress -------------------------------------
    # Image-download IS the bottleneck on this kind of workload (URL-based
    # image datasets); we keep the GPU fed by pre-fetching each batch's
    # images concurrently via a thread pool. The ds_for_embed accessor still
    # triggers HF datasets' lazy URL fetch, but multiple at once.
    import requests
    from PIL import Image as PILImage

    n = len(ds)
    embeddings: list[list[float] | None] = [None] * n
    valid_mask: list[bool] = [False] * n
    skipped = 0
    t0 = time.time()

    # Build a list of (idx, url) for direct fetching. We bypass datasets'
    # cast machinery here so we can control concurrency and timeouts.
    urls = ds[image_col]  # list of strings if original col was string
    using_urls = isinstance(urls[0], str) if urls else False

    sess = requests.Session()
    sess.headers.update({"User-Agent": "uv-scripts-colab/clip-embed"})

    def _fetch_one(i: int):
        try:
            if using_urls:
                resp = sess.get(urls[i], timeout=fetch_timeout)
                resp.raise_for_status()
                img = PILImage.open(io.BytesIO(resp.content))
            else:
                img = ds_for_embed[i][image_col]
                if img is None:
                    return i, None
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.load()  # decode now while still in the worker thread
            return i, img
        except Exception:
            return i, None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[rate]}[/cyan]"),
        TextColumn("[red]{task.fields[skipped]}[/red]"),
        console=console,
        transient=False,
        refresh_per_second=2,  # throttle stdout volume for stable WS connection
    ) as progress:
        task = progress.add_task(
            "[cyan]embedding[/cyan]",
            total=n,
            rate="",
            skipped="",
        )
        with ThreadPoolExecutor(max_workers=fetch_workers) as pool:
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                images = []
                idxs = []
                # Fetch this batch's images in parallel
                for i, img in pool.map(_fetch_one, range(start, end)):
                    if img is None:
                        skipped += 1
                    else:
                        images.append(img)
                        idxs.append(i)

                if not images:
                    progress.update(
                        task, advance=(end - start),
                        skipped=f"skip:{skipped}" if skipped else "",
                    )
                    continue

                with torch.no_grad():
                    inputs = processor(images=images, return_tensors="pt").to(device)
                    feats = _extract_image_features(model, inputs)
                    feats = torch.nn.functional.normalize(feats, dim=-1)
                    if out_dim is None:
                        out_dim = feats.shape[-1]
                    vecs = feats.cpu().numpy().tolist()

                for i, v in zip(idxs, vecs):
                    embeddings[i] = v
                    valid_mask[i] = True

                elapsed = max(time.time() - t0, 1e-6)
                done = start + (end - start)
                rate = f"{done / elapsed:,.1f} img/s"
                progress.update(
                    task,
                    advance=(end - start),
                    rate=rate,
                    skipped=(f"skip:{skipped}" if skipped else ""),
                )

    ok = sum(valid_mask)
    console.print(
        f"[green]✓[/green] embedded [bold]{ok:,}/{n:,}[/bold] images in "
        f"[bold]{time.time() - t0:.1f}s[/bold] "
        + (f"([red]{skipped} skipped[/red])" if skipped else "")
    )

    # --- Filter out failed rows and assemble output dataset -----------------
    # Drop rows where the image couldn't be loaded.
    if skipped:
        keep_idxs = [i for i, v in enumerate(valid_mask) if v]
        ds = ds.select(keep_idxs)
        embeddings = [e for e in embeddings if e is not None]

    out = ds.add_column("embedding", embeddings)

    if out_dim is None:
        console.print("[red]No images embedded successfully — nothing to push.[/red]")
        sys.exit(1)

    # --- Push ---------------------------------------------------------------
    with console.status(f"[cyan]Pushing to {output_id}...", spinner="dots"):
        t0 = time.time()
        out.push_to_hub(
            output_id,
            token=hf_token,
            commit_message=f"Add embeddings: {model_id} (dim={out_dim}, n={ok:,})",
        )
    console.print(
        f"[green]✓[/green] pushed [bold]{output_id}[/bold] "
        f"({time.time() - t0:.1f}s)"
    )

    url = f"https://huggingface.co/datasets/{output_id}"
    console.print(
        Panel.fit(
            f"[bold green]Done.[/bold green]\n\n"
            f"{ok:,} rows × {out_dim}-dim image embeddings\n"
            f"[link]{url}[/link]",
            title="result",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
