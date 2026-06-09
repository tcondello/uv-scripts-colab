# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=3.0",
#   "sentence-transformers>=3.0",
#   "torch>=2.4",
#   "huggingface-hub>=0.25",
#   "rich>=13.0",
# ]
# ///
"""embed-dataset.py — embed a HF dataset on a GPU and push results back.

Reads a Hugging Face dataset, runs a sentence-transformers model over the text
column, and writes a new dataset back to the Hub with an `embedding` column
added (one fixed-dim vector per row).

Designed for `bin/colab-hf-run` — config comes from environment variables so
the wrapper can inject `HF_TOKEN` and dataset names into the Colab kernel
before invoking. CLI args work too, for local `uv run` testing.

Env vars (or positional CLI args, in this order):
    INPUT_DATASET       HF dataset repo id (e.g. stanfordnlp/sst2)
    OUTPUT_DATASET      HF dataset repo id to push to (e.g. you/sst2-embeddings)
    TEXT_COLUMN         Column to embed   [auto-detect: sentence|text|review|content]
    MODEL_ID            Embedding model   [sentence-transformers/all-MiniLM-L6-v2]
    BATCH_SIZE          Batch size        [64]
    LIMIT               Max rows to embed [no limit]
    SPLIT               Dataset split     [train]
    HF_TOKEN            Hub token         (required to push output)

Usage:
    # Via the wrapper (recommended — handles token injection):
    bin/colab-hf-run recipes/embed-dataset.py

    # Locally on a GPU box with uv:
    INPUT_DATASET=stanfordnlp/sst2 OUTPUT_DATASET=you/sst2-emb \\
        uv run recipes/embed-dataset.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time


def _ensure_deps() -> None:
    """Install deps that aren't in the Colab base image."""
    try:
        import sentence_transformers  # noqa: F401
        import datasets  # noqa: F401
        import torch  # noqa: F401
        import rich  # noqa: F401
    except ImportError:
        print("[deps] installing sentence-transformers + rich...", flush=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "sentence-transformers>=3.0",
                "datasets>=3.0",
                "rich>=13.0",
            ]
        )


def _cfg(name: str, *, default: str | None = None, pos: int | None = None) -> str | None:
    """Read config from env var first, then positional CLI arg, then default."""
    if name in os.environ and os.environ[name]:
        return os.environ[name]
    if pos is not None and len(sys.argv) > pos:
        return sys.argv[pos]
    return default


def main() -> None:
    _ensure_deps()

    import torch
    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi
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
    from sentence_transformers import SentenceTransformer

    console = Console(force_terminal=True)

    input_id = _cfg("INPUT_DATASET", pos=1)
    output_id = _cfg("OUTPUT_DATASET", pos=2)
    text_col = _cfg("TEXT_COLUMN", pos=3)
    model_id = _cfg(
        "MODEL_ID",
        pos=4,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    batch_size = int(_cfg("BATCH_SIZE", default="64"))
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

    if not hf_token:
        console.print(
            "[yellow]Warning: no HF_TOKEN set.[/yellow] Push to the Hub will "
            "fail unless OUTPUT_DATASET is a repo you can write to "
            "unauthenticated (it isn't)."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"

    console.print(
        Panel.fit(
            f"[bold]embed-dataset[/bold]\n"
            f"input:   [cyan]{input_id}[/cyan] (split={split}"
            + (f", limit={limit:,}" if limit else "")
            + ")\n"
            f"output:  [green]{output_id}[/green]\n"
            f"model:   [magenta]{model_id}[/magenta]\n"
            f"device:  [yellow]{gpu_name}[/yellow]\n"
            f"batch:   {batch_size}",
            title="config",
            border_style="blue",
        )
    )

    # --- Load dataset --------------------------------------------------------
    split_expr = f"{split}[:{limit}]" if limit else split
    with console.status(f"[cyan]Loading {input_id} ({split_expr})...", spinner="dots"):
        t0 = time.time()
        ds = load_dataset(input_id, split=split_expr)
    console.print(
        f"[green]✓[/green] loaded [bold]{len(ds):,}[/bold] rows "
        f"({time.time() - t0:.1f}s), columns: {ds.column_names}"
    )

    # Pick a text column if not specified
    if not text_col:
        for candidate in ("sentence", "text", "review", "content", "title"):
            if candidate in ds.column_names:
                text_col = candidate
                break
        else:
            text_col = ds.column_names[0]
    console.print(f"[dim]embedding column:[/dim] [bold]{text_col}[/bold]")

    texts = [str(x) for x in ds[text_col]]

    # --- Load model ----------------------------------------------------------
    with console.status(f"[cyan]Loading {model_id}...", spinner="dots"):
        t0 = time.time()
        model = SentenceTransformer(model_id, device=device)
    dim = model.get_sentence_embedding_dimension()
    console.print(
        f"[green]✓[/green] loaded model ({time.time() - t0:.1f}s), "
        f"embedding dim = [bold]{dim}[/bold]"
    )

    # --- Embed in batches with live progress --------------------------------
    embeddings: list[list[float]] = []
    n = len(texts)
    t0 = time.time()
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
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(
            f"[cyan]embedding[/cyan]",
            total=n,
            rate="",
        )
        for start in range(0, n, batch_size):
            batch = texts[start : start + batch_size]
            vecs = model.encode(
                batch,
                batch_size=len(batch),
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            embeddings.extend(vecs.tolist())
            elapsed = max(time.time() - t0, 1e-6)
            rate = f"{len(embeddings) / elapsed:,.0f} rows/s"
            progress.update(task, advance=len(batch), rate=rate)

    console.print(
        f"[green]✓[/green] embedded [bold]{n:,}[/bold] rows in "
        f"[bold]{time.time() - t0:.1f}s[/bold] "
        f"([cyan]{n / (time.time() - t0):,.0f} rows/s[/cyan])"
    )

    # --- Build output dataset and push --------------------------------------
    out = ds.add_column("embedding", embeddings)

    with console.status(f"[cyan]Pushing to {output_id}...", spinner="dots"):
        t0 = time.time()
        out.push_to_hub(
            output_id,
            token=hf_token,
            commit_message=(
                f"Add embeddings: {model_id} (dim={dim}, n={n:,})"
            ),
        )
    console.print(
        f"[green]✓[/green] pushed [bold]{output_id}[/bold] "
        f"({time.time() - t0:.1f}s)"
    )

    api = HfApi(token=hf_token)
    url = f"https://huggingface.co/datasets/{output_id}"
    console.print(
        Panel.fit(
            f"[bold green]Done.[/bold green]\n\n"
            f"{n:,} rows × {dim}-dim embeddings\n"
            f"[link]{url}[/link]",
            title="result",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
