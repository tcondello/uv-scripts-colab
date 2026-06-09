# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=3.0",
#   "pinecone>=5.0",
#   "huggingface-hub>=0.25",
#   "rich>=13.0",
# ]
# ///
"""upsert-to-pinecone.py — push HF embeddings into a Pinecone index.

Reads a HF dataset that has `imdbId`, `poster` (URL), and `embedding` columns
(produced by `recipes/clip-embed.py`), creates a serverless Pinecone index if
it doesn't exist, and upserts the vectors keyed by `imdbId` with the poster
URL in metadata.

This is a *local* script (not a Colab recipe) — your vectors already live on
HF; this just gets them into Pinecone for similarity search.

Env vars (or positional CLI args):
    HF_DATASET          HF embeddings dataset  [Tim-Pinecone/movie-posters-siglip-embeddings]
    INDEX_NAME          Pinecone index name    [movie-posters-siglip]
    NAMESPACE           Pinecone namespace     [""]
    PINECONE_API_KEY    Pinecone API key       (required)
    PINECONE_CLOUD      Cloud                  [aws]
    PINECONE_REGION     Region                 [us-east-1]
    ID_COLUMN           Column to use as id    [imdbId]
    META_COLUMNS        Comma-sep metadata cols [poster]

Usage:
    PINECONE_API_KEY=pck_... uv run bonus/upsert-to-pinecone.py
"""
from __future__ import annotations

import os
import sys
import time
from typing import Iterator


def _cfg(name: str, *, default: str | None = None, pos: int | None = None) -> str | None:
    if name in os.environ and os.environ[name]:
        return os.environ[name]
    if pos is not None and len(sys.argv) > pos:
        return sys.argv[pos]
    return default


def main() -> None:
    from datasets import load_dataset
    from pinecone import Pinecone, ServerlessSpec, Vector
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

    console = Console()

    api_key = _cfg("PINECONE_API_KEY")
    if not api_key:
        console.print(
            "[red]error:[/red] PINECONE_API_KEY not set.\n"
            "       export PINECONE_API_KEY=pck_... (from https://app.pinecone.io)\n"
            "       or run: pc auth login && pc config get-api-key"
        )
        sys.exit(1)

    hf_dataset = _cfg(
        "HF_DATASET",
        pos=1,
        default="Tim-Pinecone/movie-posters-siglip-embeddings",
    )
    index_name = _cfg("INDEX_NAME", pos=2, default="movie-posters-siglip")
    namespace = _cfg("NAMESPACE", default="")
    cloud = _cfg("PINECONE_CLOUD", default="aws")
    region = _cfg("PINECONE_REGION", default="us-east-1")
    id_column = _cfg("ID_COLUMN", default="imdbId")
    meta_columns = [
        c.strip() for c in _cfg("META_COLUMNS", default="poster").split(",") if c.strip()
    ]

    # --- Load embeddings dataset --------------------------------------------
    with console.status(f"[cyan]Loading {hf_dataset}...", spinner="dots"):
        t0 = time.time()
        ds = load_dataset(hf_dataset, split="train")
    console.print(
        f"[green]✓[/green] loaded [bold]{len(ds):,}[/bold] rows "
        f"({time.time() - t0:.1f}s), columns: {ds.column_names}"
    )

    missing = [c for c in (id_column, "embedding", *meta_columns) if c not in ds.column_names]
    if missing:
        console.print(f"[red]error:[/red] dataset is missing columns: {missing}")
        sys.exit(2)

    dim = len(ds[0]["embedding"])

    console.print(
        Panel.fit(
            f"[bold]upsert-to-pinecone[/bold]\n"
            f"from:      [cyan]{hf_dataset}[/cyan]\n"
            f"to:        [green]{index_name}[/green] (ns=[yellow]{namespace or '\"\"'}[/yellow])\n"
            f"dim:       {dim}\n"
            f"id:        {id_column}\n"
            f"metadata:  {', '.join(meta_columns)}\n"
            f"region:    {cloud}/{region}",
            title="config",
            border_style="blue",
        )
    )

    # --- Pinecone client + index --------------------------------------------
    pc = Pinecone(api_key=api_key)

    existing = [i["name"] for i in pc.list_indexes()]
    if index_name not in existing:
        console.print(
            f"[dim]Index '{index_name}' not found — creating "
            f"(dim={dim}, metric=cosine, {cloud}/{region})...[/dim]"
        )
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        # Wait for the index to come ready before upserting.
        with console.status("[cyan]Waiting for index to come ready...", spinner="dots"):
            for _ in range(60):
                desc = pc.describe_index(index_name)
                if getattr(desc, "status", {}) and desc.status.get("ready"):
                    break
                time.sleep(2)
        console.print(f"[green]✓[/green] index '{index_name}' ready")
    else:
        console.print(f"[dim]Reusing existing index '{index_name}'[/dim]")

    index = pc.Index(index_name)

    # --- Build + upsert vectors ---------------------------------------------
    def iter_vectors() -> Iterator[Vector]:
        for row in ds:
            yield Vector(
                id=str(row[id_column]),
                values=row["embedding"],
                metadata={c: row[c] for c in meta_columns if row.get(c) is not None},
            )

    vectors = list(iter_vectors())
    n = len(vectors)
    batch_size = 100
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
    ) as progress:
        task = progress.add_task("[cyan]upserting[/cyan]", total=n, rate="")
        sent = 0
        for start in range(0, n, batch_size):
            chunk = vectors[start : start + batch_size]
            index.upsert(vectors=chunk, namespace=namespace)
            sent += len(chunk)
            elapsed = max(time.time() - t0, 1e-6)
            progress.update(
                task,
                advance=len(chunk),
                rate=f"{sent / elapsed:,.0f} vec/s",
            )

    stats = index.describe_index_stats()
    console.print(
        f"[green]✓[/green] upserted [bold]{n:,}[/bold] vectors in "
        f"[bold]{time.time() - t0:.1f}s[/bold]"
    )
    console.print(
        Panel.fit(
            f"[bold green]Done.[/bold green]\n\n"
            f"index:      {index_name}\n"
            f"dim:        {dim}\n"
            f"total vecs: {stats.get('total_vector_count', '?')}\n"
            f"namespaces: {list(stats.get('namespaces', {}).keys())}",
            title="result",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
