# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=3.0",
#   "pinecone>=5.0",
#   "huggingface-hub>=0.25",
#   "rich>=13.0",
# ]
# ///
"""find-similar.py — query Pinecone for posters that look like a given one.

Given an imdbId (e.g. tt0111161 for Shawshank), or a row index, look up its
embedding in the HF dataset, query Pinecone for the top-K nearest neighbors,
and write an HTML page that shows the query poster next to the similar ones.
Opens the page in your default browser.

Env vars (or positional CLI args):
    QUERY               imdbId or integer row index    (required)
    TOP_K               How many similar posters       [10]
    HF_DATASET          HF embeddings dataset          [Tim-Pinecone/movie-posters-siglip-embeddings]
    INDEX_NAME          Pinecone index name            [movie-posters-siglip]
    NAMESPACE           Pinecone namespace             [""]
    PINECONE_API_KEY    Pinecone API key               (required)
    OUTPUT_HTML         Path to write HTML to          [/tmp/posters-similar.html]
    NO_OPEN             If set, don't open browser

Usage:
    # By IMDb id (Shawshank):
    PINECONE_API_KEY=pck_... uv run bonus/find-similar.py tt0111161

    # By row index:
    PINECONE_API_KEY=pck_... uv run bonus/find-similar.py 42
"""
from __future__ import annotations

import html
import os
import sys
import webbrowser
from pathlib import Path


def _cfg(name: str, *, default: str | None = None, pos: int | None = None) -> str | None:
    if name in os.environ and os.environ[name]:
        return os.environ[name]
    if pos is not None and len(sys.argv) > pos:
        return sys.argv[pos]
    return default


def main() -> None:
    from datasets import load_dataset
    from pinecone import Pinecone
    from rich.console import Console

    console = Console()

    api_key = _cfg("PINECONE_API_KEY")
    if not api_key:
        console.print(
            "[red]error:[/red] PINECONE_API_KEY not set. "
            "export PINECONE_API_KEY=pck_..."
        )
        sys.exit(1)

    query = _cfg("QUERY", pos=1)
    if not query:
        console.print(
            "[red]error:[/red] missing query. Pass an imdbId or row index, "
            "e.g. `uv run bonus/find-similar.py tt0111161`"
        )
        sys.exit(2)

    top_k = int(_cfg("TOP_K", default="10"))
    hf_dataset = _cfg(
        "HF_DATASET",
        default="Tim-Pinecone/movie-posters-siglip-embeddings",
    )
    index_name = _cfg("INDEX_NAME", default="movie-posters-siglip")
    namespace = _cfg("NAMESPACE", default="")
    output_html = Path(_cfg("OUTPUT_HTML", default="/tmp/posters-similar.html"))

    # --- Look up the query vector + poster URL ------------------------------
    console.print(f"[dim]Loading {hf_dataset}...[/dim]")
    ds = load_dataset(hf_dataset, split="train")

    query_vec = None
    query_poster = None
    query_id = None
    if query.isdigit() and not query.startswith("tt"):
        # Row-index query
        idx = int(query)
        if idx >= len(ds):
            console.print(f"[red]error:[/red] row {idx} out of range ({len(ds)})")
            sys.exit(2)
        query_vec = ds[idx]["embedding"]
        query_poster = ds[idx].get("poster")
        query_id = str(ds[idx].get("imdbId", idx))
    else:
        # imdbId query — scan the dataset (small enough, ~10k rows)
        for row in ds:
            if str(row.get("imdbId")) == query:
                query_vec = row["embedding"]
                query_poster = row.get("poster")
                query_id = query
                break
        if query_vec is None:
            console.print(
                f"[red]error:[/red] imdbId '{query}' not found in {hf_dataset}"
            )
            sys.exit(2)

    console.print(f"[green]✓[/green] query: [bold]{query_id}[/bold]")

    # --- Query Pinecone -----------------------------------------------------
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    console.print(f"[dim]Querying {index_name} for top {top_k} neighbors...[/dim]")
    res = index.query(
        vector=query_vec,
        top_k=top_k + 1,  # +1 because the query itself is usually #1
        namespace=namespace,
        include_metadata=True,
    )

    matches = [
        m for m in res.get("matches", []) if str(m.get("id")) != query_id
    ][:top_k]

    if not matches:
        console.print("[red]No matches returned.[/red]")
        sys.exit(1)

    console.print(f"[green]✓[/green] got {len(matches)} matches\n")
    for i, m in enumerate(matches, 1):
        console.print(
            f"  {i:>2}. [bold]{m['id']}[/bold]  score=[cyan]{m['score']:.3f}[/cyan]  "
            f"[dim]{(m.get('metadata') or {}).get('poster', '')[:80]}[/dim]"
        )

    # --- Render HTML --------------------------------------------------------
    def _img_url(meta: dict | None, fallback: str | None = None) -> str:
        if meta and meta.get("poster"):
            return str(meta["poster"])
        return fallback or ""

    cards = []
    for i, m in enumerate(matches, 1):
        url = _img_url(m.get("metadata"))
        cards.append(
            f"""
            <div class="card">
              <div class="rank">#{i}</div>
              {'<img src="' + html.escape(url) + '"/>' if url else '<div class="no-img">no poster</div>'}
              <div class="meta">
                <code>{html.escape(str(m['id']))}</code>
                <span class="score">{m['score']:.3f}</span>
              </div>
            </div>
            """
        )

    query_img = (
        f'<img src="{html.escape(str(query_poster))}"/>' if query_poster
        else '<div class="no-img">no poster</div>'
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Similar posters to {html.escape(query_id)}</title>
  <style>
    :root {{ color-scheme: dark light; font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif; }}
    body {{ margin: 0; padding: 24px; background: #0f0f10; color: #eee; }}
    h1 {{ font-weight: 500; font-size: 18px; margin: 0 0 4px; }}
    .sub {{ color: #888; font-size: 13px; margin-bottom: 24px; }}
    .row {{ display: flex; flex-wrap: wrap; gap: 20px; }}
    .query, .card {{ width: 180px; flex: 0 0 auto; }}
    .query img, .card img {{ width: 100%; height: 270px; object-fit: cover; border-radius: 6px; background: #222; }}
    .query {{ outline: 2px solid #6a8; padding: 6px; border-radius: 8px; }}
    .rank {{ position: relative; top: 22px; left: 6px; width: 28px; height: 28px; background: #2a2a2c; color: #ddd; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; }}
    .meta {{ display: flex; justify-content: space-between; margin-top: 6px; font-size: 12px; color: #aaa; }}
    .score {{ color: #6cd; font-variant-numeric: tabular-nums; }}
    .no-img {{ height: 270px; display: flex; align-items: center; justify-content: center; background: #1a1a1c; color: #555; border-radius: 6px; font-size: 12px; }}
    code {{ font-family: ui-monospace, SF Mono, monospace; }}
    .divider {{ width: 1px; background: #2a2a2c; margin: 0 8px; }}
  </style>
</head>
<body>
  <h1>Posters most similar to <code>{html.escape(query_id)}</code></h1>
  <div class="sub">index: <code>{html.escape(index_name)}</code> · top {len(matches)} · SigLIP embeddings via Colab T4 → Pinecone</div>
  <div class="row">
    <div class="query">
      <div class="rank">Q</div>
      {query_img}
      <div class="meta"><code>{html.escape(query_id)}</code><span class="score">query</span></div>
    </div>
    <div class="divider"></div>
    {''.join(cards)}
  </div>
</body>
</html>
"""

    output_html.write_text(html_doc)
    console.print(f"\n[green]✓[/green] wrote {output_html}")

    if not _cfg("NO_OPEN"):
        webbrowser.open(f"file://{output_html.resolve()}")


if __name__ == "__main__":
    main()
