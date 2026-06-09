# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gliner>=0.2.16",
#   "datasets>=3.0",
#   "torch>=2.4",
#   "huggingface-hub>=0.25",
#   "rich>=13.0",
# ]
# ///
"""gliner-pii.py — extract PII from a HF dataset with zero-shot NER.

Reads a Hugging Face dataset, runs [GLiNER](https://github.com/urchade/GLiNER)
over the specified text column to find PII entities (zero-shot — no
fine-tuning required), and pushes a new dataset back to the Hub with a
`pii_entities` column added.

Each row gets a list of records like:

    [{"start": 12, "end": 27, "text": "alice@acme.com", "label": "email", "score": 0.94},
     {"start": 43, "end": 56, "text": "415-555-0199", "label": "phone number", "score": 0.88}]

Original columns are preserved, so the output dataset is composable —
feed it to a masking recipe, a Label Studio reviewer, or a downstream
filter that drops rows containing high-risk labels.

> **GLiNER predictions are bootstrap labels.** Useful as a first pass
> or a triage filter; not ground truth. Always human-review before
> taking destructive action (e.g. deleting rows, masking content).

Env vars (or positional args, in order):
    INPUT_DATASET     HF dataset id (required)
    OUTPUT_DATASET    HF dataset id to push to (required)
    TEXT_COLUMN       Text col to scan       [auto: text|content|prompt|message|body]
    MODEL_ID          GLiNER model           [urchade/gliner_multi_pii-v1]
    ENTITY_TYPES      Comma-sep entity list  [PII default list — see below]
    THRESHOLD         Confidence cutoff      [0.5]
    BATCH_SIZE        Rows per batch         [8]
    LIMIT             Max rows               [no limit]
    SPLIT             Dataset split          [train]
    MAX_TEXT_CHARS    Truncate long text     [8000]
    HF_TOKEN          For push               (required)

Usage:
    # Via the wrapper (recommended):
    INPUT_DATASET=ai4privacy/pii-masking-200k \\
    OUTPUT_DATASET=you/pii-masking-200k-gliner \\
    LIMIT=500 \\
        bin/colab-hf-run recipes/gliner-pii.py

    # Override the entity list:
    ENTITY_TYPES="email,phone number,credit card" \\
        bin/colab-hf-run recipes/gliner-pii.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

# Sensible default PII labels. Kept lowercase (GLiNER is case-insensitive)
# and phrased the way the model card examples show. Override via ENTITY_TYPES.
DEFAULT_PII_LABELS = [
    "person",
    "email",
    "phone number",
    "address",
    "city",
    "country",
    "date of birth",
    "social security number",
    "credit card number",
    "passport number",
    "drivers license",
    "ip address",
    "url",
    "organization",
    "username",
    "password",
]


def _ensure_deps() -> None:
    """Install deps not in the Colab base image."""
    try:
        import gliner  # noqa: F401
        import datasets  # noqa: F401
        import torch  # noqa: F401
        import rich  # noqa: F401
    except ImportError:
        print("[deps] installing gliner + rich...", flush=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "gliner>=0.2.16",
                "datasets>=3.0",
                "rich>=13.0",
            ]
        )


def _cfg(name: str, *, default: str | None = None, pos: int | None = None) -> str | None:
    """Read config from env var first, then positional arg, then default.

    Note: sys.argv inside `colab exec` is the kernel's argv; we skip any
    positional arg that ends in '.json' (kernel runtime path).
    """
    if name in os.environ and os.environ[name]:
        return os.environ[name]
    if pos is not None and pos < len(sys.argv) and not sys.argv[pos].endswith(".json"):
        return sys.argv[pos]
    return default


def main() -> None:
    _ensure_deps()

    import torch
    from datasets import load_dataset
    from gliner import GLiNER
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

    console = Console(force_terminal=True)

    input_id = _cfg("INPUT_DATASET", pos=1)
    output_id = _cfg("OUTPUT_DATASET", pos=2)
    text_col = _cfg("TEXT_COLUMN", pos=3)
    model_id = _cfg("MODEL_ID", pos=4, default="urchade/gliner_multi_pii-v1")
    entity_types_raw = _cfg("ENTITY_TYPES")
    entity_types = (
        [s.strip() for s in entity_types_raw.split(",") if s.strip()]
        if entity_types_raw
        else DEFAULT_PII_LABELS
    )
    threshold = float(_cfg("THRESHOLD", default="0.5"))
    batch_size = int(_cfg("BATCH_SIZE", default="8"))
    max_chars = int(_cfg("MAX_TEXT_CHARS", default="8000"))
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
            f"[bold]gliner-pii[/bold]\n"
            f"input:    [cyan]{input_id}[/cyan] (split={split}"
            + (f", limit={limit:,}" if limit else "")
            + ")\n"
            f"output:   [green]{output_id}[/green]\n"
            f"model:    [magenta]{model_id}[/magenta]\n"
            f"device:   [yellow]{gpu_name}[/yellow]\n"
            f"labels:   {', '.join(entity_types[:6])}"
            + (f", +{len(entity_types) - 6} more" if len(entity_types) > 6 else "")
            + f"\n"
            f"threshold: {threshold}  batch_size: {batch_size}  max_chars: {max_chars}",
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

    # --- Pick the text column ----------------------------------------------
    if not text_col:
        for candidate in (
            "text",
            "content",
            "prompt",
            "message",
            "body",
            "source_text",
            "input",
        ):
            if candidate in ds.column_names:
                text_col = candidate
                break
        else:
            raise SystemExit(
                f"No text column found in {ds.column_names}. Set TEXT_COLUMN."
            )
    console.print(f"[dim]text column:[/dim] [bold]{text_col}[/bold]")

    # --- Load GLiNER model --------------------------------------------------
    with console.status(f"[cyan]Loading {model_id}...", spinner="dots"):
        t0 = time.time()
        model = GLiNER.from_pretrained(model_id).to(device)
        model.eval()
    console.print(f"[green]✓[/green] loaded model ({time.time() - t0:.1f}s)")

    # --- Run extraction with live progress ----------------------------------
    n = len(ds)
    pii_entities: list[list[dict]] = []
    total_ents = 0
    label_counts: dict[str, int] = {}
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
        TextColumn("[green]{task.fields[ents]}[/green]"),
        console=console,
        transient=False,
        refresh_per_second=2,
    ) as progress:
        task = progress.add_task(
            "[cyan]extracting PII[/cyan]",
            total=n,
            rate="",
            ents="",
        )
        # GLiNER's batch_predict_entities takes a list of texts.
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            texts = [(t or "")[:max_chars] for t in ds[text_col][start:end]]
            non_empty_idx = [i for i, t in enumerate(texts) if t.strip()]

            try:
                if non_empty_idx:
                    sub_texts = [texts[i] for i in non_empty_idx]
                    sub_preds = model.batch_predict_entities(
                        sub_texts, entity_types, threshold=threshold
                    )
                else:
                    sub_preds = []
            except Exception as e:
                console.print(f"[yellow]warn[/yellow]: batch failed: {e}")
                sub_preds = [[] for _ in non_empty_idx]

            # Re-align predictions to the full batch (empty rows get [])
            batch_out: list[list[dict]] = [[] for _ in range(end - start)]
            for pos, preds in zip(non_empty_idx, sub_preds):
                cleaned = []
                for ent in preds:
                    cleaned.append(
                        {
                            "start": int(ent.get("start", 0)),
                            "end": int(ent.get("end", 0)),
                            "text": str(ent.get("text", "")),
                            "label": str(ent.get("label", "")),
                            "score": float(ent.get("score", 0.0)),
                        }
                    )
                batch_out[pos] = cleaned
                total_ents += len(cleaned)
                for c in cleaned:
                    label_counts[c["label"]] = label_counts.get(c["label"], 0) + 1

            pii_entities.extend(batch_out)

            elapsed = max(time.time() - t0, 1e-6)
            rate = f"{(end) / elapsed:,.1f} rows/s"
            progress.update(
                task,
                advance=(end - start),
                rate=rate,
                ents=f"ents:{total_ents:,}",
            )

    console.print(
        f"[green]✓[/green] scanned [bold]{n:,}[/bold] rows in "
        f"[bold]{time.time() - t0:.1f}s[/bold] — "
        f"found [bold cyan]{total_ents:,}[/bold cyan] PII entities"
    )

    # --- Top-label summary --------------------------------------------------
    if label_counts:
        top = sorted(label_counts.items(), key=lambda kv: kv[1], reverse=True)
        console.print("\n[bold]Top labels:[/bold]")
        for label, count in top[:10]:
            console.print(f"  {label:<28} [cyan]{count:>6,}[/cyan]")

    # --- Build + push output dataset ---------------------------------------
    out = ds.add_column("pii_entities", pii_entities)

    with console.status(f"[cyan]Pushing to {output_id}...", spinner="dots"):
        t0 = time.time()
        out.push_to_hub(
            output_id,
            token=hf_token,
            commit_message=(
                f"Add pii_entities: {model_id} "
                f"(thresh={threshold}, found {total_ents:,} ents over {n:,} rows)"
            ),
        )
    console.print(
        f"[green]✓[/green] pushed [bold]{output_id}[/bold] "
        f"({time.time() - t0:.1f}s)"
    )

    url = f"https://huggingface.co/datasets/{output_id}"
    console.print(
        Panel.fit(
            f"[bold green]Done.[/bold green]\n\n"
            f"{n:,} rows · {total_ents:,} PII entities · "
            f"{len(label_counts)} distinct labels\n"
            f"[link]{url}[/link]",
            title="result",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
