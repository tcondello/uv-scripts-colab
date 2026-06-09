# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=3.0",
#   "huggingface-hub>=0.25",
#   "rich>=13.0",
#   "numpy>=1.24",
#   "Pillow>=10.0",
# ]
# ///
"""dataset-stats.py — profile a HF dataset, publish a stats report.

Walks every column of an HF dataset, dispatches stats by dtype
(numeric → percentiles, string → length distribution + samples,
class label → value counts, image/audio → format + dimension/duration,
sequence → length distribution), prints a Rich table to your terminal,
and pushes back a new dataset whose README.md *is* the stats report —
plus a `column_stats` table for programmatic queries.

CPU-only (no GPU needed). The same wrapper still works — just runs
faster because there's nothing to put on the device.

Env vars (or positional args):
    INPUT_DATASET    HF dataset id to profile (required)
    OUTPUT_DATASET   HF dataset id where the report lands (required)
    SPLIT            Split to profile             [train]
    LIMIT            Sample rows for stats        [10000, 0 = whole split]
    TOP_N            Top-K values to keep per column [10]
    HF_TOKEN         For push                     (required)

Usage:
    INPUT_DATASET=stanfordnlp/sst2 \\
    OUTPUT_DATASET=you/sst2-stats \\
        bin/colab-hf-run recipes/dataset-stats.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter


def _ensure_deps() -> None:
    try:
        import datasets  # noqa: F401
        import numpy  # noqa: F401
        import rich  # noqa: F401
        import PIL  # noqa: F401
    except ImportError:
        print("[deps] installing datasets + numpy + Pillow + rich...", flush=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "datasets>=3.0",
                "numpy>=1.24",
                "Pillow>=10.0",
                "rich>=13.0",
            ]
        )


def _cfg(name: str, *, default: str | None = None, pos: int | None = None) -> str | None:
    if name in os.environ and os.environ[name]:
        return os.environ[name]
    if pos is not None and pos < len(sys.argv) and not sys.argv[pos].endswith(".json"):
        return sys.argv[pos]
    return default


def _fmt_num(v: float | int | None) -> str:
    if v is None:
        return "—"
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:,.2f}"


def _profile_numeric(values: list) -> dict:
    import numpy as np

    arr = np.array([v for v in values if v is not None], dtype="float64")
    if arr.size == 0:
        return {"kind": "numeric", "n": 0}
    pcts = np.percentile(arr, [1, 25, 50, 75, 99]).tolist()
    return {
        "kind": "numeric",
        "n": int(arr.size),
        "min": float(arr.min()),
        "p1": pcts[0],
        "p25": pcts[1],
        "p50": pcts[2],
        "p75": pcts[3],
        "p99": pcts[4],
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "distinct": int(np.unique(arr).size),
    }


def _profile_string(values: list, top_n: int = 10) -> dict:
    import numpy as np

    strs = [v for v in values if v is not None and v != ""]
    if not strs:
        return {"kind": "string", "n": 0}
    lens = np.array([len(s) for s in strs])
    counter = Counter(strs)
    top = counter.most_common(top_n)
    # Pull a couple of representative samples (shortest, median, longest)
    sorted_by_len = sorted(strs, key=len)
    samples = {
        "shortest": sorted_by_len[0][:200],
        "median": sorted_by_len[len(sorted_by_len) // 2][:200],
        "longest": sorted_by_len[-1][:200],
    }
    return {
        "kind": "string",
        "n": len(strs),
        "distinct": len(counter),
        "len_min": int(lens.min()),
        "len_p50": int(np.percentile(lens, 50)),
        "len_mean": float(lens.mean()),
        "len_p99": int(np.percentile(lens, 99)),
        "len_max": int(lens.max()),
        "top_values": [{"value": str(v)[:100], "count": c} for v, c in top],
        "samples": samples,
    }


def _profile_classlabel(values: list, names: list[str], top_n: int = 10) -> dict:
    counter = Counter([v for v in values if v is not None])
    total = sum(counter.values())
    return {
        "kind": "class_label",
        "n": total,
        "num_classes": len(names),
        "value_counts": [
            {
                "id": int(v),
                "name": names[int(v)] if 0 <= int(v) < len(names) else str(v),
                "count": int(c),
                "pct": round(100 * c / total, 2) if total else 0.0,
            }
            for v, c in counter.most_common(top_n)
        ],
    }


def _profile_sequence(values: list) -> dict:
    import numpy as np

    lens = [len(v) for v in values if v is not None and hasattr(v, "__len__")]
    if not lens:
        return {"kind": "sequence", "n": 0}
    arr = np.array(lens)
    return {
        "kind": "sequence",
        "n": len(lens),
        "len_min": int(arr.min()),
        "len_p50": int(np.percentile(arr, 50)),
        "len_mean": float(arr.mean()),
        "len_p99": int(np.percentile(arr, 99)),
        "len_max": int(arr.max()),
    }


def _profile_image(values: list) -> dict:
    """Stats for HF Image feature (PIL images)."""
    import numpy as np

    widths, heights = [], []
    formats: Counter = Counter()
    n_ok = 0
    for img in values:
        if img is None:
            continue
        try:
            widths.append(img.width)
            heights.append(img.height)
            formats[getattr(img, "format", None) or "?"] += 1
            n_ok += 1
        except Exception:
            pass
    if not n_ok:
        return {"kind": "image", "n": 0}
    return {
        "kind": "image",
        "n": n_ok,
        "width_p50": int(np.percentile(widths, 50)),
        "width_p99": int(np.percentile(widths, 99)),
        "height_p50": int(np.percentile(heights, 50)),
        "height_p99": int(np.percentile(heights, 99)),
        "formats": dict(formats),
    }


def _profile_audio(values: list) -> dict:
    """Stats for HF Audio feature: sampling rates + durations."""
    import numpy as np

    srs: Counter = Counter()
    durs = []
    n_ok = 0
    for a in values:
        if a is None:
            continue
        try:
            sr = a.get("sampling_rate")
            arr = a.get("array")
            if sr and arr is not None:
                srs[int(sr)] += 1
                durs.append(len(arr) / sr)
                n_ok += 1
        except Exception:
            pass
    if not n_ok:
        return {"kind": "audio", "n": 0}
    durs_arr = np.array(durs)
    return {
        "kind": "audio",
        "n": n_ok,
        "sample_rates": dict(srs),
        "duration_min": float(durs_arr.min()),
        "duration_p50": float(np.percentile(durs_arr, 50)),
        "duration_mean": float(durs_arr.mean()),
        "duration_p99": float(np.percentile(durs_arr, 99)),
        "duration_max": float(durs_arr.max()),
        "total_hours": float(durs_arr.sum() / 3600),
    }


def main() -> None:
    _ensure_deps()

    from datasets import Audio, ClassLabel, Dataset, Image, Sequence, Value, load_dataset
    from huggingface_hub import HfApi
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console(force_terminal=True)

    input_id = _cfg("INPUT_DATASET", pos=1)
    output_id = _cfg("OUTPUT_DATASET", pos=2)
    split = _cfg("SPLIT", default="train")
    limit_raw = _cfg("LIMIT", default="10000")
    limit = int(limit_raw)  # 0 = no limit
    top_n = int(_cfg("TOP_N", default="10"))
    hf_token = _cfg("HF_TOKEN")

    if not input_id or not output_id:
        console.print("[red]Set INPUT_DATASET and OUTPUT_DATASET.[/red]")
        sys.exit(2)

    console.print(
        Panel.fit(
            f"[bold]dataset-stats[/bold]\n"
            f"input:    [cyan]{input_id}[/cyan] (split={split}"
            + (f", limit={limit:,}" if limit else ", whole split")
            + f")\n"
            f"output:   [green]{output_id}[/green]\n"
            f"top_n:    {top_n} (per-column samples / value counts)",
            title="config",
            border_style="blue",
        )
    )

    # --- Load ---------------------------------------------------------------
    with console.status(f"[cyan]Loading {input_id} (split={split})...", spinner="dots"):
        t0 = time.time()
        ds = load_dataset(input_id, split=split)
    full_n = len(ds)
    if limit and full_n > limit:
        console.print(
            f"[dim]Sampling [bold]{limit:,}[/bold] of {full_n:,} rows for stats "
            f"(set LIMIT=0 to scan the whole split).[/dim]"
        )
        ds_sample = ds.select(range(limit))
    else:
        ds_sample = ds
    console.print(
        f"[green]✓[/green] loaded {full_n:,} rows in {time.time() - t0:.1f}s; "
        f"profiling {len(ds_sample):,}"
    )

    # --- Walk columns -------------------------------------------------------
    features = ds_sample.features
    column_stats = []
    t0 = time.time()
    with console.status("[cyan]Profiling columns...", spinner="dots"):
        for col, feat in features.items():
            values = ds_sample[col]
            n_total = len(values)
            n_null = sum(1 for v in values if v is None)

            # Dispatch by feature type
            stats: dict
            if isinstance(feat, ClassLabel):
                stats = _profile_classlabel(values, feat.names, top_n=top_n)
            elif isinstance(feat, Image):
                stats = _profile_image(values)
            elif isinstance(feat, Audio):
                stats = _profile_audio(values)
            elif isinstance(feat, Sequence):
                stats = _profile_sequence(values)
            elif isinstance(feat, Value):
                # Numeric or string
                if feat.dtype in (
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                    "float16",
                    "float32",
                    "float64",
                ):
                    stats = _profile_numeric(values)
                elif feat.dtype == "string":
                    stats = _profile_string(values, top_n=top_n)
                elif feat.dtype == "bool":
                    counter = Counter([bool(v) for v in values if v is not None])
                    stats = {
                        "kind": "bool",
                        "n": sum(counter.values()),
                        "value_counts": [
                            {"value": str(k), "count": v} for k, v in counter.most_common()
                        ],
                    }
                else:
                    stats = {"kind": f"value({feat.dtype})", "n": n_total - n_null}
            else:
                # Dict-like / struct / unsupported
                stats = {"kind": feat.__class__.__name__, "n": n_total - n_null}

            column_stats.append(
                {
                    "column": col,
                    "feature_type": feat.__class__.__name__
                    + (f"({feat.dtype})" if isinstance(feat, Value) else ""),
                    "kind": stats.get("kind"),
                    "n_total": n_total,
                    "n_null": n_null,
                    "null_pct": round(100 * n_null / n_total, 2) if n_total else 0,
                    "stats_json": json.dumps(stats, ensure_ascii=False, default=str),
                }
            )
    console.print(
        f"[green]✓[/green] profiled {len(column_stats)} columns ({time.time() - t0:.1f}s)"
    )

    # --- Print summary table to terminal ------------------------------------
    table = Table(
        title=f"\n{input_id} · {split} · {len(ds_sample):,} rows profiled",
        show_lines=False,
    )
    table.add_column("column", style="cyan")
    table.add_column("type", style="magenta")
    table.add_column("kind", style="green")
    table.add_column("null %", justify="right", style="yellow")
    table.add_column("highlights", style="dim")

    for s in column_stats:
        stats = json.loads(s["stats_json"])
        kind = stats.get("kind", "?")
        if kind == "numeric":
            hl = f"min={_fmt_num(stats.get('min'))}  p50={_fmt_num(stats.get('p50'))}  max={_fmt_num(stats.get('max'))}  distinct={stats.get('distinct')}"
        elif kind == "string":
            hl = f"distinct={stats.get('distinct')}  len p50={stats.get('len_p50')}  max={stats.get('len_max')}"
        elif kind == "class_label":
            top = stats.get("value_counts", [])[:3]
            hl = "  ".join(f"{v['name']}={v['count']}" for v in top)
        elif kind == "image":
            hl = f"w p50={stats.get('width_p50')}  h p50={stats.get('height_p50')}  fmts={list(stats.get('formats',{}).keys())}"
        elif kind == "audio":
            hl = f"hours={stats.get('total_hours',0):.2f}  dur p50={stats.get('duration_p50',0):.1f}s  srs={list(stats.get('sample_rates',{}).keys())}"
        elif kind == "sequence":
            hl = f"len p50={stats.get('len_p50')}  max={stats.get('len_max')}"
        elif kind == "bool":
            hl = "  ".join(f"{v['value']}={v['count']}" for v in stats.get("value_counts", []))
        else:
            hl = "—"

        table.add_row(
            s["column"],
            s["feature_type"],
            kind or "?",
            f"{s['null_pct']:.1f}%",
            hl,
        )

    console.print(table)

    # --- Build markdown report ----------------------------------------------
    md_lines = [
        f"# Stats for `{input_id}`",
        "",
        f"Generated by [`dataset-stats.py`](https://github.com/tcondello/uv-scripts-colab/blob/main/recipes/dataset-stats.py) over the `{split}` split.",
        "",
        f"- **Total rows in split:** {full_n:,}",
        f"- **Rows profiled:** {len(ds_sample):,}",
        f"- **Columns:** {len(column_stats)}",
        "",
        "## Column overview",
        "",
        "| column | type | kind | null % | highlights |",
        "|---|---|---|---:|---|",
    ]
    for s in column_stats:
        stats = json.loads(s["stats_json"])
        kind = stats.get("kind", "?")
        if kind == "numeric":
            hl = f"min={_fmt_num(stats.get('min'))} · p50={_fmt_num(stats.get('p50'))} · max={_fmt_num(stats.get('max'))} · distinct={stats.get('distinct')}"
        elif kind == "string":
            hl = f"distinct={stats.get('distinct'):,} · len p50={stats.get('len_p50')} · max={stats.get('len_max')}"
        elif kind == "class_label":
            top = stats.get("value_counts", [])[:3]
            hl = " · ".join(f"`{v['name']}`={v['count']}" for v in top)
        elif kind == "image":
            hl = f"w p50={stats.get('width_p50')} · h p50={stats.get('height_p50')} · fmts={list(stats.get('formats',{}).keys())}"
        elif kind == "audio":
            hl = f"hours={stats.get('total_hours',0):.2f} · dur p50={stats.get('duration_p50',0):.1f}s"
        elif kind == "sequence":
            hl = f"len p50={stats.get('len_p50')} · max={stats.get('len_max')}"
        elif kind == "bool":
            hl = " · ".join(f"{v['value']}={v['count']}" for v in stats.get("value_counts", []))
        else:
            hl = "—"
        md_lines.append(
            f"| `{s['column']}` | `{s['feature_type']}` | {kind} | {s['null_pct']:.1f}% | {hl} |"
        )

    md_lines += ["", "## Per-column detail", ""]
    for s in column_stats:
        stats = json.loads(s["stats_json"])
        md_lines.append(f"### `{s['column']}`")
        md_lines.append("")
        md_lines.append(
            f"- **type:** `{s['feature_type']}`  •  **kind:** {stats.get('kind')}  •  "
            f"**null:** {s['n_null']:,} / {s['n_total']:,} ({s['null_pct']:.1f}%)"
        )
        md_lines.append("")
        md_lines.append("```json")
        md_lines.append(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
        md_lines.append("```")
        md_lines.append("")

    md_lines += [
        "## Reproduce",
        "",
        "```bash",
        f"INPUT_DATASET={input_id} \\",
        f"OUTPUT_DATASET={output_id} \\",
        f"SPLIT={split} LIMIT={limit} \\",
        "  bin/colab-hf-run recipes/dataset-stats.py",
        "```",
        "",
        "## Machine-readable",
        "",
        "Per-column rows live in the dataset's `train` split: one row per column with `column`, `feature_type`, `kind`, `n_total`, `n_null`, `null_pct`, and `stats_json` (full per-kind detail).",
    ]
    report_md = "\n".join(md_lines)

    # --- Push structured stats dataset + README ----------------------------
    stats_ds = Dataset.from_list(column_stats)
    with console.status(f"[cyan]Pushing to {output_id}...", spinner="dots"):
        t0 = time.time()
        stats_ds.push_to_hub(
            output_id,
            token=hf_token,
            commit_message=f"Stats for {input_id} / {split} ({len(ds_sample):,} rows profiled)",
        )
        # Overwrite the default auto-generated README with our report
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=report_md.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=output_id,
            repo_type="dataset",
            commit_message="Add stats report (README.md)",
        )
    console.print(
        f"[green]✓[/green] pushed [bold]{output_id}[/bold] "
        f"({time.time() - t0:.1f}s)"
    )

    url = f"https://huggingface.co/datasets/{output_id}"
    console.print(
        Panel.fit(
            f"[bold green]Done.[/bold green]\n\n"
            f"{len(column_stats)} columns profiled · {len(ds_sample):,} rows scanned\n"
            f"[link]{url}[/link]",
            title="result",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
