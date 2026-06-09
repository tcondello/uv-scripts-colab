# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=3.0",
#   "transformers>=4.49",
#   "torch>=2.4",
#   "Pillow>=10.0",
#   "huggingface-hub>=0.25",
#   "rich>=13.0",
#   "qwen-vl-utils",
#   "accelerate>=0.30",
# ]
# ///
"""vlm-detect.py — VLM-as-labeller object detection on a managed Colab GPU.

Reads an HF image dataset, runs each image through a vision-language model
with your free-form detection prompt, parses the model's JSON bounding-box
output, and pushes a new dataset back with three columns added:

    detections      list[dict] of {bbox_2d, label, sub_label}
    raw_response    full model output (string) for debugging / audit
    inference_info  JSON-string metadata (model, prompt, image_size, ts)

Modelled on davanstrien's `qwen3vl-detect.py` so the **prompt format and the
output schema are identical** — you can literally copy a prompt from his
recipe (or from a Label Studio / FiftyOne flow) and feed it in here.

The default model is **Qwen2.5-VL-3B-Instruct** (~6 GB at fp16) which fits a
T4 with comfortable headroom. For better quality on bigger hardware override
with `MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct` on an L4/A100, or
`MODEL_ID=Qwen/Qwen2.5-VL-72B-Instruct` on H100. Note: this is qualitatively
weaker than davanstrien's 35B MoE default — useful for triage, prototyping,
and small-scale labelling, not as a drop-in replacement for the big model.

Env vars (or positional args, in order):
    INPUT_DATASET     HF image dataset id (required)
    OUTPUT_DATASET    HF dataset id to push to (required)
    IMAGE_COLUMN      Image column                 [auto: image|img|poster|photo]
    MODEL_ID          VLM identifier               [Qwen/Qwen2.5-VL-3B-Instruct]
    DETECT_PROMPT     Free-form detection prompt   [davanstrien default — see source]
    MAX_NEW_TOKENS    Cap response length          [1024]
    TEMPERATURE       Sampling temperature         [0.0 — deterministic]
    BATCH_SIZE        Images per inference batch   [1]  (Qwen2.5-VL is single-image
                                                          friendly; raise on big GPUs)
    LIMIT             Max rows                     [no limit]
    SPLIT             Dataset split                [train]
    HF_TOKEN          For push                     (required)

Usage:
    INPUT_DATASET=pinecone/movie-posters \\
    OUTPUT_DATASET=you/movie-posters-detections \\
    LIMIT=20 \\
        bin/colab-hf-run recipes/vlm-detect.py

    # With your own prompt (the marquee use case):
    DETECT_PROMPT="Detect every person in the image. Return JSON ..." \\
        bin/colab-hf-run recipes/vlm-detect.py
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import subprocess
import sys
import time


# davanstrien's exact default prompt — same one his Qwen3-VL recipe uses, so
# the output JSON shape matches between repos and downstream tooling.
DEFAULT_DETECT_PROMPT = (
    "Detect every distinct object in the image. For each object, output a JSON "
    "object with keys: bbox_2d (an array of four numbers [x1, y1, x2, y2]), "
    "label (the object category), and sub_label (a short descriptive attribute "
    'or "" if none applies). Return a JSON array of these objects. Example: '
    '[{"bbox_2d": [x1, y1, x2, y2], "label": "car", "sub_label": "red"}].'
)


def _ensure_deps() -> None:
    """Probe for required deps via a SUBPROCESS — importing transformers in
    this process pins the (possibly old) version into sys.modules so a
    later pip-upgrade can't take effect. Also: deliberately don't pass `-U`
    or include `datasets` / `Pillow` / `pyarrow` in the install list —
    cascade-upgrading `pyarrow` on a kernel that already has it loaded
    produces an ABI mismatch (`IpcReadOptions size changed`).
    """
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "from transformers import Qwen2_5_VLForConditionalGeneration; "
            "import qwen_vl_utils, accelerate, rich",
        ],
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        return  # All deps present at sufficient versions.

    print(
        "[deps] installing transformers>=4.49 + qwen-vl-utils + accelerate...",
        flush=True,
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers>=4.49",
            "qwen-vl-utils",
            "accelerate>=0.30",
            "rich>=13.0",
        ]
    )


def _cfg(name: str, *, default: str | None = None, pos: int | None = None) -> str | None:
    if name in os.environ and os.environ[name]:
        return os.environ[name]
    if pos is not None and pos < len(sys.argv) and not sys.argv[pos].endswith(".json"):
        return sys.argv[pos]
    return default


def _parse_detections(text: str) -> list[dict]:
    """Tolerant JSON parser for VLM bounding-box output.

    Modelled after davanstrien's parser. Handles fenced code blocks, comments,
    trailing commas, and falls back to per-object regex extraction if the
    whole-array parse fails.
    """
    if not text:
        return []

    # Strip fenced code blocks
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.replace("```", "")

    # Try whole-array parse first
    for candidate in re.findall(r"\[\s*\{.*?\}\s*\]", cleaned, re.DOTALL):
        try:
            arr = json.loads(candidate)
            if isinstance(arr, list):
                return [_normalize_det(d) for d in arr if isinstance(d, dict)]
        except json.JSONDecodeError:
            continue

    # Per-object fallback: find each {... "bbox_2d": [...] ...} object
    objs: list[dict] = []
    pattern = r'\{[^{}]*"bbox_2d"\s*:\s*\[[\d\s.,\-]+\][^{}]*\}'
    for match in re.findall(pattern, cleaned, re.DOTALL):
        try:
            d = json.loads(match)
            objs.append(_normalize_det(d))
        except json.JSONDecodeError:
            continue
    return objs


def _normalize_det(d: dict) -> dict:
    """Return a single detection in the canonical shape."""
    bbox = d.get("bbox_2d") or d.get("bbox") or []
    try:
        bbox = [float(x) for x in bbox][:4]
    except (TypeError, ValueError):
        bbox = []
    return {
        "bbox_2d": bbox,
        "label": str(d.get("label", "") or ""),
        "sub_label": str(d.get("sub_label", "") or ""),
    }


def main() -> None:
    _ensure_deps()

    import torch
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
    from transformers import AutoProcessor

    # Prefer the explicit Qwen2.5-VL class (most reliable for this model
    # family). Fall back to AutoModelForVision2Seq for other VLMs.
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _VLModel
        _is_qwen25 = True
    except ImportError:
        from transformers import AutoModelForVision2Seq as _VLModel  # type: ignore
        _is_qwen25 = False

    console = Console(force_terminal=True)

    input_id = _cfg("INPUT_DATASET", pos=1)
    output_id = _cfg("OUTPUT_DATASET", pos=2)
    image_col = _cfg("IMAGE_COLUMN", pos=3)
    model_id = _cfg("MODEL_ID", pos=4, default="Qwen/Qwen2.5-VL-3B-Instruct")
    prompt = _cfg("DETECT_PROMPT", default=DEFAULT_DETECT_PROMPT)
    max_new_tokens = int(_cfg("MAX_NEW_TOKENS", default="1024"))
    temperature = float(_cfg("TEMPERATURE", default="0.0"))
    batch_size = int(_cfg("BATCH_SIZE", default="1"))
    limit_raw = _cfg("LIMIT")
    limit = int(limit_raw) if limit_raw else None
    split = _cfg("SPLIT", default="train")
    hf_token = _cfg("HF_TOKEN")

    if not input_id or not output_id:
        console.print(
            "[red]Missing required config.[/red] Set INPUT_DATASET and OUTPUT_DATASET."
        )
        sys.exit(2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    dtype = torch.float16 if device == "cuda" else torch.float32

    console.print(
        Panel.fit(
            f"[bold]vlm-detect[/bold]\n"
            f"input:    [cyan]{input_id}[/cyan] (split={split}"
            + (f", limit={limit:,}" if limit else "")
            + ")\n"
            f"output:   [green]{output_id}[/green]\n"
            f"model:    [magenta]{model_id}[/magenta]\n"
            f"device:   [yellow]{gpu_name}[/yellow] ({dtype})\n"
            f"prompt:   {prompt[:90]}"
            + ("…" if len(prompt) > 90 else "")
            + f"\n"
            f"max_new_tokens: {max_new_tokens}  temperature: {temperature}",
            title="config",
            border_style="blue",
        )
    )

    # --- Load dataset --------------------------------------------------------
    from datasets import Image as HfImage

    with console.status(f"[cyan]Loading {input_id} (split={split})...", spinner="dots"):
        t0 = time.time()
        ds = load_dataset(input_id, split=split)
    if limit and len(ds) > limit:
        ds = ds.select(range(limit))
    console.print(
        f"[green]✓[/green] loaded [bold]{len(ds):,}[/bold] rows "
        f"({time.time() - t0:.1f}s), columns: {ds.column_names}"
    )

    if not image_col:
        for candidate in ("image", "img", "poster", "photo", "url", "image_url"):
            if candidate in ds.column_names:
                image_col = candidate
                break
        else:
            raise SystemExit(f"No image column in {ds.column_names}. Set IMAGE_COLUMN.")

    # Cast a *copy* for embedding so HF lazily downloads URL columns; preserve
    # original schema on push (same trick as clip-embed.py).
    feat = ds.features.get(image_col)
    if feat.__class__.__name__ != "Image":
        console.print(f"[dim]Casting {image_col} to Image feature for lazy fetch.[/dim]")
        ds_for_infer = ds.cast_column(image_col, HfImage())
    else:
        ds_for_infer = ds
    console.print(f"[dim]image column:[/dim] [bold]{image_col}[/bold]")

    # --- Load VLM ------------------------------------------------------------
    with console.status(f"[cyan]Loading {model_id}...", spinner="dots"):
        t0 = time.time()
        processor = AutoProcessor.from_pretrained(model_id)
        model = _VLModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
        )
        model.eval()
    console.print(
        f"[green]✓[/green] loaded model ({time.time() - t0:.1f}s) — "
        f"class=[bold]{_VLModel.__name__}[/bold]"
    )

    # --- Run inference -------------------------------------------------------
    n = len(ds_for_infer)
    detections_col: list[list[dict]] = [[] for _ in range(n)]
    raw_responses: list[str] = [""] * n
    inference_info_col: list[str] = [""] * n
    skipped = 0
    total_det = 0
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
        TextColumn("[green]{task.fields[det]}[/green]"),
        console=console,
        transient=False,
        refresh_per_second=2,
    ) as progress:
        task = progress.add_task(
            "[cyan]detecting[/cyan]",
            total=n,
            rate="",
            det="",
        )
        for i in range(n):
            try:
                pil_img = ds_for_infer[i][image_col]
                if pil_img is None:
                    skipped += 1
                    progress.update(task, advance=1)
                    continue
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=[text],
                    images=[pil_img],
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                with torch.no_grad():
                    gen_kwargs = dict(
                        max_new_tokens=max_new_tokens,
                        do_sample=(temperature > 0),
                    )
                    if temperature > 0:
                        gen_kwargs["temperature"] = temperature
                    generated = model.generate(**inputs, **gen_kwargs)

                # Strip prompt tokens from the generated output
                trimmed = generated[:, inputs.input_ids.shape[1] :]
                response = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
                raw_responses[i] = response

                dets = _parse_detections(response)
                detections_col[i] = dets
                total_det += len(dets)

                inference_info_col[i] = json.dumps(
                    {
                        "model_id": model_id,
                        "prompt": prompt,
                        "image_size": [pil_img.width, pil_img.height],
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                    }
                )
            except Exception as e:
                skipped += 1
                raw_responses[i] = f"[ERROR] {e}"

            elapsed = max(time.time() - t0, 1e-6)
            progress.update(
                task,
                advance=1,
                rate=f"{(i + 1) / elapsed:,.2f} img/s",
                det=f"det:{total_det}",
            )

    ok = n - skipped
    console.print(
        f"[green]✓[/green] processed [bold]{ok:,}/{n:,}[/bold] images in "
        f"[bold]{time.time() - t0:.1f}s[/bold] — "
        f"found [bold cyan]{total_det:,}[/bold cyan] detections "
        + (f"([red]{skipped} skipped[/red])" if skipped else "")
    )

    # --- Build + push output dataset ----------------------------------------
    out = ds.add_column("detections", detections_col)
    out = out.add_column("raw_response", raw_responses)
    out = out.add_column("inference_info", inference_info_col)

    with console.status(f"[cyan]Pushing to {output_id}...", spinner="dots"):
        t0 = time.time()
        out.push_to_hub(
            output_id,
            token=hf_token,
            commit_message=(
                f"Add VLM detections: {model_id} "
                f"({total_det:,} detections over {ok:,} images)"
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
            f"{ok:,} images · {total_det:,} detections · "
            f"model={model_id.split('/')[-1]}\n"
            f"[link]{url}[/link]",
            title="result",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
