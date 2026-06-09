# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets[audio]>=3.0",
#   "transformers>=4.45",
#   "torch>=2.4",
#   "torchaudio",
#   "librosa",
#   "soundfile",
#   "huggingface-hub>=0.25",
#   "rich>=13.0",
# ]
# ///
"""whisper-transcribe.py — transcribe an HF audio dataset on a Colab GPU.

Reads a Hugging Face audio dataset, runs OpenAI Whisper over each clip on
GPU, and pushes back a new dataset with `transcription` (and `language`,
if multilingual) columns added. Original columns are preserved so the
output composes with whatever you do next.

Whisper-large-v3 (~3 GB) sits well within T4. For very long audio
(>30 s), the recipe uses chunking via the ASR pipeline's
`chunk_length_s` so memory stays bounded.

Env vars (or positional args, in order):
    INPUT_DATASET       HF audio dataset id (required)
    OUTPUT_DATASET      HF dataset id to push to (required)
    AUDIO_COLUMN        Audio column                 [auto: audio|wav|speech|sound]
    MODEL_ID            Whisper variant              [openai/whisper-large-v3]
    WHISPER_LANGUAGE    Force a language (e.g. english, french, japanese)
                        Default: english. Whisper accepts language *names*,
                        not ISO codes. Do NOT use plain `LANGUAGE` — that's
                        a system locale var the Colab kernel pre-sets to
                        `en_US:en`, which Whisper rejects.
    TASK                transcribe | translate       [transcribe]
    CHUNK_LENGTH_S      Chunk for long audio         [0 = disabled]
    BATCH_SIZE          Clips per GPU batch          [8]
    LIMIT               Max rows                     [no limit]
    SPLIT               Dataset split                [train]
    HF_TOKEN            For push                     (required)

Usage:
    INPUT_DATASET=hf-internal-testing/librispeech_asr_dummy \\
    OUTPUT_DATASET=you/librispeech-whisper-transcripts \\
        bin/colab-hf-run recipes/whisper-transcribe.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time


def _ensure_deps() -> None:
    """Install deps not in the Colab base image."""
    try:
        import datasets  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import librosa  # noqa: F401
        import soundfile  # noqa: F401
        import rich  # noqa: F401
    except ImportError:
        print("[deps] installing transformers + audio deps...", flush=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "datasets[audio]>=3.0",
                "transformers>=4.45",
                "librosa",
                "soundfile",
                "rich>=13.0",
            ]
        )


def _cfg(name: str, *, default: str | None = None, pos: int | None = None) -> str | None:
    if name in os.environ and os.environ[name]:
        return os.environ[name]
    if pos is not None and pos < len(sys.argv) and not sys.argv[pos].endswith(".json"):
        return sys.argv[pos]
    return default


def main() -> None:
    _ensure_deps()

    import torch
    from datasets import Audio, load_dataset
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
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    console = Console(force_terminal=True)

    input_id = _cfg("INPUT_DATASET", pos=1)
    output_id = _cfg("OUTPUT_DATASET", pos=2)
    audio_col = _cfg("AUDIO_COLUMN", pos=3)
    model_id = _cfg("MODEL_ID", pos=4, default="openai/whisper-large-v3")
    # Whisper wants language NAMES like "english", "french", "japanese" — not
    # ISO codes. We deliberately do NOT read `LANGUAGE`: that's a standard
    # POSIX locale env var (the Colab kernel pre-sets it to `en_US:en`) and
    # Whisper rejects it.
    language = _cfg("WHISPER_LANGUAGE", default="english")
    task = _cfg("TASK", default="transcribe")  # 'transcribe' or 'translate'
    # 0 disables chunking (default). Set to e.g. 30 if you have long audio
    # (>30s) — note the transformers warning that this is "very experimental"
    # for seq2seq, and that chunked pipelines can mis-detect language codes
    # (we saw 'Unsupported language: en_us' on librispeech_asr_dummy).
    chunk_length_s = int(_cfg("CHUNK_LENGTH_S", default="0"))
    batch_size = int(_cfg("BATCH_SIZE", default="8"))
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
            f"[bold]whisper-transcribe[/bold]\n"
            f"input:    [cyan]{input_id}[/cyan] (split={split}"
            + (f", limit={limit:,}" if limit else "")
            + ")\n"
            f"output:   [green]{output_id}[/green]\n"
            f"model:    [magenta]{model_id}[/magenta]\n"
            f"device:   [yellow]{gpu_name}[/yellow] ({dtype})\n"
            f"task:     {task}  language: {language}\n"
            f"batch:    {batch_size}  chunk_length_s: {chunk_length_s}",
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

    # --- Pick audio column + ensure 16kHz sampling --------------------------
    if not audio_col:
        for candidate in ("audio", "wav", "speech", "sound", "voice"):
            if candidate in ds.column_names:
                audio_col = candidate
                break
        else:
            raise SystemExit(f"No audio column in {ds.column_names}. Set AUDIO_COLUMN.")
    console.print(f"[dim]audio column:[/dim] [bold]{audio_col}[/bold]")

    # Whisper wants 16 kHz. cast_column with Audio(sampling_rate=16000) resamples
    # on access — cheap.
    ds_for_asr = ds.cast_column(audio_col, Audio(sampling_rate=16000))

    # --- Load Whisper directly (not via pipeline) ---------------------------
    # The transformers ASR pipeline has a known issue where datasets' locale
    # tags (e.g. `en_us` from librispeech-style metadata) get fed into the
    # generator and Whisper rejects them. Using the processor + model
    # directly with explicit `forced_decoder_ids` gives us deterministic
    # control over language + task.
    with console.status(f"[cyan]Loading {model_id}...", spinner="dots"):
        t0 = time.time()
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device)
        model.eval()
    lang_for_prompt = language  # used in result column
    console.print(
        f"[green]✓[/green] loaded model ({time.time() - t0:.1f}s) — "
        f"forced language=[bold]{lang_for_prompt}[/bold], task=[bold]{task}[/bold]"
    )

    # --- Transcribe ---------------------------------------------------------
    n = len(ds_for_asr)
    transcriptions: list[str] = [""] * n
    detected_langs: list[str | None] = [None] * n
    total_seconds = 0.0
    failed = 0
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
        TextColumn("[yellow]{task.fields[rtf]}[/yellow]"),
        console=console,
        transient=False,
        refresh_per_second=2,
    ) as progress:
        task_id = progress.add_task(
            "[cyan]transcribing[/cyan]",
            total=n,
            rate="",
            rtf="",
        )
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            audio_inputs = []
            durations = []
            for i in range(start, end):
                try:
                    a = ds_for_asr[i][audio_col]
                    audio_inputs.append({"array": a["array"], "sampling_rate": a["sampling_rate"]})
                    durations.append(len(a["array"]) / a["sampling_rate"])
                except Exception as e:
                    console.print(f"[yellow]warn[/yellow]: row {i}: {e}")
                    audio_inputs.append(None)
                    durations.append(0.0)

            valid = [(idx, a) for idx, a in enumerate(audio_inputs) if a is not None]
            if not valid:
                failed += end - start
                progress.update(task_id, advance=(end - start))
                continue

            try:
                # Pass raw arrays + sampling_rate; processor handles batching
                # + padding + mel-spectrogram conversion.
                inputs = processor(
                    [a["array"] for _, a in valid],
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                ).to(device)
                input_features = inputs.input_features
                if dtype == torch.float16:
                    input_features = input_features.half()

                with torch.no_grad():
                    # Modern transformers Whisper API: pass `language` + `task`
                    # directly to generate(). The old `forced_decoder_ids` route
                    # conflicts with `max_new_tokens` and is deprecated.
                    generated_ids = model.generate(
                        input_features,
                        language=language,
                        task=task,
                        max_new_tokens=440,
                    )
                texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            except Exception as e:
                console.print(f"[red]error[/red] in batch {start}-{end}: {e}")
                failed += end - start
                progress.update(task_id, advance=(end - start))
                continue

            for (local_idx, _), text in zip(valid, texts):
                global_idx = start + local_idx
                transcriptions[global_idx] = (text or "").strip()
                detected_langs[global_idx] = lang_for_prompt

            total_seconds += sum(durations)
            elapsed = max(time.time() - t0, 1e-6)
            rtf = total_seconds / elapsed if total_seconds > 0 else 0.0
            progress.update(
                task_id,
                advance=(end - start),
                rate=f"{(end) / elapsed:,.1f} rows/s",
                rtf=f"{rtf:,.1f}x realtime" if rtf > 0 else "",
            )

    console.print(
        f"[green]✓[/green] transcribed [bold]{n - failed:,}/{n:,}[/bold] clips in "
        f"[bold]{time.time() - t0:.1f}s[/bold] "
        + (f"([red]{failed} failed[/red])" if failed else "")
        + f" — {total_seconds:.0f}s audio, "
        f"[cyan]{total_seconds / max(time.time() - t0, 1e-6):.1f}x realtime[/cyan]"
    )

    # --- Build output dataset and push --------------------------------------
    out = ds.add_column("transcription", transcriptions)
    if any(detected_langs):
        out = out.add_column("language", [l or "" for l in detected_langs])

    with console.status(f"[cyan]Pushing to {output_id}...", spinner="dots"):
        t0 = time.time()
        out.push_to_hub(
            output_id,
            token=hf_token,
            commit_message=f"Add transcriptions: {model_id} ({n - failed:,}/{n:,} clips)",
        )
    console.print(
        f"[green]✓[/green] pushed [bold]{output_id}[/bold] "
        f"({time.time() - t0:.1f}s)"
    )

    url = f"https://huggingface.co/datasets/{output_id}"
    console.print(
        Panel.fit(
            f"[bold green]Done.[/bold green]\n\n"
            f"{n - failed:,} clips · {total_seconds:.0f}s audio · "
            f"model={model_id.split('/')[-1]}\n"
            f"[link]{url}[/link]",
            title="result",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
