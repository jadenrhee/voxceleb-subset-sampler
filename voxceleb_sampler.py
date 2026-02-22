#!/usr/bin/env python3
"""Shard-aware VoxCeleb subset sampler for HF WebDataset mirrors.

This script is optimized for datasets like:
  gaunernst/voxceleb2-dev-wds

It is intentionally split into three phases:
1) Streaming metadata phase:
   - Stream rows via `load_dataset(..., streaming=True)`.
   - Build lightweight candidate metadata (speaker, relpath, seconds, bytes, shard info).
2) Selection phase:
   - Diversity-first selection with deterministic seeded randomness.
   - Phase 1: one utterance per speaker where budget allows.
   - Phase 2: round-robin fill across speakers with per-speaker cap.
3) Shard-grouped materialization phase:
   - Group selected samples by shard.
   - Download each shard at most once.
   - Extract only selected members from each tar shard.

No full-dataset download is required.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse


EPS = 1e-9
BYTES_PER_GB = 1024 ** 3


@dataclass
class Candidate:
    """One selectable utterance candidate from streaming metadata."""

    speaker_id: str
    relpath: str
    seconds: float
    bytes: int
    shard_id: str
    shard_filename: Optional[str]
    shard_url: Optional[str]
    member_name: str


def candidate_to_dict(item: Candidate) -> Dict[str, object]:
    """Serialize candidate for index persistence."""
    return {
        "speaker_id": item.speaker_id,
        "relpath": item.relpath,
        "seconds": item.seconds,
        "bytes": item.bytes,
        "shard_id": item.shard_id,
        "shard_filename": item.shard_filename,
        "shard_url": item.shard_url,
        "member_name": item.member_name,
    }


def candidate_from_dict(payload: Dict[str, object]) -> Candidate:
    """Deserialize candidate from index row."""
    return Candidate(
        speaker_id=str(payload["speaker_id"]),
        relpath=str(payload["relpath"]),
        seconds=float(payload["seconds"]),
        bytes=int(payload["bytes"]),
        shard_id=str(payload["shard_id"]),
        shard_filename=(str(payload["shard_filename"]) if payload.get("shard_filename") else None),
        shard_url=(str(payload["shard_url"]) if payload.get("shard_url") else None),
        member_name=str(payload["member_name"]),
    )


def save_index(path: Path, speaker_to_candidates: Dict[str, List[Candidate]]) -> None:
    """Write newline-delimited JSON index for fast future runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for speaker_id in sorted(speaker_to_candidates.keys()):
            for item in speaker_to_candidates[speaker_id]:
                f.write(json.dumps(candidate_to_dict(item), sort_keys=True))
                f.write("\n")


def load_index(path: Path) -> Dict[str, List[Candidate]]:
    """Load newline-delimited JSON index produced by save_index()."""
    speaker_to_candidates: Dict[str, List[Candidate]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    continue
                item = candidate_from_dict(payload)
            except Exception as exc:
                raise SystemExit(f"Error: malformed index at line {line_no}: {exc}") from exc
            speaker_to_candidates[item.speaker_id].append(item)
    return dict(speaker_to_candidates)


def eprint(message: str) -> None:
    """Write progress logs to stderr."""
    print(message, file=sys.stderr)


def normalize_extensions(raw_exts: Sequence[str]) -> Set[str]:
    """Normalize extension tokens into lower-case dotted set."""
    exts: Set[str] = set()
    for token in raw_exts:
        for piece in token.split(","):
            ext = piece.strip().lower()
            if not ext:
                continue
            if not ext.startswith("."):
                ext = f".{ext}"
            exts.add(ext)
    return exts


def safe_float(value: object) -> Optional[float]:
    """Parse positive float or return None."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def safe_int(value: object) -> Optional[int]:
    """Parse positive int or return None."""
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def infer_duration_from_wav_bytes(blob: bytes) -> Optional[float]:
    """Get WAV duration from header bytes without full audio decoding."""
    try:
        with wave.open(io.BytesIO(blob), "rb") as wav_file:
            framerate = wav_file.getframerate()
            frames = wav_file.getnframes()
            if framerate <= 0:
                return None
            duration = frames / float(framerate)
            if duration <= 0:
                return None
            return duration
    except Exception:
        return None


def parse_resolve_url_to_filename(url: str) -> Optional[str]:
    """Extract HF repo-relative filename from a resolve URL.

    Example:
      .../datasets/<repo>/resolve/main/path/to/shard.tar -> path/to/shard.tar
    """
    parsed = urlparse(url)
    path = parsed.path
    marker = "/resolve/"
    idx = path.find(marker)
    if idx < 0:
        return None
    tail = path[idx + len(marker) :]
    parts = [p for p in tail.split("/") if p]
    if len(parts) < 2:
        return None
    # First token is revision (main/tag/commit), remainder is filename.
    return "/".join(parts[1:])


def parse_hf_dataset_uri(uri: str) -> Optional[Tuple[str, str, str]]:
    """Parse `hf://datasets/<repo>@<revision>/<path>` URIs.

    Returns:
      (repo_id, revision, filename)
    """
    if not uri.startswith("hf://datasets/"):
        return None
    tail = uri[len("hf://datasets/") :]
    if "@" in tail:
        # Preferred form: hf://datasets/<namespace>/<repo>@<revision>/<filename>
        repo_part, rest = tail.split("@", 1)
        if "/" not in rest:
            return None
        revision, filename = rest.split("/", 1)
        repo_id = repo_part
    else:
        # Fallback form: hf://datasets/<namespace>/<repo>/<filename>
        parts = [p for p in tail.split("/") if p]
        if len(parts) < 3:
            return None
        repo_id = "/".join(parts[:2])
        revision = "main"
        filename = "/".join(parts[2:])

    if not repo_id or not filename:
        return None
    return repo_id, revision or "main", filename


def infer_speaker_from_relpath(relpath: str) -> Optional[str]:
    """Infer VoxCeleb speaker id from first path segment."""
    parts = [part for part in relpath.split("/") if part]
    if not parts:
        return None
    return parts[0]


def normalize_relpath(path: str) -> str:
    """Normalize to portable forward-slash path."""
    return str(PurePosixPath(path))


def select_audio_field(example: Dict[str, object], preferred: str) -> Optional[str]:
    """Pick which column in a row contains audio payload/metadata."""
    if preferred in example:
        return preferred
    for key in ("wav", "audio", "flac", "mp3", "m4a"):
        if key in example:
            return key
    return None


# ---------------------------------------------------------------------------
# Phase 1: streaming metadata phase
# ---------------------------------------------------------------------------
def stream_metadata(args: argparse.Namespace) -> Tuple[Dict[str, List[Candidate]], Dict[str, int]]:
    """Stream HF rows and produce per-speaker candidate lists.

    This phase intentionally avoids full-audio decoding. It uses metadata fields,
    raw WAV headers, or lightweight bytes length where available.
    """
    if not args.hf_http_debug:
        # Keep external library logs concise unless explicitly requested.
        os.environ["HF_HUB_VERBOSITY"] = "error"
        os.environ["HF_DATASETS_VERBOSITY"] = "error"

    try:
        from datasets import Audio, load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Error: `datasets` is required. Install with `pip install datasets`."
        ) from exc

    eprint(
        f"Streaming metadata from {args.hf_dataset} "
        f"(split={args.hf_split}, name={args.hf_name})..."
    )

    rows = load_dataset(
        path=args.hf_dataset,
        name=args.hf_name,
        split=args.hf_split,
        streaming=True,
    )

    # Ensure metadata scan does not decode waveforms into arrays.
    # This is critical for large runs where full decode is prohibitively slow.
    if args.hf_disable_audio_decode:
        cast_field = args.hf_audio_field
        try:
            rows = rows.cast_column(cast_field, Audio(decode=False))
            eprint(f"Audio decode disabled for field '{cast_field}' (Audio(decode=False)).")
        except Exception as exc:
            eprint(
                f"Warning: could not cast '{cast_field}' to Audio(decode=False): {exc}. "
                "Proceeding without forced cast."
            )

    exts = normalize_extensions(args.exts)
    speaker_to_candidates: Dict[str, List[Candidate]] = defaultdict(list)
    stats = defaultdict(int)
    start_time = time.monotonic()
    last_heartbeat = start_time
    unique_speakers: Set[str] = set()
    kept_seconds_total = 0.0

    need_seconds = args.target_seconds is not None
    need_bytes = args.target_gb is not None
    eprint("Streaming iterator ready. Waiting for first row...")

    for idx, example in enumerate(rows):
        if args.hf_max_examples is not None and idx >= args.hf_max_examples:
            break
        stats["rows_seen"] += 1

        if args.hf_progress_every > 0 and stats["rows_seen"] % args.hf_progress_every == 0:
            eprint(
                "Progress: "
                f"rows_seen={stats['rows_seen']}, "
                f"rows_kept={stats['rows_kept']}, "
                f"skip_ext={stats['skip_ext']}, "
                f"skip_missing_key={stats['skip_missing_key']}, "
                f"skip_missing_audio_field={stats['skip_missing_audio_field']}, "
                f"skip_missing_speaker={stats['skip_missing_speaker']}, "
                f"skip_missing_seconds={stats['skip_missing_seconds']}, "
                f"skip_missing_bytes={stats['skip_missing_bytes']}, "
                f"estimated_seconds_used={stats['estimated_seconds_used']}, "
                f"estimated_bytes_used={stats['estimated_bytes_used']}"
            )

        if not isinstance(example, dict):
            stats["skip_not_dict"] += 1
            continue

        audio_field = select_audio_field(example, args.hf_audio_field)
        audio_payload = example.get(audio_field) if audio_field else None
        audio_member_ext: Optional[str] = None
        if isinstance(audio_payload, dict):
            payload_path = audio_payload.get("path")
            if isinstance(payload_path, str):
                payload_ext = Path(payload_path).suffix.lower()
                if payload_ext:
                    audio_member_ext = payload_ext
        key_val = example.get(args.hf_key_field)
        key = str(key_val) if key_val is not None else None

        if not key:
            # Fallback from explicit relpath field if present.
            rel_raw = example.get(args.hf_relpath_field)
            if isinstance(rel_raw, str) and rel_raw.strip():
                key = rel_raw

        if not key:
            stats["skip_missing_key"] += 1
            continue

        relpath = normalize_relpath(key)
        ext = Path(relpath).suffix.lower()

        if ext:
            if exts and ext not in exts:
                stats["skip_ext"] += 1
                continue
            member_name = relpath
        else:
            if audio_field is None:
                stats["skip_missing_audio_field"] += 1
                continue
            inferred_suffix = audio_member_ext if audio_member_ext else f".{audio_field}"
            member_name = f"{relpath}{inferred_suffix}"
            relpath = member_name
            if exts and Path(relpath).suffix.lower() not in exts:
                stats["skip_ext"] += 1
                continue

        speaker_id: Optional[str] = None
        speaker_val = example.get(args.hf_speaker_field)
        if speaker_val is not None:
            speaker_id = str(speaker_val)
        if not speaker_id:
            speaker_id = infer_speaker_from_relpath(relpath)
        if not speaker_id:
            stats["skip_missing_speaker"] += 1
            continue

        seconds = safe_float(example.get(args.hf_seconds_field))
        bytes_value = safe_int(example.get(args.hf_bytes_field))

        # Try to recover duration/size without decoding full waveform.
        if isinstance(audio_payload, dict):
            if seconds is None:
                seconds = safe_float(audio_payload.get("duration"))
            if bytes_value is None:
                bytes_value = safe_int(audio_payload.get("bytes"))

        if isinstance(audio_payload, (bytes, bytearray)):
            # With decode disabled, bytes may be present but are optional metadata.
            # We never decode waveform arrays here.
            blob = bytes(audio_payload)
            if bytes_value is None:
                bytes_value = len(blob)
            if seconds is None and Path(relpath).suffix.lower() == ".wav":
                seconds = infer_duration_from_wav_bytes(blob)

        # If targets require a metric and it is unavailable, optionally use estimates.
        if need_seconds and seconds is None:
            if args.hf_allow_metric_estimates:
                seconds = float(args.estimated_seconds_per_file)
                stats["estimated_seconds_used"] += 1
            else:
                stats["skip_missing_seconds"] += 1
                continue
        if need_bytes and bytes_value is None:
            if args.hf_allow_metric_estimates:
                bytes_value = int(args.estimated_bytes_per_file)
                stats["estimated_bytes_used"] += 1
            else:
                stats["skip_missing_bytes"] += 1
                continue

        # Fill non-required metrics with 0 for bookkeeping.
        seconds_final = seconds if seconds is not None else 0.0
        bytes_final = bytes_value if bytes_value is not None else 0

        shard_url = None
        shard_filename = None

        # WebDataset usually provides `__url__` for shard origin.
        url_val = example.get(args.hf_url_field)
        if isinstance(url_val, str) and url_val.strip():
            shard_url = url_val
            shard_filename = parse_resolve_url_to_filename(shard_url)

        # Allow explicit shard filename field fallback.
        if shard_filename is None:
            file_val = example.get(args.hf_shard_field)
            if isinstance(file_val, str) and file_val.strip():
                shard_filename = file_val

        # Stable shard id used for grouping during materialization.
        shard_id = shard_filename or shard_url or "unknown-shard"

        candidate = Candidate(
            speaker_id=speaker_id,
            relpath=relpath,
            seconds=seconds_final,
            bytes=bytes_final,
            shard_id=shard_id,
            shard_filename=shard_filename,
            shard_url=shard_url,
            member_name=member_name,
        )
        speaker_to_candidates[speaker_id].append(candidate)
        stats["rows_kept"] += 1
        unique_speakers.add(speaker_id)
        kept_seconds_total += seconds_final

        if args.hf_heartbeat_seconds > 0:
            now = time.monotonic()
            if now - last_heartbeat >= args.hf_heartbeat_seconds:
                elapsed = now - start_time
                eprint(
                    "Heartbeat: "
                    f"elapsed_s={elapsed:.1f}, "
                    f"rows_seen={stats['rows_seen']}, "
                    f"rows_kept={stats['rows_kept']}, "
                    f"rows_per_sec={(stats['rows_seen'] / elapsed) if elapsed > 0 else 0.0:.2f}"
                )
                last_heartbeat = now

        if args.optimized_mode:
            speakers_target = max(
                args.min_speakers + args.optimized_speaker_buffer,
                args.optimized_min_speakers,
            )
            speakers_ok = len(unique_speakers) >= speakers_target
            kept_ok = stats["rows_kept"] >= args.optimized_min_candidates
            seconds_ok = True
            if args.target_seconds is not None:
                seconds_ok = (
                    kept_seconds_total >= args.target_seconds * args.optimized_seconds_multiplier
                )
            if speakers_ok and kept_ok and seconds_ok:
                eprint(
                    "Optimized early-stop reached: "
                    f"rows_seen={stats['rows_seen']}, "
                    f"rows_kept={stats['rows_kept']}, "
                    f"unique_speakers={len(unique_speakers)}, "
                    f"kept_seconds={kept_seconds_total:.2f}"
                )
                break

    eprint(
        "Streaming summary: "
        f"rows_seen={stats['rows_seen']}, "
        f"rows_kept={stats['rows_kept']}, "
        f"skip_ext={stats['skip_ext']}, "
        f"skip_missing_key={stats['skip_missing_key']}, "
        f"skip_missing_audio_field={stats['skip_missing_audio_field']}, "
        f"skip_missing_speaker={stats['skip_missing_speaker']}, "
        f"skip_missing_seconds={stats['skip_missing_seconds']}, "
        f"skip_missing_bytes={stats['skip_missing_bytes']}, "
        f"estimated_seconds_used={stats['estimated_seconds_used']}, "
        f"estimated_bytes_used={stats['estimated_bytes_used']}"
    )
    return dict(speaker_to_candidates), dict(stats)


# ---------------------------------------------------------------------------
# Phase 2: diversity-first deterministic selection phase
# ---------------------------------------------------------------------------
def target_reached(
    total_seconds: float,
    total_bytes: int,
    target_seconds: Optional[float],
    target_bytes: Optional[int],
) -> bool:
    """Stop when any configured target is reached."""
    checks: List[bool] = []
    if target_seconds is not None:
        checks.append(total_seconds >= target_seconds - EPS)
    if target_bytes is not None:
        checks.append(total_bytes >= target_bytes)
    return any(checks)


def next_eligible(
    speaker_id: str,
    queues: Dict[str, List[Candidate]],
    cursors: Dict[str, int],
    used_seconds_by_speaker: Dict[str, float],
    max_per_speaker_seconds: float,
) -> Optional[Candidate]:
    """Fetch next candidate for a speaker that respects the per-speaker cap."""
    items = queues[speaker_id]
    idx = cursors[speaker_id]
    while idx < len(items):
        candidate = items[idx]
        idx += 1
        if used_seconds_by_speaker[speaker_id] + candidate.seconds > max_per_speaker_seconds + EPS:
            continue
        cursors[speaker_id] = idx
        return candidate
    cursors[speaker_id] = idx
    return None


def select_candidates(
    speaker_to_candidates: Dict[str, List[Candidate]],
    target_seconds: Optional[float],
    target_bytes: Optional[int],
    min_speakers: int,
    max_per_speaker_seconds: float,
    seed: int,
) -> Tuple[List[Candidate], List[str]]:
    """Diversity-first deterministic sampling.

    Phase 1:
      maximize unique speakers, adding at most one sample per speaker first.
    Phase 2:
      balanced round-robin depth expansion across speakers.
    """
    rng = random.Random(seed)

    queues: Dict[str, List[Candidate]] = {}
    for speaker_id, items in speaker_to_candidates.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        queues[speaker_id] = shuffled

    speakers = sorted([s for s, items in queues.items() if items])
    cursors = {s: 0 for s in speakers}
    used_seconds_by_speaker: Dict[str, float] = defaultdict(float)

    selected: List[Candidate] = []
    selected_speakers: Set[str] = set()
    total_seconds = 0.0
    total_bytes = 0
    notes: List[str] = []

    def add(item: Candidate) -> None:
        nonlocal total_seconds, total_bytes
        selected.append(item)
        selected_speakers.add(item.speaker_id)
        used_seconds_by_speaker[item.speaker_id] += item.seconds
        total_seconds += item.seconds
        total_bytes += item.bytes

    # Phase 1: breadth first (one-per-speaker order is shuffled but deterministic by seed).
    phase1_order = list(speakers)
    rng.shuffle(phase1_order)
    for speaker_id in phase1_order:
        if target_reached(total_seconds, total_bytes, target_seconds, target_bytes):
            break
        item = next_eligible(
            speaker_id=speaker_id,
            queues=queues,
            cursors=cursors,
            used_seconds_by_speaker=used_seconds_by_speaker,
            max_per_speaker_seconds=max_per_speaker_seconds,
        )
        if item is not None:
            add(item)

    if len(selected_speakers) < min_speakers:
        notes.append(
            f"Requested min_speakers={min_speakers}, reached {len(selected_speakers)} under constraints."
        )

    # Phase 2: balanced depth with round-robin to prevent dominance.
    while not target_reached(total_seconds, total_bytes, target_seconds, target_bytes):
        round_speakers = [s for s in speakers if cursors[s] < len(queues[s])]
        if not round_speakers:
            break

        rng.shuffle(round_speakers)
        progressed = False
        for speaker_id in round_speakers:
            if target_reached(total_seconds, total_bytes, target_seconds, target_bytes):
                break
            item = next_eligible(
                speaker_id=speaker_id,
                queues=queues,
                cursors=cursors,
                used_seconds_by_speaker=used_seconds_by_speaker,
                max_per_speaker_seconds=max_per_speaker_seconds,
            )
            if item is None:
                continue
            add(item)
            progressed = True

        if not progressed:
            notes.append("No further eligible items under max_per_speaker_seconds.")
            break

    return selected, notes


# ---------------------------------------------------------------------------
# Phase 3: shard-grouped materialization phase
# ---------------------------------------------------------------------------
def download_shard_once(
    shard_id: str,
    shard_filename: Optional[str],
    shard_url: Optional[str],
    args: argparse.Namespace,
    cache_dir: Path,
) -> Path:
    """Download one shard tar once, preferring HF hub filename when possible."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Try repo-relative file download first.
    if shard_filename:
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "`huggingface_hub` is required for shard filename downloads. "
                "Install with `pip install huggingface_hub`."
            ) from exc

        cached_path = hf_hub_download(
            repo_id=args.hf_dataset,
            repo_type="dataset",
            filename=shard_filename,
            revision=args.hf_revision,
            local_dir=str(cache_dir),
        )
        return Path(cached_path)

    # Handle HF URI format seen in some WebDataset streams.
    if shard_url and shard_url.startswith("hf://datasets/"):
        parsed = parse_hf_dataset_uri(shard_url)
        if parsed is None:
            raise RuntimeError(f"Unsupported HF dataset URI format: {shard_url}")
        repo_id, revision, filename = parsed
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "`huggingface_hub` is required for hf:// shard downloads. "
                "Install with `pip install huggingface_hub`."
            ) from exc

        cached_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            revision=revision or args.hf_revision,
            local_dir=str(cache_dir),
        )
        return Path(cached_path)

    # Fallback to direct HTTP URL if provided.
    if shard_url:
        try:
            import requests  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "`requests` is required for direct shard URL download. "
                "Install with `pip install requests`."
            ) from exc

        basename = os.path.basename(urlparse(shard_url).path) or "shard.tar"
        dst = cache_dir / f"{abs(hash(shard_id))}_{basename}"
        if dst.exists() and dst.stat().st_size > 0:
            return dst

        with requests.get(shard_url, stream=True, timeout=180) as response:
            response.raise_for_status()
            with dst.open("wb") as out_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out_file.write(chunk)
        return dst

    raise RuntimeError(f"No shard filename/url available for shard_id={shard_id}")


def extract_selected_from_shard(
    shard_path: Path,
    selected_items: List[Candidate],
    subset_root: Path,
    allowed_exts: Set[str],
) -> None:
    """Extract only selected members from one shard tar."""
    # Use normalized names to be robust to leading './'.
    exact_members: Dict[str, Candidate] = {
        item.member_name.lstrip("./"): item for item in selected_items
    }
    stem_members: Dict[str, List[Candidate]] = defaultdict(list)
    for item in selected_items:
        stem = str(PurePosixPath(item.member_name.lstrip("./")).with_suffix(""))
        stem_members[stem].append(item)

    with tarfile.open(shard_path, "r") as tar:
        matched_item_ids: Set[int] = set()
        for member in tar:
            if not member.isfile():
                continue
            normalized_name = member.name.lstrip("./")
            member_ext = Path(normalized_name).suffix.lower()
            if allowed_exts and member_ext not in allowed_exts:
                continue
            item = exact_members.get(normalized_name)
            if item is None:
                stem = str(PurePosixPath(normalized_name).with_suffix(""))
                candidates = stem_members.get(stem, [])
                if len(candidates) == 1:
                    item = candidates[0]
                    # Correct stale extension from index (e.g. .wav -> .m4a).
                    item.member_name = normalized_name
                    item.relpath = normalized_name
            if item is None:
                continue
            if id(item) in matched_item_ids:
                continue

            dst_path = subset_root / item.relpath
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            with extracted:
                with dst_path.open("wb") as out_file:
                    shutil.copyfileobj(extracted, out_file)
            matched_item_ids.add(id(item))

        missing = [item.member_name for item in selected_items if id(item) not in matched_item_ids]
        if missing:
            missing_preview = sorted(missing)[:5]
            raise RuntimeError(
                f"Missing {len(missing)} selected members in shard {shard_path}. "
                f"Examples: {missing_preview}"
            )


def materialize_subset(
    selected: List[Candidate],
    subset_root: Path,
    args: argparse.Namespace,
) -> None:
    """Group by shard, download each shard once, and extract selected members."""
    if args.mode == "manifest":
        return

    by_shard: Dict[str, List[Candidate]] = defaultdict(list)
    for item in selected:
        by_shard[item.shard_id].append(item)

    cache_dir = args.out_root / ".shard_cache"

    for shard_id in sorted(by_shard.keys()):
        items = by_shard[shard_id]
        any_item = items[0]
        eprint(f"Materializing {len(items)} samples from shard: {shard_id}")
        shard_path = download_shard_once(
            shard_id=shard_id,
            shard_filename=any_item.shard_filename,
            shard_url=any_item.shard_url,
            args=args,
            cache_dir=cache_dir,
        )
        extract_selected_from_shard(
            shard_path=shard_path,
            selected_items=items,
            subset_root=subset_root,
            allowed_exts=normalize_extensions(args.exts),
        )


def get_audio_duration_seconds(path: Path) -> float:
    """Get local audio duration for supported file types."""
    ext = path.suffix.lower()
    if ext == ".wav":
        with wave.open(str(path), "rb") as wav_file:
            rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            if rate <= 0:
                raise ValueError(f"Invalid WAV sample rate: {path}")
            return frames / float(rate)
    if ext == ".m4a":
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            raise RuntimeError("ffprobe is required to compute m4a durations.")
        out = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
        ).strip()
        dur = float(out)
        if dur <= 0:
            raise ValueError(f"Non-positive duration from ffprobe: {path}")
        return dur
    raise ValueError(f"Unsupported audio extension: {path.suffix}")


def refresh_selected_metrics_from_subset(subset_root: Path, selected: List[Candidate]) -> None:
    """Replace estimated metrics with true on-disk metrics after copy."""
    for item in selected:
        local_path = subset_root / item.relpath
        if not local_path.exists():
            continue
        item.bytes = local_path.stat().st_size
        try:
            item.seconds = get_audio_duration_seconds(local_path)
        except Exception:
            # Keep prior value if duration probing fails.
            pass


def build_dry_run_report(
    selected: List[Candidate],
    assumed_shard_mb: float,
) -> Dict[str, float]:
    """Compute pre-download estimates from selected candidates."""
    unique_shards = {item.shard_id for item in selected}
    selected_media_bytes = sum(item.bytes for item in selected)
    selected_seconds = sum(item.seconds for item in selected)
    estimated_shard_download_bytes = int(len(unique_shards) * assumed_shard_mb * 1024 * 1024)
    return {
        "num_selected_files": float(len(selected)),
        "num_unique_shards": float(len(unique_shards)),
        "selected_media_bytes": float(selected_media_bytes),
        "selected_media_gb": selected_media_bytes / BYTES_PER_GB,
        "selected_seconds": selected_seconds,
        "estimated_shard_download_bytes": float(estimated_shard_download_bytes),
        "estimated_shard_download_gb": estimated_shard_download_bytes / BYTES_PER_GB,
        "assumed_shard_mb": assumed_shard_mb,
    }


def print_dry_run_report(report: Dict[str, float]) -> None:
    """Print a concise download-planning report."""
    print("Dry-run report:")
    print(f"- selected_files: {int(report['num_selected_files'])}")
    print(f"- unique_shards: {int(report['num_unique_shards'])}")
    print(f"- selected_media_gb: {report['selected_media_gb']:.4f}")
    print(f"- selected_seconds: {report['selected_seconds']:.2f}")
    print(f"- estimated_shard_download_gb: {report['estimated_shard_download_gb']:.4f}")
    print(f"- assumed_shard_mb: {report['assumed_shard_mb']:.1f}")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def write_manifest(path: Path, selected: Iterable[Candidate]) -> None:
    """Write selected file list."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker_id", "relpath", "seconds", "bytes"])
        for item in selected:
            writer.writerow([item.speaker_id, item.relpath, f"{item.seconds:.6f}", item.bytes])


def write_stats(
    path: Path,
    selected: List[Candidate],
    seed: int,
    cli_args: Dict[str, object],
    notes: List[str],
    streaming_stats: Dict[str, int],
    dry_run_report: Dict[str, float],
) -> None:
    """Write summary metrics and debug counters."""
    speaker_seconds: Dict[str, float] = defaultdict(float)
    total_seconds = 0.0
    total_bytes = 0
    for item in selected:
        speaker_seconds[item.speaker_id] += item.seconds
        total_seconds += item.seconds
        total_bytes += item.bytes

    top_speakers = sorted(speaker_seconds.items(), key=lambda x: (-x[1], x[0]))[:10]

    payload = {
        "num_files": len(selected),
        "num_speakers": len(speaker_seconds),
        "total_bytes": total_bytes,
        "total_gb": total_bytes / BYTES_PER_GB,
        "total_seconds": total_seconds,
        "top_speakers_by_seconds": [
            {"speaker_id": speaker, "seconds": secs} for speaker, secs in top_speakers
        ],
        "seed": seed,
        "cli_args": cli_args,
        "notes": notes,
        "streaming_stats": streaming_stats,
        "dry_run_report": dry_run_report,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Shard-aware VoxCeleb subset sampler for HF WebDataset mirrors"
    )

    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--target_gb", type=float, default=None)
    parser.add_argument("--target_seconds", type=float, default=None)
    parser.add_argument(
        "--save_index",
        type=Path,
        default=None,
        help="Write streamed candidate metadata index (jsonl) for fast future runs",
    )
    parser.add_argument(
        "--load_index",
        type=Path,
        default=None,
        help="Load candidate metadata index (jsonl) instead of streaming HF",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_per_speaker_seconds", type=float, default=90.0)
    parser.add_argument("--min_speakers", type=int, default=0)
    parser.add_argument("--mode", choices=["copy", "manifest"], default="copy")
    parser.add_argument(
        "--dry_run_report",
        action="store_true",
        help="Print shard/download estimate before materialization",
    )
    parser.add_argument(
        "--assumed_shard_mb",
        type=float,
        default=100.0,
        help="Assumed shard size used for dry-run download estimate",
    )
    parser.add_argument("--exts", nargs="+", default=[".wav", ".m4a"])

    # HF / WebDataset controls.
    parser.add_argument("--hf_dataset", default="gaunernst/voxceleb2-dev-wds")
    parser.add_argument("--hf_name", default=None)
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--hf_revision", default="main")

    parser.add_argument("--hf_speaker_field", default="speaker_id")
    parser.add_argument("--hf_relpath_field", default="relpath")
    parser.add_argument("--hf_seconds_field", default="seconds")
    parser.add_argument("--hf_bytes_field", default="bytes")

    parser.add_argument("--hf_key_field", default="__key__")
    parser.add_argument("--hf_url_field", default="__url__")
    parser.add_argument("--hf_shard_field", default="__shard__")
    parser.add_argument("--hf_audio_field", default="wav")
    parser.add_argument(
        "--hf_disable_audio_decode",
        action="store_true",
        help="Cast audio field to Audio(decode=False) during metadata streaming",
    )
    parser.add_argument(
        "--hf_progress_every",
        type=int,
        default=1000,
        help="Log streaming progress every N rows (0 disables progress logs)",
    )
    parser.add_argument(
        "--hf_heartbeat_seconds",
        type=int,
        default=20,
        help="Log heartbeat every N seconds during streaming (0 disables heartbeat)",
    )
    parser.add_argument(
        "--hf_allow_metric_estimates",
        action="store_true",
        help="Use per-file estimate fallbacks when seconds/bytes metadata is missing",
    )
    parser.add_argument(
        "--estimated_seconds_per_file",
        type=float,
        default=4.0,
        help="Fallback seconds value when duration metadata is missing",
    )
    parser.add_argument(
        "--estimated_bytes_per_file",
        type=int,
        default=160000,
        help="Fallback bytes value when size metadata is missing",
    )
    parser.add_argument(
        "--hf_http_debug",
        action="store_true",
        help="Enable verbose HF hub/datasets HTTP logs",
    )

    parser.add_argument(
        "--optimized_mode",
        action="store_true",
        help="Enable early-stop metadata scan once coverage thresholds are met",
    )
    parser.add_argument(
        "--optimized_seconds_multiplier",
        type=float,
        default=2.0,
        help="In optimized mode, scan until kept_seconds >= target_seconds * multiplier",
    )
    parser.add_argument(
        "--optimized_speaker_buffer",
        type=int,
        default=300,
        help="In optimized mode, require min_speakers + this buffer before early-stop",
    )
    parser.add_argument(
        "--optimized_min_speakers",
        type=int,
        default=800,
        help="Absolute minimum unique speakers before optimized early-stop",
    )
    parser.add_argument(
        "--optimized_min_candidates",
        type=int,
        default=5000,
        help="Absolute minimum kept candidates before optimized early-stop",
    )

    parser.add_argument("--hf_max_examples", type=int, default=None)

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate required constraints."""
    if args.target_gb is None and args.target_seconds is None:
        raise SystemExit("Error: provide --target_gb and/or --target_seconds.")
    if args.target_gb is not None and args.target_gb <= 0:
        raise SystemExit("Error: --target_gb must be > 0.")
    if args.target_seconds is not None and args.target_seconds <= 0:
        raise SystemExit("Error: --target_seconds must be > 0.")
    if args.max_per_speaker_seconds <= 0:
        raise SystemExit("Error: --max_per_speaker_seconds must be > 0.")
    if args.min_speakers < 0:
        raise SystemExit("Error: --min_speakers must be >= 0.")
    if args.assumed_shard_mb <= 0:
        raise SystemExit("Error: --assumed_shard_mb must be > 0.")
    if args.hf_progress_every < 0:
        raise SystemExit("Error: --hf_progress_every must be >= 0.")
    if args.hf_heartbeat_seconds < 0:
        raise SystemExit("Error: --hf_heartbeat_seconds must be >= 0.")
    if args.estimated_seconds_per_file <= 0:
        raise SystemExit("Error: --estimated_seconds_per_file must be > 0.")
    if args.estimated_bytes_per_file <= 0:
        raise SystemExit("Error: --estimated_bytes_per_file must be > 0.")
    if args.optimized_seconds_multiplier <= 0:
        raise SystemExit("Error: --optimized_seconds_multiplier must be > 0.")
    if args.optimized_speaker_buffer < 0:
        raise SystemExit("Error: --optimized_speaker_buffer must be >= 0.")
    if args.optimized_min_speakers < 0:
        raise SystemExit("Error: --optimized_min_speakers must be >= 0.")
    if args.optimized_min_candidates < 0:
        raise SystemExit("Error: --optimized_min_candidates must be >= 0.")
    if args.load_index is not None and not args.load_index.exists():
        raise SystemExit(f"Error: --load_index file does not exist: {args.load_index}")


def main() -> int:
    """Run metadata streaming, selection, and shard-grouped materialization."""
    args = parse_args()
    validate_args(args)

    target_bytes = int(args.target_gb * BYTES_PER_GB) if args.target_gb is not None else None
    target_seconds = float(args.target_seconds) if args.target_seconds is not None else None

    args.out_root.mkdir(parents=True, exist_ok=True)
    subset_root = args.out_root / "subset"
    if args.mode == "copy" and not args.dry_run_report:
        if subset_root.exists() and any(subset_root.iterdir()):
            raise SystemExit(
                f"Error: output subset directory is not empty: {subset_root}. "
                "Use a fresh --out_root or empty this directory."
            )
        subset_root.mkdir(parents=True, exist_ok=True)

    # Phase 1: metadata source (index or streaming).
    if args.load_index is not None:
        eprint(f"Loading metadata index: {args.load_index}")
        speaker_to_candidates = load_index(args.load_index)
        streaming_stats = {"rows_seen": 0, "rows_kept": sum(len(v) for v in speaker_to_candidates.values())}
    else:
        speaker_to_candidates, streaming_stats = stream_metadata(args)
        if args.save_index is not None:
            eprint(f"Saving metadata index: {args.save_index}")
            save_index(args.save_index, speaker_to_candidates)
    total_candidates = sum(len(v) for v in speaker_to_candidates.values())
    eprint(
        f"Metadata scan complete: {total_candidates} candidates across "
        f"{len(speaker_to_candidates)} speakers."
    )
    if total_candidates == 0:
        rows_seen = streaming_stats.get("rows_seen", 0)
        skip_missing_seconds = streaming_stats.get("skip_missing_seconds", 0)
        skip_missing_bytes = streaming_stats.get("skip_missing_bytes", 0)
        skip_missing_key = streaming_stats.get("skip_missing_key", 0)
        skip_missing_speaker = streaming_stats.get("skip_missing_speaker", 0)
        skip_ext = streaming_stats.get("skip_ext", 0)
        hint = (
            " Check HF field mappings (--hf_key_field, --hf_speaker_field, "
            "--hf_seconds_field, --hf_bytes_field, --hf_audio_field) and/or use "
            "--hf_allow_metric_estimates when metadata is unavailable."
        )
        if rows_seen > 0 and skip_missing_seconds == rows_seen and target_seconds is not None:
            hint = (
                " All rows were missing duration metadata for --target_seconds. "
                "Set --hf_seconds_field to a valid duration column, or run with "
                "--hf_allow_metric_estimates --estimated_seconds_per_file <value>."
            )
        elif rows_seen > 0 and skip_missing_bytes == rows_seen and target_bytes is not None:
            hint = (
                " All rows were missing byte-size metadata for --target_gb. "
                "Set --hf_bytes_field to a valid size column, or run with "
                "--hf_allow_metric_estimates --estimated_bytes_per_file <value>."
            )
        raise SystemExit(
            "Error: no candidates found with current field/extension settings. "
            f"rows_seen={rows_seen}, skip_ext={skip_ext}, "
            f"skip_missing_key={skip_missing_key}, "
            f"skip_missing_speaker={skip_missing_speaker}, "
            f"skip_missing_seconds={skip_missing_seconds}, "
            f"skip_missing_bytes={skip_missing_bytes}. "
            f"{hint}"
        )

    has_eligible = any(
        item.seconds <= args.max_per_speaker_seconds + EPS
        for items in speaker_to_candidates.values()
        for item in items
    )
    if not has_eligible:
        raise SystemExit(
            "Error: no items are short enough for --max_per_speaker_seconds. Increase the cap."
        )

    # Phase 2: diversity-first selection.
    selected, notes = select_candidates(
        speaker_to_candidates=speaker_to_candidates,
        target_seconds=target_seconds,
        target_bytes=target_bytes,
        min_speakers=args.min_speakers,
        max_per_speaker_seconds=float(args.max_per_speaker_seconds),
        seed=args.seed,
    )
    if not selected:
        raise SystemExit("Error: no files selected. Relax constraints and try again.")

    dry_run_report = build_dry_run_report(selected, args.assumed_shard_mb)
    if args.dry_run_report:
        print_dry_run_report(dry_run_report)

    # Early-exit strategy for planning runs:
    # when dry_run_report is requested, skip materialization entirely.
    if args.dry_run_report:
        eprint("Dry-run report requested: skipping shard materialization.")
    else:
        # Phase 3: shard-grouped materialization.
        eprint(f"Materialization mode: {args.mode}")
        materialize_subset(selected=selected, subset_root=subset_root, args=args)
        if args.mode == "copy":
            eprint("Refreshing true file metrics from extracted subset...")
            refresh_selected_metrics_from_subset(subset_root, selected)

    # Outputs.
    manifest_path = args.out_root / "manifest.csv"
    stats_path = args.out_root / "stats.json"

    write_manifest(manifest_path, selected)

    cli_args = {
        "out_root": str(args.out_root),
        "target_gb": args.target_gb,
        "target_seconds": args.target_seconds,
        "save_index": str(args.save_index) if args.save_index is not None else None,
        "load_index": str(args.load_index) if args.load_index is not None else None,
        "seed": args.seed,
        "max_per_speaker_seconds": args.max_per_speaker_seconds,
        "min_speakers": args.min_speakers,
        "mode": args.mode,
        "dry_run_report": args.dry_run_report,
        "assumed_shard_mb": args.assumed_shard_mb,
        "exts": args.exts,
        "hf_dataset": args.hf_dataset,
        "hf_name": args.hf_name,
        "hf_split": args.hf_split,
        "hf_revision": args.hf_revision,
        "hf_speaker_field": args.hf_speaker_field,
        "hf_relpath_field": args.hf_relpath_field,
        "hf_seconds_field": args.hf_seconds_field,
        "hf_bytes_field": args.hf_bytes_field,
        "hf_key_field": args.hf_key_field,
        "hf_url_field": args.hf_url_field,
        "hf_shard_field": args.hf_shard_field,
        "hf_audio_field": args.hf_audio_field,
        "hf_disable_audio_decode": args.hf_disable_audio_decode,
        "hf_progress_every": args.hf_progress_every,
        "hf_heartbeat_seconds": args.hf_heartbeat_seconds,
        "hf_allow_metric_estimates": args.hf_allow_metric_estimates,
        "estimated_seconds_per_file": args.estimated_seconds_per_file,
        "estimated_bytes_per_file": args.estimated_bytes_per_file,
        "hf_http_debug": args.hf_http_debug,
        "hf_max_examples": args.hf_max_examples,
        "optimized_mode": args.optimized_mode,
        "optimized_seconds_multiplier": args.optimized_seconds_multiplier,
        "optimized_speaker_buffer": args.optimized_speaker_buffer,
        "optimized_min_speakers": args.optimized_min_speakers,
        "optimized_min_candidates": args.optimized_min_candidates,
    }
    write_stats(
        path=stats_path,
        selected=selected,
        seed=args.seed,
        cli_args=cli_args,
        notes=notes,
        streaming_stats=streaming_stats,
        dry_run_report=dry_run_report,
    )

    print(f"Subset path: {subset_root}" if args.mode == "copy" else "Subset path: (not materialized; mode=manifest)")
    print(f"Manifest: {manifest_path}")
    print(f"Stats: {stats_path}")
    print(f"Selected files: {len(selected)}")
    print(f"Selected speakers: {len({item.speaker_id for item in selected})}")
    print(f"Selected seconds: {sum(item.seconds for item in selected):.2f}")
    print(f"Selected GB: {sum(item.bytes for item in selected) / BYTES_PER_GB:.4f}")
    if notes:
        print("Notes:")
        for note in notes:
            print(f"- {note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
