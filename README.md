# voxceleb-subset-sampler

VoxCeleb is nearly 1TB in size, which makes downloading and processing the full dataset impractical for quick experiments. This script streams the dataset and creates a smaller, reproducible subset suitable for fast speaker identification development without requiring a full download.

## What this script does

- Streams dataset metadata (no full dataset download)
- Builds a lightweight index of candidates
- Selects a deterministic subset using seeded randomness
- Groups selections by shard to minimize redundant downloads
- Extracts only the selected files

## Usage

```bash
python voxceleb_sampler.py --help

## Runbook (copy/paste)

### 1) One-time setup
```bash
python3 -m pip install -U datasets huggingface_hub requests

export HF_TOKEN=<YOUR_HF_READ_TOKEN>


2) Build a reusable metadata index (fast, no full dataset download)

Replace:

<INDEX_DIR> e.g. ./runs/index

<INDEX_FILE> e.g. voxceleb2_wds_index_v2.jsonl

python3 voxceleb_sampler.py \
  --out_root <INDEX_DIR>/build_run \
  --target_gb 2.0 \
  --mode manifest \
  --dry_run_report \
  --hf_disable_audio_decode \
  --hf_allow_metric_estimates \
  --estimated_bytes_per_file 160000 \
  --optimized_mode \
  --optimized_min_candidates 10000 \
  --optimized_min_speakers 1000 \
  --optimized_speaker_buffer 300 \
  --hf_progress_every 5000 \
  --hf_heartbeat_seconds 10 \
  --save_index <INDEX_DIR>/<INDEX_FILE>

3) Create the actual subset from the saved index

Replace:

<OUT_DIR> e.g. ./runs/subset_team

<INDEX_DIR>/<INDEX_FILE> from step 2

tune --target_gb, --min_speakers, --max_per_speaker_seconds

python3 voxceleb_sampler.py \
  --out_root <OUT_DIR> \
  --target_gb 2.0 \
  --min_speakers 1200 \
  --max_per_speaker_seconds 60 \
  --mode copy \
  --hf_allow_metric_estimates \
  --estimated_bytes_per_file 160000 \
  --load_index <INDEX_DIR>/<INDEX_FILE>

4) Outputs

Subset audio: <OUT_DIR>/subset/

Manifest: <OUT_DIR>/manifest.csv

Stats: <OUT_DIR>/stats.json

5) Quick sanity checks

python3 - <<'PY'
from pathlib import Path
root=Path("<OUT_DIR>/subset")
counts={}
for p in root.rglob("*"):
    if p.is_file():
        counts[p.suffix.lower()]=counts.get(p.suffix.lower(),0)+1
print(counts)
PY

Expected: mostly/only '.m4a'.
