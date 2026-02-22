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
