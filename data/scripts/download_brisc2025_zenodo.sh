#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="${ROOT_DIR}/raw/brisc2025"
API_URL="https://zenodo.org/api/records/17524350"

mkdir -p "${RAW_DIR}"
cd "${RAW_DIR}"

echo "Downloading BRISC2025 from Zenodo..."
echo "Source: https://doi.org/10.5281/zenodo.17524350"

python3 - <<'PY' > brisc2025_files.tsv
import json
import urllib.request

api_url = "https://zenodo.org/api/records/17524350"
with urllib.request.urlopen(api_url) as response:
    payload = json.loads(response.read().decode("utf-8"))
for item in payload["files"]:
    if item["key"] == "brisc2025.zip":
        print(f"{item['key']}\t{item['links']['self']}")
PY

while IFS=$'\t' read -r filename url; do
  if [[ -f "${filename}" ]]; then
    echo "Skipping existing ${filename}"
    continue
  fi
  echo "Downloading ${filename}"
  curl -L --fail --retry 5 --retry-delay 5 -o "${filename}" "${url}"
done < brisc2025_files.tsv

echo "Downloaded BRISC2025 files to: ${RAW_DIR}"
echo "Next step:"
echo "  python data/scripts/preprocess_brisc2025.py --zip_path data/raw/brisc2025/brisc2025.zip --output_dir data/processed/Brisc2025 --overwrite"
