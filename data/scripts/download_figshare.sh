#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="${ROOT_DIR}/raw/figshare"
API_URL="https://api.figshare.com/v2/articles/1512427/versions/5"

mkdir -p "${RAW_DIR}"
cd "${RAW_DIR}"

echo "Downloading Figshare brain tumor dataset files..."
echo "Source: https://doi.org/10.6084/m9.figshare.1512427.v5"

python3 - <<'PY' > figshare_files.tsv
import json
import urllib.request

api_url = "https://api.figshare.com/v2/articles/1512427/versions/5"
with urllib.request.urlopen(api_url) as response:
    payload = json.loads(response.read().decode("utf-8"))
for item in payload["files"]:
    print(f"{item['name']}\t{item['download_url']}")
PY

while IFS=$'\t' read -r filename url; do
  if [[ -f "${filename}" ]]; then
    echo "Skipping existing ${filename}"
    continue
  fi
  echo "Downloading ${filename}"
  curl -L --fail --retry 3 -o "${filename}" "${url}"
done < figshare_files.tsv

echo "Downloaded Figshare files to: ${RAW_DIR}"
echo "Next step: unzip the brainTumorDataPublic_*.zip files and convert .mat files into data/processed/Figshare."
