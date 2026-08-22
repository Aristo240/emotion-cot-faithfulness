#!/bin/sh
# Build an anonymized supplementary zip for a double-blind submission.
# Excludes .git (commit metadata carries the author's name and email) and
# scrubs absolute paths and the repo URL from tracked text files.
set -e
SRC="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-/tmp/anon_supplementary}"
rm -rf "$OUT"; mkdir -p "$OUT/supplementary"
cd "$SRC"
git ls-files -z | xargs -0 -I{} sh -c 'mkdir -p "$0/supplementary/$(dirname {})" && cp "{}" "$0/supplementary/{}"' "$OUT"
cd "$OUT/supplementary"
# scrub identifiers from text files only (leave .jsonl/.npz data alone unless they match)
grep -rlIE 'naamarozen|Aristo240|rozenn@post\.bgu\.ac\.il|/home/gamir|/specific/scratches|Users/naamarozen' . 2>/dev/null | while read f; do
  sed -i -E \
    -e 's#/Users/naamarozen/Desktop/Naama/Projects AI/emotion-cot-faithfulness/emotion-cot-faithfulness_16Apr#<REPO_ROOT>#g' \
    -e 's#/specific/scratches/scratch/naamarozen/emotion-cot-faithfulness#<REPO_ROOT>#g' \
    -e 's#/specific/scratches/scratch/naamarozen/conda_envs/emotion-cot#<CONDA_ENV>#g' \
    -e 's#/home/gamir/naamarozen/bin/tectonic#<TECTONIC>#g' \
    -e 's#/home/gamir/naamarozen/gfs/emotion-cot#<REPO_ROOT>#g' \
    -e 's#github\.com/Aristo240/emotion-cot-faithfulness#<ANONYMIZED REPO>#g' \
    -e 's#naamarozen240@gmail\.com#<EMAIL>#g' \
    -e 's#rozenn@post\.bgu\.ac\.il#<EMAIL>#g' \
    -e 's#Naama Rozen#<AUTHOR>#g' \
    -e 's#naamarozen#<AUTHOR>#g' \
    -e 's#Aristo240#<AUTHOR>#g' "$f"
done
cd "$OUT"
zip -qr supplementary.zip supplementary
echo "built $OUT/supplementary.zip"
