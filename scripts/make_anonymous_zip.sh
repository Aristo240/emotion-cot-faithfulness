#!/bin/sh
# Build an anonymized supplementary zip for a double-blind submission.
# Excludes .git (commit metadata carries the author's name and email) and
# scrubs absolute paths and the repo URL from tracked text files.
#
# 2026-08-23: the previous version shipped ITSELF, and its own sed rules spell
# the author's email out in escaped form (rozenn@post\.bgu\.ac\.il). The scrub
# pattern matches the unescaped address, so it never matched its own source and
# the address went into the payload twice. Two changes:
#   1. EXCLUDE list, this file first -- a scrubber cannot scrub itself.
#   2. A verification pass that EXITS NONZERO if any identifier survives.
# The handoff called the old payload "verified" on the strength of one manual
# grep. Verification that is not run by the build is not verification.
set -e
SRC="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-/tmp/anon_supplementary}"
rm -rf "$OUT"; mkdir -p "$OUT/supplementary"
cd "$SRC"

# Files kept out of the payload. The scrubber itself must be here. The working
# notes are here because reviewers do not need them and they discuss the
# submission's own history; drop these two lines to ship them.
EXCLUDE="scripts/make_anonymous_zip.sh
HANDOFF.md
CONTINUATION_NOTES.md
docs/writing_style.md
annotation/worksheet.txt
annotation/labels.csv
annotation/key.json"

git ls-files | grep -vxF "$EXCLUDE" | while read -r f; do
  mkdir -p "$OUT/supplementary/$(dirname "$f")"
  cp "$f" "$OUT/supplementary/$f"
done
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

# ---- verification: the payload must contain no identifier, in any escaping.
# Matches the address with an optional backslash before each dot, which is what
# the old version missed. EmoBank sentence text legitimately contains unrelated
# third-party addresses (e.g. arunrob@gmail.com), so this looks for OUR
# identifiers only, not for "@" in general.
cd "$OUT/supplementary"
LEAK=$(grep -rlIE 'naamarozen|Aristo240|Naama[ _]?Rozen|rozenn(\\?\.)?@?|rozenn@post(\\?\.)bgu|bgu\\?\.ac\\?\.il' . 2>/dev/null || true)
if [ -n "$LEAK" ]; then
  echo "ANONYMITY CHECK FAILED -- identifiers survive in:" >&2
  echo "$LEAK" >&2
  exit 1
fi
echo "anonymity check passed: no author identifier in the payload"

cd "$OUT"
zip -qr supplementary.zip supplementary
echo "built $OUT/supplementary.zip"
