#!/usr/bin/env bash
# Regenerate the canonical §10.0 fixture set, and install each fixture only if it
# reproduces the pinned tensors **byte for byte** (tools/check_dump_reproduction.py).
#
# Why this script exists: the exact prompt texts are part of a fixture's identity, but
# before now they lived only inside the gitignored manifests, so a fixture could not be
# regenerated from the repository. They are committed under speccheck/fixtures/ and
# referenced below. `data/long.txt` (also gitignored) is the same text as p300.txt.
#
#   ./tools/dump_fixtures.sh                 # every fixture
#   ./tools/dump_fixtures.sh smoke dec8      # just these
#
# Each generation goes to $TMP first; a fixture whose old pins do not reproduce stops
# the run with the offending names printed, and the pinned directory is left untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FIX=speccheck/fixtures
TMP=${TMP:-/tmp/refs_regen}
mkdir -p "$TMP"

run() {
  local tag=$1; shift
  if [ ! -d "data/refs/$tag" ]; then
    echo "no pinned fixture at data/refs/$tag - refusing (this script verifies reproduction," \
         "it does not create fixtures)" >&2
    return 1
  fi
  rm -rf "$TMP/$tag"
  echo "=== $tag: reference_dump.py $*"
  "$PY" tools/reference_dump.py --out "$TMP/$tag" "$@"
  "$PY" tools/check_dump_reproduction.py "data/refs/$tag" "$TMP/$tag"
  cp "$TMP/$tag"/*.npy "$TMP/$tag/manifest.json" "data/refs/$tag/"
  echo "=== $tag installed ($(ls "data/refs/$tag"/*.npy | wc -l) pins)"
}

SEL="${*// /,}"
want() { [ -z "$SEL" ] || case ",$SEL," in *",$1,"*) return 0;; *) return 1;; esac; }

if want smoke; then
  run smoke --message "The capital of France is" --decode 2
fi
if want dec8; then
  run dec8 --message "The capital of France is" --decode 8
fi
if want prefix24dec8; then
  run prefix24dec8 --prompt "$(cat "$FIX/prefix24dec8.txt")" --decode 8
fi
if want p8; then
  run p8 --prompt "$(cat "$FIX/p8.txt")" --decode 0
fi
if want p64; then
  run p64 --message "$(cat "$FIX/p64.txt")" --decode 0
fi
if want p300; then
  run p300 --message "$(cat "$FIX/p300.txt")" --decode 0
fi
