#!/usr/bin/env bash
# All eight fixtures, one at a time, each inside its own lock acquisition, each log self-documenting.
set -uo pipefail
cd "$(dirname "$0")/../.."
for fx in dec8 dec64f dec64t dec300f mt32f p128 wide520f chunk64b; do
    bash tmp/d9_sizing/run_gate.sh "$fx" tmp/d9_sizing/new 2>&1 | sed 's/^/  /'
done
