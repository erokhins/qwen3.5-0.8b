#!/bin/sh
# Q10 A/B wrapper: the shipped binary with the contig weight walk as its default, so
# engine/gate.py (which passes no mapping flag) can gate the alternative mapping.
exec /home/erokhins/erokhins_projects/qwen3.5-0.8b/engine/build/decode_run --mlp-map contig "$@"
