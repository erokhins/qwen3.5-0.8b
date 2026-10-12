#pragma once
// Observed launch counters — the instrument that can tell a batched wave from a per-row wave.
//
// WHY THIS EXISTS, and why it is not just a debug print. P4 batched glue and the MLP inside the
// prefill wave (`submit_wave_body`), and the five fixtures came back with the worst per-step logits
// deficit bit-identical before and after (x1.000 on all four deep fixtures). That is the numerically
// correct answer — each batched kernel reads a weight row once per tile without changing any
// per-element reduction order — but it is ALSO the exact answer an engine that never reached the
// batched kernels would print: identical arithmetic submitted one row at a time produces identical
// bits. Output equality cannot distinguish "batching is free" from "batching is dead code", and
// nothing else in the tree could. So the proof has to be about the SUBMISSIONS, not the outputs:
// how many launches did a wave issue, and what row width did each one ask for. That is what this
// file counts and `engine/gate.py` section (h) judges.
//
// It is the same trap class as `prototype/mlp_gemv/results.md`'s "a negative test that loads another
// layer's weights catches every injection no matter what the kernel does": a proof that cannot fail
// is not a proof. The gate's negative is `decode_run --wave-rows 1`, a run that really is one row at
// a time, and the assertion is that it looks EXACTLY like the dead-code case.
//
// WHAT IS IN SCOPE: the families that can batch — glue and the MLP's two halves (P4), the nine
// projections (P5), and the three token-parallel GDN sites (P6: conv §04.2, gates §04.3, gated norm
// §04.5). `gdn_rec` (§04.4) is deliberately NOT a family: the recurrence is a sequential scan over
// tokens, so its launcher has no row width to report and a counter that can only ever read `rows=1`
// is a check that cannot fail. Giving it a width is P9's chunked scan, a separate conditional
// package. The launch bill of the rest of a token is `planned_launch_counts()`'s, which is a
// derivation from §07 rather than an observation, and `gate.py` section (d) polices that one.
//
// Always on, not behind a switch: counting is three integer ops per launch against a measured 1.85 µs
// dispatch floor (`tools/launch_overhead_hip.cpp`), and an instrument that can be turned off is an
// instrument that gets turned off. Same choice as the `LAUNCH` plan line, which `decode_run` also
// prints unconditionally. Single-threaded by construction — the engine submits from one host thread —
// so no atomics.

#include <string>

namespace engine::launchstats {

// One launcher family, as observed on the host. `rows` means the launcher's row/token width
// argument: for glue it is the tile rows of `glue_add_norm_rows`, for the MLP it is its `tokens`, for
// the projections it is the wave width handed to `proj_gemv` — NOT that signature's `rows`, which
// counts one projection's output rows and is a different axis entirely — and for the GDN trio it is
// the wave width handed to `gdn_conv` / `gdn_gates` / `gdn_gated_norm`, i.e. tokens, since each of
// those is one launch per GDN layer per wave whatever the width.
struct Family {
    long long calls = 0;           // launches actually submitted
    long long calls_batched = 0;   // of those, the ones with rows > 1 — the batched form ran
    long long rows_total = 0;      // sum of rows over all calls: the WORK covered
    long long rows_max = 0;        // widest single launch seen

    // One submission of `rows` rows. A width below 1 is ignored: a launcher that threw before
    // launching must not be counted, and the launchers' own checks are what make that the only way
    // to reach here with a nonsense width.
    void record(int rows);

    // `calls=… calls_batched=… rows_total=… rows_max=…` — the key=value tail of one `LAUNCHSTAT`
    // line. `gate.py` parses this; keep the key names stable.
    std::string describe() const;
};

// The seven families, one accessor each so a launcher states which one it is feeding.
Family& glue();
Family& proj();
Family& mlp_gateup();
Family& mlp_down();
Family& gdn_conv();         // §04.2 windowed causal conv, one launch per GDN layer per wave
Family& gdn_gates();        // §04.3 decay/write gates, ditto
Family& gdn_gated_norm();   // §04.5 gated RMSNorm, ditto

// One `LAUNCHSTAT family=<name> <describe()>` line per family, as a single string (no trailing
// newline). `decode_run` prints it before its `RESULT` line, so the last line of a run stays the
// result line — `AGENTS.md`'s trap table has a `tail -1` story about what happens otherwise.
std::string describe_run();

}  // namespace engine::launchstats
