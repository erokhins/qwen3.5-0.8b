#pragma once
// §09 / P7 — the batched 2-row VERIFY body, as a measured side pass, not as an accept loop.
//
// WHAT THIS IS. One body of `kRows` rows, fed `[t, d]` at consecutive positions `q, q+1` over the
// request's own populated KV, through `submit_wave_body` — the same §07.1 order the shipping decode step
// runs, at width 2 instead of 1 (`engine/decode_step.h`'s note on the wave body: `submit_token_body` IS
// `submit_wave_body` with n = 1, so there is no second code path through the 24 layers). Its two logits
// rows come from ONE pass over the tied head, downloaded in ONE D2H, and argmaxed twice with §03.4's rule.
// That shape is the whole of P7's economics: `prototype/mtp_delta/D3-RESULT.md` measured 0.1505 of a step
// with it and 0.4729 — KILLED — without it, and two sequential 1-row bodies cost a whole step and are dead
// structurally.
//
// WHAT THIS IS NOT, and the line is deliberate. There is NO ACCEPT LOOP here and nothing in this file
// changes what the run emits: the loop that consumes these two rows (commit the candidate on a match, emit
// the bonus token, advance by two) is the next increment, and the point of building the body first is that
// the riskiest arithmetic in the repo gets measured before any output depends on it. So the body runs on a
// SHADOW state, the target's own state is never dirtied, and `sampled.txt` / every `STATEFP` row must come
// out byte-identical with the flag on and off — the same brutal admission the draft block holds itself to
// (`engine/mtp_draft.h`).
//
// THE ROLLBACK SURFACE, made explicit, because D3 excluded it as out of scope rather than as solved
// (`MTP-IMPL.md` §1's table, ~0.12 ms/round unpriced). A second row dirties four things:
//
//   target KV lane q+1     written by row 1 (row 0's lane q is the same bytes the target already wrote).
//                          Unreachable, not restorable: §08.6 inv. 4 makes `prefix_len` the KV length, so
//                          lane q+1 is above P and no launcher will read it; the next real step overwrites
//                          it with the correct token's K/V before attending. And it is not fingerprinted:
//                          `download_state` compacts KV to `prefix_len` lanes. **0 bytes restored.**
//   GDN state (18 layers)  advanced by BOTH rows, not position-addressed. **bytes copied = sizes()
//                          .gdn_state_bytes = 18.0 MiB.**
//   conv window            rolled by both rows (row 1 overwrites the column row 0 shifted out).
//                          **bytes = sizes().conv_window_bytes.** NOTE: that is 1.27 MiB, not the
//                          0.63 MiB `MTP-IMPL.md` §1 carries — the window is F32, and `engine/state.h`
//                          says so at §08.1 with the correction dated 2026-10-11. So the checkpoint is
//                          19.27 MiB per round, not the 18.66 MiB §1 prices. The bytes this file reports
//                          are DERIVED from `sizes()`, so the doc line cannot make them wrong.
//   per-token scratch      reused, never checkpointed: it is body-scoped, not §08.1 state.
//   `prefix_len`           a field on the shadow. **0 bytes.**
//
// THE FINDING THIS FILE IS OBLIGED TO STATE, NOT DESIGN AROUND. A snapshot taken BEFORE the body restores
// the mixer state to S_q. What a REJECT needs is S_{q+1} — the state after row 0 consumed the real token —
// because the next round's row 0 sits at position q+1 and its mixer input must already include position q.
// Restoring S_q and setting `prefix_len = q+1` skips the token at position q; keeping S_{q+2} keeps the
// rejected candidate's contribution. Neither is S_{q+1}, and `submit_wave_body` cannot hand it over: the
// GDN recurrence launch walks both rows inside ONE kernel, so there is no point in the stream where the
// state is S_{q+1} to copy. So option (i) of `MTP-IMPL.md` §1/§3 ("snapshot before the body, restore on
// reject") does not by itself produce the rejected-round state; it needs either
//   * a per-row split of the recurrence/gdn launches with a snapshot copy between them (no sync needed —
//     the copy is just another stream op — but ~18 extra launches per round at the measured 1.82 µs floor
//     plus the same 19.27 MiB copy), or
//   * the re-forward route (ii), which pays a whole body per reject.
// This file therefore implements and MEASURES the copy (it is the same copy either way, and it is the term
// that was unpriced), and hands the split-vs-re-forward choice back as the decision it is. At the derived
// 19.27 MiB, §3's own arithmetic (snapshot every round + restore on a reject, 0.133 ms rather than the
// 0.1217 ms its smaller figure implied) is what the arms script prices, and `MTP-IMPL.md` §0's `rho` row
// needs the re-derivation before the headline moves.
//
// WHY THE BODY FORCES THE HAND GEMM AND THE SEQUENTIAL SCAN. A one-row decode body is the hand GEMV and
// the sequential GDN scan BY RULE (`resolve_wave_gemm`, and the same one-row rule for `GdnScan`), so those
// are the bits row 0 has to reproduce. A width-2 body under the shipped `--gemm library` default would
// take hipBLAS's own solution for N=2 — legal, in-band, and NOT bit-identical (that is what D-11's band
// is for). Forcing hand + seq makes the verify body's row 0 the same kernel chain the decode step ran,
// which is what makes "row 0 bit-identical to today's step" a claim that can be true at all.
//
// WHY THAT FORCING IS A FLAG NOW, AND WHAT EACH SIDE OWES. `prototype/mtp_accept/ROW-SCALE.md` measured
// that the forcing — not the row count — is the round's cost: the same body over a populated 4k cache runs
// at 1.09–1.13 of a step on the amortising route and at 2.085 on the forced one. So the choice is named
// (`--mtp-route hand|amortised`, `Route` below) rather than baked in.
//
//   `hand` is the DEFAULT and is today's launch sequence bit for bit: the forcing is applied, so flag-off
//       byte-identity against a pre-flag binary stays a byte-compare rather than an argument.
//   `amortised` leaves the scratch's own resolved route in place, i.e. the body runs the route a wide body
//       takes everywhere else. It is NOT owed bit-identity with the one-row step — that is `hand`'s claim,
//       and giving it to both sides would make the switch a no-op. The bands already in `engine/gate.py`
//       judge it; no band is widened and no gate threshold moved for it. Its divergence, whatever it is,
//       is reported (see `prototype/mtp_verify/rowdiff.py`), not gated away.
//
// ONE FORCING THE SWITCH DOES NOT REACH, because it is not a route choice: the mid-round CHECKPOINT
// (`--mtp-accept`, and `--mtp-midsplit 1`) needs a one-token-step `kSeq` recurrence, because at any other
// span one launch walks both rows inside one kernel and "the state after row 0" exists at no address a
// restore could read (`decode_step.cpp`'s `gdn_span` forcing says the same thing from the launcher side).
// So an amortised round over a non-seq scan is refused at the flag rather than silently served a seq scan —
// `decode_run.cpp`'s refusal, and `verify_body`/`accept_round` throw if a checkpoint is ever armed over one.

#include <cstdint>
#include <string>
#include <vector>

#include "engine/decode_step.h"
#include "engine/state.h"

namespace engine::mtpverify {

// The funded width. One value, because one shape was priced; `kernels::lm_head_batched_xbf16` refuses any
// other and says so.
constexpr int kRows = 2;

// Which weight walk and which GDN scan a body of this shape runs — see "WHY THAT FORCING IS A FLAG NOW"
// above for what each side is and owes. One spelling, shared by `verify_body` and `accept_round` (the same
// reason `kRows` is shared rather than duplicated), so the flag and the two bodies cannot drift to three
// different meanings of "the cheap route".
enum class Route { kHand, kAmortised };

// `hand` / `amortised`, parsed or thrown. A value that silently became the default would price one route
// twice and report the second price as the other route's, which is the standing reason every A/B-naming
// flag in this engine parses-or-throws rather than coercing.
Route parse_route(const std::string& name);
const char* route_name(Route r);

// The per-round checkpoint: device buffers the size of everything a second row can dirty and a length
// field cannot undo. Allocated once per run, reused per round, exactly like the draft's arena.
//
// TWO REGIONS, not one, because a restore that has nowhere to land is not a restore. `take` copies the
// request's mixer state into the CHECKPOINT side; the shadow body runs on the SHADOW side; `restore`
// copies checkpoint → shadow, which is the per-reject cost D3 left unpriced. Aliasing the two would make
// `restore` a self-copy that could not fail — the empty-check failure class this repo keeps recording.
struct Snapshot {
    float* gdn = nullptr;            // [kNumGdnLayers][16][128][128] — the checkpoint
    float* conv = nullptr;           // [kNumGdnLayers][3][6144]
    float* shadow_gdn = nullptr;     // the same sizes; what a shadow body dirties and a restore overwrites
    float* shadow_conv = nullptr;
    long long bytes_gdn = 0;         // DERIVED from `state_sizes()` — never typed in
    long long bytes_conv = 0;
    long long bytes() const { return bytes_gdn + bytes_conv; }
};

// Round counters, printed so the cost is a number a reviewer can re-derive rather than a paragraph.
struct Stats {
    long long rounds = 0;
    long long snapshot_bytes = 0;    // bytes copied live→checkpoint over the whole run (`rounds` × bytes())
    long long restored_bytes = 0;    // bytes copied checkpoint→shadow over the whole run — the same product,
                                     // because the shadow has to be seeded every round, not only on a reject.
                                     // A run whose two figures differ has a snapshot or a restore going missing
                                     // (which is the bug this field was added to catch).
    long long guard_bytes = 0;       // bytes downloaded per round by the witnesses, when they are on; reported
                                     // so a timing arm can be shown to be running without them
    long long kv_bytes_left = 0;     // bytes the body left above `prefix_len` (lane q+1's K+V), reported
    long long lane_bytes = 0;        // bytes the anchor-lane revert copied, both ways, over the run — the
                                     // price of running a body whose row 0 re-stages a live lane, printed so a
                                     // timing arm bills it instead of discovering it later
};

Snapshot allocate_snapshot(const RequestState& st);
void free_snapshot(Snapshot& s);

// Copy the request's mixer state INTO the checkpoint side of the snapshot.
void take_snapshot(const RequestState& from, const Snapshot& to);
// Copy the checkpoint side back over the shadow side. This is the restore, and it is `bytes()` of device
// copy per round whether or not anyone ever calls it in a loop.
//
// IT IS NOT OPTIONAL, AND `shadow_state()` CANNOT SUBSTITUTE FOR IT. `shadow_state` repoints the shadow at
// the shadow buffers; it copies nothing, so a body run on a fresh shadow reads whatever `hipMalloc` handed
// back — on round 0 usually a zeroed GDN state, on later rounds the previous round's END state, i.e. two
// tokens downstream of where its first row sits. That is not a numerics gap, it is a row computed from a
// different mixer state, and it is what made this body's row 0 disagree with the step by 1−cos 0.18 while
// writing K/V derived from that wrong state into the request's own lane. `verify_body` therefore restores
// before it submits, and `capture_state_bytes` is the witness that says it did.
void restore_snapshot(const Snapshot& s);

// A request state that shares the target's KV and cap but runs on the snapshot's SHADOW mixer memory and
// starts at `prefix_len`. It copies nothing — which is exactly why the caller must `restore_snapshot` first;
// repointing is not seeding.
RequestState shadow_state(const RequestState& target, const Snapshot& snap, int prefix_len);

// Declared here so `verify_body` can hand one back; the struct and its functions are below.
struct StateBytes;
// Likewise: `verify_body` has to read the tail lane's poison BEFORE its own revert erases the evidence, so
// the type has to be nameable at its signature.
struct PoisonReport;

// Device staging for one KV lane — every attention layer, both tensors, both KV heads, i.e. exactly the
// address set one row's §05.7 append touches (`kv_lane_bytes`). `verify_body` reserves one of these PER ROW,
// saves each row's lane before it submits and writes them back afterwards; the revert note there says why a
// save-and-put-back and not a suppressed write. Allocated once per run beside the snapshot, because a
// `hipMalloc` inside a round would be a synchronisation in the middle of the cost being measured.
struct LaneStage {
    char* stage = nullptr;
    long long bytes = 0;           // total staging: `lanes` × `bytes_per_lane`
    long long bytes_per_lane = 0;  // == kv_lane_bytes(st) at allocation time
    int lanes = 0;                 // one per body row: each row of the body writes exactly one lane
};
LaneStage make_lane_stage(const RequestState& st);
void free_lane_stage(LaneStage& ls);

// One verify round. `anchor_pos` is q, the position of the first row; `tokens` holds `kRows` ids to feed
// at q, q+1. Runs on a shadow built from `snap`, so `target` is not modified in any byte this reports —
// and with `body_reads` non-null the driver also gets back, in bytes, what the shadow actually held at the
// moment the body started, which is the only witness of that claim that is not a restatement of it.
// `logits_dev` is a device buffer of at least kRows*kVocab floats owned by the caller (allocated once).
// `route` decides whether the hand-GEMV / seq-scan forcing is applied (`kHand`, today's launch sequence)
// or the scratch's own resolved route is left in place (`kAmortised`); whatever ended in force comes back
// on `Result::gemm_ran` / `scan_ran`, so the caller prints what ran, not what was asked for.
struct Result {
    std::vector<float> logits;      // [kRows][kVocab], downloaded in ONE blocking D2H
    int token[kRows] = {-1, -1};    // §03.4's argmax per row, ties to the LOWEST id
    float top[kRows] = {0.f, 0.f};
    int head_passes = 0;            // must read 1: the whole point of the batched head
    int d2h_count = 0;              // must read 1
    long long bytes_read_back = 0;  // bytes of logits downloaded
    long long lane_bytes = 0;       // KV bytes reverted per round (each way), so a timing arm can bill it
    // Filled ONLY when `capture_end_state` is set: the shadow's mixer state as the body left it, GDN
    // then conv, in that order. The split admission needs it — "split and batched agree" is a claim
    // about the END state as much as about the logits, and a body that reached the same logits by a
    // different route would be the difference — and nothing else pays for it, because it is 19.27 MiB
    // of D2H per round and no timing arm should be carrying that.
    std::vector<char> end_state;
    // Filled ONLY when `ckpt` is non-null: the mid-round checkpoint itself, GDN then conv. This is the
    // half of S_{p+1} the split exists to produce, and the only way to ask, in bytes, whether it is the
    // right half — the driver compares it against the mixer state the shipping one-row step left.
    std::vector<char> mid_state;
    // Filled ONLY when `verdict_dev` is non-null: the device verdict, in the
    // `mtpverdict::VerdictWord` order, plus the bytes it cost to read back. The admission is that
    // `verdict[kRow0Token] == token[0]` and `verdict[kRow1Token] == token[1]`, i.e. the device argmax
    // agrees with `argmax_greedy` on the same tile, tie rule included.
    int verdict[4] = {-1, -1, -1, -1};
    long long verdict_bytes = 0;
    // The route the body ACTUALLY ran, read off the scratch inside the call. Not a restatement of the
    // flag: the flag proves the parse, these prove the launch sequence, and a stale or inert binary agrees
    // with itself on everything except a field taken from the object the kernels were configured from
    // (`AGENTS.md`: a paired A/B against a stale binary reads as a clean null).
    kernels::WaveGemm gemm_ran = kernels::WaveGemm::kHand;
    kernels::GdnScan scan_ran = kernels::GdnScan::kSeq;
    // And which attention CORE the body ran, same reading and same reason, since
    // `engine/decode_step.h`'s `kMtpRoundAttnWide` made it a compiled-in rule rather than a flag: the
    // round's body takes the 16-row WMMA wave block whatever `--attn-fattn-wide` asked for, and this field
    // is the only witness in the log that it did (`decode_run` prints it as `attn_fattn_wide=`).
    int attn_wide_ran = 0;
};
// `ckpt`, when non-null, arms the M2 mid-round checkpoint for this body (`decode_step.h`'s
// `GdnCheckpoint`): the conv rolls row 0's window into it and the recurrence hands it each layer's
// state after row 0, which is the only way `S_{p+1}` can be produced without re-running the step. It
// is a parameter rather than a field of `Snapshot` because the probe and the accept round own those
// regions differently — the probe's are a witness's, the round's are its rollback.
Result verify_body(const WeightStore& weights, const RequestState& target, DecodeScratch& sc,
                   Route route, const Snapshot& snap, int anchor_pos, const int* tokens,
                   float* logits_dev,
                   LaneStage& lanes, StateBytes* body_reads = nullptr,
                   PoisonReport* tail_poison = nullptr,
                   const DecodeScratch::GdnCheckpoint* ckpt = nullptr,
                   bool capture_end_state = false, int* verdict_dev = nullptr);

// The restore witness: poison the shadow side with a pattern no body writes, restore, and compare every
// byte of both regions against the checkpoint. A PARTIAL restore cannot pass this — the untouched bytes
// keep the poison — which is the point: "the restore ran" is not the claim, "the restore was complete" is.
// Costs two downloads plus a memset of `bytes()`, so the driver calls it once per probe, not per round.
// Returns "" when every byte agrees, else a message naming the first mismatch.
std::string restore_witness(const Snapshot& snap);

// Bytes the body leaves above `prefix_len`: one KV lane's K and V for every attention layer, the only
// thing row 1 wrote that the target will later overwrite. Derived, so the printed figure cannot go stale.
long long kv_bytes_above_prefix(const RequestState& st);

// ---- the operand witness and the live-state guard -------------------------------------------------
//
// WHY THESE EXIST, AND WHAT THEY REFUSE TO DO. A row-0 gap of 1−cos 0.18 is not rounding and cannot be
// argued about; it means a row read a different operand, and the only way to say which is to compare bytes
// at the point of use. Everything below therefore copies bytes and reports the FIRST difference with an
// address. Nothing here computes a metric, so nothing here can be band-judged or widened.
//
// The capture covers the mixer state, the conv window, and K and V for the lanes BELOW the row's own
// position — `[0, prefix_len)`, i.e. exactly the lanes §08.6's invariant 4 says are populated. A row's own
// lane is excluded because it is what that row is about to write, and because the poison below lives there.
struct StateBytes {
    int token = -1;                    // the id the row is fed; -1 when no token is under discussion
    int pos = 0;                       // `prefix_len` at capture time, i.e. the row's position
    int kv_lanes = 0;                  // lanes captured: [0, kv_lanes)
    std::vector<char> gdn, conv, kv;   // `kv` is the per-(layer, K/V, head) regions, concatenated
    // One name per concatenated region, in the order the capture filled `kv`, so a difference is reported
    // as "which layer's K, which head, which lane" instead of an offset the reader has to divide by 512
    // mentally. Same source as the regions themselves, so it cannot describe a layout the capture did not
    // use — which is the only reason a named address is trustworthy.
    std::vector<std::string> kv_names;
};

// Download what a row sitting at `st.prefix_len` would read before its first launch.
StateBytes capture_state_bytes(const RequestState& st, int token);

// "" when two captures agree byte for byte; otherwise the field, the offset inside it and both byte values,
// in the order a body reads them: token, position, GDN state, conv window, KV lanes. A difference in the
// LANES CAPTURED counts as a difference, so a short capture cannot read as "identical over what it saw".
std::string first_state_diff(const StateBytes& a, const StateBytes& b);

// Not just the first difference: how many bytes moved, and which (region, lane) pairs they fall in, capped
// at `max_places` labels. `first_state_diff` answers "where did it start"; this answers "how big is it", and
// the two together are what separate "the body re-staged its own lane" (a construction problem, fixable by
// not computing that row) from "the body wrote all over the cache" (a bug, and an emergency). Prints "same"
// when there is nothing to say.
std::string state_diff_summary(const StateBytes& a, const StateBytes& b, size_t max_places = 4);
// The same comparison over the two mixer regions only. This is the one the driver uses to witness that the
// comparison BITES: the step is required to move the mixer state, so a mixer diff that reads "same" after a
// step is a broken comparator, not a quiet step (the empty-parse failure family, `AGENTS.md`).
std::string first_mixer_diff(const StateBytes& a, const StateBytes& b);

// The byte a poisoned lane holds. Chosen because it is not a plausible activation byte pattern: a lane of
// real K or V is BF16 values, and 0xA5A5 / 0xA5A5 is a negative NaN in every lane of the region.
constexpr unsigned char kPoisonByte = 0xA5;

// Write `kPoisonByte` over one KV lane — K and V, every attention layer, both KV heads. Refuses a lane
// below `prefix_len`: poisoning a lane the request can read would change the run, which is the opposite of
// what a witness is for. A lane at or above `prefix_len` is unread until the next append overwrites it
// (§08.6 inv. 3 and 4), which is what makes poisoning it safe AND makes "still poison after the body" a
// statement about where the body wrote rather than about the cache.
void poison_kv_lane(const RequestState& st, int lane);

// Bytes of a poisoned lane still equal to `kPoisonByte` after the body ran, and how many (layer, K/V, head)
// regions of it were never written at all. A lane the append DID write can still contain 0xA5 bytes by
// coincidence — a real BF16 lane will contain a few, roughly one byte in 256 — so the byte count alone
// cannot be the verdict; `regions_unwritten` is. A region whose every byte is still poison was not written,
// which is the case worth catching: it means the body's staging went somewhere other than its own lane.
struct PoisonReport {
    long long survivors = 0;        // bytes still equal to kPoisonByte across the whole lane
    int regions_unwritten = 0;      // regions whose lane is entirely poison
    int regions = 0;                // regions inspected, so the two above have a denominator
    long long lane_bytes = 0;       // bytes of one region's lane
    bool written() const { return regions_unwritten == 0; }
};
PoisonReport poisoned_lane_report(const RequestState& st, int lane);

// Bytes in one KV lane across the whole request (K and V, all attention layers, both heads). Derived, so
// the printed poison figure and `kv_bytes_above_prefix` cannot disagree about what a lane costs.
long long kv_lane_bytes(const RequestState& st);

}  // namespace engine::mtpverify
