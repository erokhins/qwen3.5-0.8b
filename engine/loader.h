#pragma once
// Weight residency for the text path of Qwen/Qwen3.5-0.8B.
//
// Contract this file implements (nothing more):
//   * §02.5 name normalisation — `model.language_model.* → lm.*`,
//     `mtp.layers.0.* → mtp.layer0.*`, `model.visual.*` skipped (A-09), and a loud
//     failure on any tensor that is neither loaded nor explicitly skipped (§10 gate G0).
//   * one `hipMalloc` + `hipMemcpy` per checkpoint tensor — never `hipMallocManaged`,
//     which degrades silently to ~52 GB/s near the pool limit on this box while
//     `hipMalloc` fails loudly (devices/strix-halo-unified-memory.md).
//   * the tied head (§01.1, §03.4): `lm_head.weight` does not exist in the checkpoint,
//     so it is registered as a second *name* for the embedding allocation — one blob,
//     two names, no second 485 MiB copy.
//   * a warm pass over every resident page before anything is timed or served, because
//     the first GPU touch of fresh pages runs ~35 GB/s instead of ~240 GB/s (same doc).

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine {

enum class Dtype { BF16, F32 };

// §02.5 classification of one checkpoint tensor name.
enum class TensorPlan { Load, SkipVision, Unknown };

// Fills `out` with the normalised engine name when the return value is Load.
TensorPlan normalize_name(const std::string& checkpoint_name, std::string& out);

const char* dtype_name(Dtype d);
size_t dtype_size(Dtype d);

// Streaming SHA-256 (FIPS 180-4). Only consumer is the load-verification pass: hashing
// the bytes that are actually on the device is what proves name map + file offset +
// dtype + tying are simultaneously right.
class Sha256 {
public:
    void update(const void* data, size_t nbytes);
    std::string hex();  // finalises; the object is not reusable after this

private:
    void flush_block();

    uint32_t h_[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    uint64_t total_bytes_ = 0;
    unsigned char buf_[64] = {};
    size_t buflen_ = 0;
    bool finalised_ = false;
};

std::string sha256_hex(const void* data, size_t nbytes);

// One engine-visible tensor. `device` is a raw device pointer (bf16 rows are row-major,
// `shape` is as stored: a linear weight is [out, in], so y = x·Wᵀ — §01 preamble).
struct Weight {
    std::string name;             // normalised name this entry is registered under
    std::string storage_owner;    // normalised name whose hipMalloc backs `device`
    std::string checkpoint_name;  // original name in the shard ("" for an engine alias)
    std::vector<int64_t> shape;   // empty shape + nbytes 0 marks an alias-less placeholder
    Dtype dtype = Dtype::BF16;
    size_t nbytes = 0;
    void* device = nullptr;

    bool is_alias() const { return storage_owner != name; }
    int64_t numel() const;
    std::string shape_string() const;  // "6144x1024", "1024" for 1-D
};

// Everything the loader measured while loading. Timings are wall-clock milliseconds;
// this box's `hipEventElapsedTime` cannot see sub-millisecond work, and these phases are
// all long enough that wall-clock is the honest instrument (prototype/README.md rule 7).
struct LoadStats {
    std::vector<std::string> shard_paths;

    double parse_ms = 0.0;       // safetensors header(s)
    double stage_alloc_ms = 0.0; // the pinned staging buffer (hipHostMalloc is not free)
    double alloc_ms = 0.0;       // the hipMallocs (page reservation, no transfer)
    double stage_read_ms = 0.0;  // shard bytes -> pinned staging (file/page-cache I/O)
    double h2d_ms = 0.0;         // hipMemcpy H2D only, destination pages still fresh
    double warm_ms = 0.0;        // first GPU pass over every resident page
    double steady_h2d_ms = 0.0;  // same bytes uploaded again, pages committed (see steady_bytes)
    double d2h_ms = 0.0;         // copy-back, verification only
    double verify_ms = 0.0;      // D2H + SHA-256, only when verification is on
    double total_ms = 0.0;       // whole load(), verification excluded

    size_t header_bytes = 0;
    size_t stage_bytes = 0;      // the largest tensor, which is what staging must hold
    size_t uploaded_bytes = 0;   // bytes hipMalloc'd (an alias copies nothing)
    size_t steady_bytes = 0;     // bytes the steady-H2D measurement moved
    size_t warm_bytes = 0;
    size_t n_loaded = 0;          // checkpoint tensors claimed
    size_t n_skipped_vision = 0;  // model.visual.* (§01.8)

    size_t device_total_bytes = 0;
    size_t free_before_bytes = 0;
    size_t free_after_stage_bytes = 0;   // after the pinned staging buffer exists
    size_t free_after_upload_bytes = 0;
    size_t free_after_warm_bytes = 0;

    bool device_verified = false;
    size_t n_verified = 0;
    std::vector<std::string> verify_failures;  // human-readable, empty == clean

    unsigned long long warm_checksum = 0;  // non-zero proves the warm pass read bytes
};

class WeightStore {
public:
    struct Options {
        std::string model_dir = "models/Qwen3.5-0.8B";
        // Copy every resident blob back once and SHA-256 it against the file. ~1.55 GB
        // of D2H, so it is a verification-time cost only (engine/loader_test.cpp).
        bool verify_device_bytes = false;
        // Re-upload once after warming to separate cold-H2D (first-touch destination) from
        // the steady 76-82 GB/s band. Reads the same bytes back into the same allocations.
        bool measure_steady_h2d = true;
    };

    // Resolves shards (model.safetensors.index.json — A-09 — else any *.safetensors in
    // the directory), uploads the text path, warms the pages. Throws std::runtime_error
    // on any tensor it cannot classify, on a name collision, and on a HIP failure.
    static WeightStore load(const Options& opts, LoadStats& stats);

    ~WeightStore();
    WeightStore(const WeightStore&) = delete;
    WeightStore& operator=(const WeightStore&) = delete;
    WeightStore(WeightStore&&) noexcept;
    WeightStore& operator=(WeightStore&&) noexcept;

    const Weight& at(std::string_view normalized_name) const;
    const Weight* find(std::string_view normalized_name) const;

    size_t lookup_count() const { return index_.size(); }  // names, aliases included
    size_t allocation_count() const { return n_allocations_; }
    size_t resident_bytes() const { return resident_bytes_; }
    size_t skipped_vision_count() const { return skipped_vision_; }
    // checkpoint name → normalised name, sorted; the forward direction of the map.
    const std::vector<std::pair<std::string, std::string>>& ckpt_to_engine() const {
        return ckpt_to_engine_;
    }

private:
    WeightStore() = default;  // only load() fills one

    struct Slot {
        Weight w;
        std::string shard_path;
        uint64_t file_off = 0;
    };

    void free_all();

    std::vector<Slot> slots_;
    std::unordered_map<std::string, size_t> index_;
    std::vector<std::pair<std::string, std::string>> ckpt_to_engine_;
    size_t n_allocations_ = 0;
    size_t resident_bytes_ = 0;
    size_t skipped_vision_ = 0;
};

}  // namespace engine
