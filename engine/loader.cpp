#include "loader.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <unistd.h>
#include <unordered_set>

#define HIP_TRY(expr)                                                                  \
    do {                                                                               \
        hipError_t hip_try_rc_ = (expr);                                               \
        if (hip_try_rc_ != hipSuccess) {                                               \
            throw std::runtime_error(std::string("HIP failure at ") + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + ": " + #expr + " -> " + \
                                     hipGetErrorName(hip_try_rc_) + " (" +             \
                                     hipGetErrorString(hip_try_rc_) + ")");            \
        }                                                                              \
    } while (0)

namespace engine {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error(msg); }

bool starts_with(const std::string& s, const char* prefix) {
    const size_t n = std::strlen(prefix);
    return s.size() >= n && s.compare(0, n, prefix) == 0;
}

// ---------------------------------------------------------------------------
// Minimal JSON reader. The safetensors header and model.safetensors.index.json are
// flat, machine-written documents; a full parser would be more machinery than those
// two need. Everything skipped is still skipped *checked*, so a malformed header is
// an error rather than a silently missing tensor.
// ---------------------------------------------------------------------------
class JsonReader {
public:
    JsonReader(const char* data, size_t size) : p_(data), end_(data + size) {}

    void ws() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
    }
    bool at(char c) const { return p_ < end_ && *p_ == c; }
    char peek() {
        if (p_ >= end_) fail("JSON: unexpected end of input");
        return *p_;
    }
    char next() {
        if (p_ >= end_) fail("JSON: unexpected end of input");
        return *p_++;
    }
    void expect(char c) {
        ws();
        if (peek() != c) fail(std::string("JSON: expected '") + c + "' got '" + peek() + "'");
        ++p_;
    }

    std::string string_value() {
        ws();
        expect('"');
        std::string out;
        for (;;) {
            const char c = next();
            if (c == '"') return out;
            if (c != '\\') { out.push_back(c); continue; }
            const char e = next();
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'n': out.push_back('\n'); break;
                case 't': out.push_back('\t'); break;
                case 'r': out.push_back('\r'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'u': {
                    // Header keys we care about are ASCII; keep the code point if it is.
                    unsigned cp = 0;
                    for (int i = 0; i < 4; ++i) {
                        const char h = next();
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
                        else fail("JSON: bad \\u escape");
                    }
                    out.push_back(cp < 0x80 ? (char)cp : '?');
                    break;
                }
                default: fail("JSON: bad escape");
            }
        }
    }

    long long int_value() {
        ws();
        const char* start = p_;
        if (p_ < end_ && *p_ == '-') ++p_;
        while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        const std::string tok(start, (size_t)(p_ - start));
        if (tok.empty()) fail("JSON: expected an integer");
        errno = 0;
        const long long v = std::strtoll(tok.c_str(), nullptr, 10);
        if (errno == ERANGE) fail("JSON: integer out of range: " + tok);
        return v;
    }

    // After '{': false once the object is closed (the '}' is consumed); otherwise the
    // reader sits on the member's value.
    bool next_member(std::string& key) {
        ws();
        if (at(',')) { ++p_; ws(); }
        if (at('}')) { ++p_; return false; }
        key = string_value();
        expect(':');
        return true;
    }
    bool next_element() {
        ws();
        if (at(',')) { ++p_; ws(); }
        if (at(']')) { ++p_; return false; }
        return true;
    }

    std::vector<long long> int_array() {
        expect('[');
        std::vector<long long> out;
        while (next_element()) out.push_back(int_value());
        return out;
    }

    void skip_value() {
        ws();
        switch (peek()) {
            case '{': {
                expect('{');
                std::string k;
                while (next_member(k)) skip_value();
                return;
            }
            case '[': {
                expect('[');
                while (next_element()) skip_value();
                return;
            }
            case '"': string_value(); return;
            default:
                // true / false / null first, then a number. A miss must not be an error
                // here: numbers are the common case (data_offsets, total_size).
                if (try_literal("true") || try_literal("false") || try_literal("null")) return;
                int_value();
                return;
        }
    }

private:
    bool try_literal(const char* lit) {
        const size_t n = std::strlen(lit);
        if ((size_t)(end_ - p_) < n || std::strncmp(p_, lit, n) != 0) return false;
        p_ += n;
        return true;
    }

    const char* p_;
    const char* end_;
};

struct FileTensor {
    std::string name;
    std::string dtype_str;
    std::vector<int64_t> shape;
    uint64_t off_begin = 0;
    uint64_t off_end = 0;
};

Dtype parse_dtype(const std::string& s, const std::string& tensor) {
    if (s == "BF16") return Dtype::BF16;
    if (s == "F32") return Dtype::F32;
    fail("loader: tensor " + tensor + " has dtype " + s +
         ", which the text path does not expect (§01.6: only BF16 and F32 exist)");
}

size_t file_size(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) fail("loader: cannot open " + path);
    return (size_t)f.tellg();
}

struct FileCloser {
    FILE* f = nullptr;
    FileCloser() = default;
    explicit FileCloser(FILE* p) : f(p) {}
    ~FileCloser() { close(); }
    FileCloser(const FileCloser&) = delete;
    FileCloser& operator=(const FileCloser&) = delete;
    FILE* get() const { return f; }
    void reset(FILE* p) { close(); f = p; }
    void close() {
        if (f) fclose(f);
        f = nullptr;
    }
};

// Reads the 8-byte little-endian header length, then the header itself.
std::vector<char> read_safetensors_header(FILE* f, const std::string& path, size_t* header_bytes) {
    unsigned char lenbuf[8];
    if (fread(lenbuf, 1, 8, f) != 8) fail("loader: " + path + " is too short for a header");
    uint64_t hlen = 0;
    for (int i = 0; i < 8; ++i) hlen |= (uint64_t)lenbuf[i] << (8 * i);
    if (hlen == 0 || hlen > (16ull << 20)) fail("loader: implausible header length in " + path);
    std::vector<char> json((size_t)hlen + 1, '\0');
    if (fread(json.data(), 1, (size_t)hlen, f) != (size_t)hlen)
        fail("loader: truncated header in " + path);
    *header_bytes = (size_t)hlen;
    return json;
}

std::vector<FileTensor> parse_header(const std::vector<char>& json, const std::string& path,
                                     size_t data_start, size_t file_bytes) {
    JsonReader jr(json.data(), json.size() - 1);
    std::vector<FileTensor> out;
    jr.expect('{');
    std::string key;
    while (jr.next_member(key)) {
        if (key == "__metadata__") { jr.skip_value(); continue; }
        FileTensor t;
        t.name = key;
        jr.expect('{');
        std::string mk;
        while (jr.next_member(mk)) {
            if (mk == "dtype") {
                t.dtype_str = jr.string_value();
            } else if (mk == "shape") {
                for (long long d : jr.int_array()) t.shape.push_back((int64_t)d);
            } else if (mk == "data_offsets") {
                std::vector<long long> o = jr.int_array();
                if (o.size() != 2)
                    fail("loader: " + path + ": " + key + " has no two-element data_offsets");
                t.off_begin = (uint64_t)o[0];
                t.off_end = (uint64_t)o[1];
            } else {
                jr.skip_value();
            }
        }
        // The same arithmetic the authority tool asserts (tools/weights_report.py::size_of).
        const size_t bytes = (size_t)(t.off_end - t.off_begin);
        const size_t esz = dtype_size(parse_dtype(t.dtype_str, key));
        int64_t numel = 1;
        for (int64_t d : t.shape) numel *= d;
        if (bytes != (size_t)numel * esz)
            fail("loader: " + path + ": " + key + " offsets/shape/dtype disagree: " +
                 std::to_string(bytes) + " bytes vs " + std::to_string(numel) + " x " +
                 std::to_string(esz));
        if (data_start + t.off_end > file_bytes)
            fail("loader: " + path + ": " + key + " data range runs past the end of the file");
        out.push_back(std::move(t));
    }
    return out;
}

std::vector<std::string> resolve_shards(const std::string& model_dir) {
    std::vector<std::string> shards;
    std::unordered_set<std::string> seen;
    std::ifstream ix(model_dir + "/model.safetensors.index.json");
    if (ix) {
        // This checkpoint's shard is `model.safetensors-00001-of-00001.safetensors`, not
        // `model.safetensors`; A-09 says take the shard list from the index.
        const std::string text((std::istreambuf_iterator<char>(ix)),
                              std::istreambuf_iterator<char>());
        JsonReader jr(text.data(), text.size());
        jr.expect('{');
        std::string key;
        while (jr.next_member(key)) {
            if (key != "weight_map") { jr.skip_value(); continue; }
            jr.expect('{');
            std::string tensor_name;
            while (jr.next_member(tensor_name)) {
                const std::string shard_file = jr.string_value();
                if (seen.insert(shard_file).second) shards.push_back(model_dir + "/" + shard_file);
            }
        }
    } else {
        DIR* d = opendir(model_dir.c_str());
        if (!d) fail("loader: cannot open model directory " + model_dir);
        while (struct dirent* e = readdir(d)) {
            const std::string n = e->d_name;
            if (n.size() > 13 && n.compare(n.size() - 13, 13, ".safetensors") == 0)
                shards.push_back(model_dir + "/" + n);
        }
        closedir(d);
    }
    if (shards.empty()) fail("loader: no safetensors shard found under " + model_dir);
    std::sort(shards.begin(), shards.end());
    return shards;
}

// ---------------------------------------------------------------------------
// Warm pass: read every resident byte once so the pages fault in here instead of
// inside the first decode kernel — first GPU touch runs ~35 GB/s, steady ~240 GB/s
// (devices/strix-halo-unified-memory.md). Same shape as the read kernel in
// tools/bw_probe_hip.cpp: grid-stride over 16-byte words, plus a byte tail.
// ---------------------------------------------------------------------------
__global__ void warm_read_kernel(const unsigned char* __restrict__ base, size_t nbytes,
                                 unsigned long long* __restrict__ out) {
    const uint4* v = reinterpret_cast<const uint4*>(base);
    const size_t n16 = nbytes / 16;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned long long acc = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
        const uint4 x = v[i];
        acc += (unsigned long long)x.x + x.y + x.z + x.w;
    }
    const size_t rem = nbytes - n16 * 16;
    if (blockIdx.x == 0 && threadIdx.x < rem) acc += base[n16 * 16 + threadIdx.x];

    __shared__ unsigned long long sh[256];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int off = (int)blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < (unsigned)off) sh[threadIdx.x] += sh[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sh[0]);
}

}  // namespace

const char* dtype_name(Dtype d) { return d == Dtype::BF16 ? "BF16" : "F32"; }
size_t dtype_size(Dtype d) { return d == Dtype::BF16 ? 2 : 4; }

int64_t Weight::numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return shape.empty() ? 0 : n;
}

std::string Weight::shape_string() const {
    std::string s;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) s += "x";
        s += std::to_string(shape[i]);
    }
    return s;
}

// §02.5, verbatim. Anything this does not classify is an error at the call site,
// never a silent skip (§02.5 / §10 gate G0).
TensorPlan normalize_name(const std::string& checkpoint_name, std::string& out) {
    static const char* kVisual = "model.visual.";
    static const char* kLm = "model.language_model.";
    static const char* kMtpLayer0 = "mtp.layers.0.";
    if (starts_with(checkpoint_name, kVisual)) return TensorPlan::SkipVision;
    if (starts_with(checkpoint_name, kLm)) {
        out = "lm." + checkpoint_name.substr(std::strlen(kLm));
        return TensorPlan::Load;
    }
    if (starts_with(checkpoint_name, kMtpLayer0)) {
        out = "mtp.layer0." + checkpoint_name.substr(std::strlen(kMtpLayer0));
        return TensorPlan::Load;
    }
    if (starts_with(checkpoint_name, "mtp.")) {
        out = checkpoint_name;
        return TensorPlan::Load;
    }
    return TensorPlan::Unknown;
}

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4). Plain by design: its only job is proving that the bytes on
// the device are the bytes in the file, so being obviously correct beats being fast.
// `loader_test.cpp` runs the FIPS "abc" vector before trusting it.
// ---------------------------------------------------------------------------
namespace {
constexpr uint32_t kK[64] = {
    // floor(frac(cbrt(p)) * 2^32) for the first 64 primes, derived rather than typed from
    // memory: a wrong entry here silently breaks every hash, and 64 hand-copied hex words
    // is exactly where that happens. The loader_test FIPS vectors are the check.
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

inline uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
}  // namespace

void Sha256::flush_block() {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)buf_[4 * i] << 24) | ((uint32_t)buf_[4 * i + 1] << 16) |
               ((uint32_t)buf_[4 * i + 2] << 8) | (uint32_t)buf_[4 * i + 3];
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6],
             hh = h_[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t t1 = hh + S1 + ch + kK[i] + w[i];
        const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
    h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += hh;
}

void Sha256::update(const void* data, size_t nbytes) {
    if (finalised_) fail("Sha256: update() after hex()");
    const unsigned char* p = (const unsigned char*)data;
    total_bytes_ += nbytes;
    while (nbytes) {
        const size_t take = std::min(nbytes, (size_t)(64 - buflen_));
        std::memcpy(buf_ + buflen_, p, take);
        buflen_ += take; p += take; nbytes -= take;
        if (buflen_ == 64) { flush_block(); buflen_ = 0; }
    }
}

std::string Sha256::hex() {
    if (finalised_) fail("Sha256: hex() called twice");
    const uint64_t bits = total_bytes_ * 8ull;
    const size_t padlen = (buflen_ < 56) ? (56 - buflen_) : (120 - buflen_);
    std::vector<unsigned char> pad(padlen + 8, 0);
    pad[0] = 0x80;
    for (int i = 0; i < 8; ++i) pad[padlen + i] = (unsigned char)(bits >> (56 - 8 * i));
    update(pad.data(), pad.size());
    finalised_ = true;
    static const char* kHex = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (int i = 0; i < 8; ++i)
        for (int s = 28; s >= 0; s -= 8) {
            out.push_back(kHex[(h_[i] >> s) & 0xfu]);
            out.push_back(kHex[(h_[i] >> (s - 4)) & 0xfu]);
        }
    return out;
}

std::string sha256_hex(const void* data, size_t nbytes) {
    Sha256 s;
    s.update(data, nbytes);
    return s.hex();
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------
WeightStore::WeightStore(WeightStore&&) noexcept = default;
WeightStore& WeightStore::operator=(WeightStore&&) noexcept = default;

WeightStore WeightStore::load(const Options& opts, LoadStats& st) {
    const auto t_total = Clock::now();
    WeightStore store;
    HIP_TRY(hipSetDevice(0));
    size_t free_bytes = 0, total_bytes = 0;
    HIP_TRY(hipMemGetInfo(&free_bytes, &total_bytes));
    st.free_before_bytes = free_bytes;
    st.device_total_bytes = total_bytes;

    // ---- phase 1: parse, then classify every tensor in every shard --------------
    auto t_parse = Clock::now();
    st.shard_paths = resolve_shards(opts.model_dir);
    struct Pending {
        FileTensor t;
        std::string norm;
        std::string shard;
        uint64_t file_off = 0;
    };
    std::vector<Pending> pending;
    std::unordered_set<std::string> claimed;
    for (const std::string& shard : st.shard_paths) {
        const size_t fbytes = file_size(shard);
        FileCloser fp(fopen(shard.c_str(), "rb"));
        if (!fp.get()) fail("loader: cannot open " + shard);
        size_t hlen = 0;
        std::vector<char> json = read_safetensors_header(fp.get(), shard, &hlen);
        st.header_bytes += hlen;
        for (FileTensor& t : parse_header(json, shard, 8 + hlen, fbytes)) {
            std::string norm;
            switch (normalize_name(t.name, norm)) {
                case TensorPlan::SkipVision:
                    ++st.n_skipped_vision;
                    continue;
                case TensorPlan::Load:
                    break;
                case TensorPlan::Unknown:
                    fail("loader: " + shard + ": tensor " + t.name +
                         " is neither in the §02.5 load set nor an explicitly skipped "
                         "vision tensor (gate G0: every tensor must be claimed)");
            }
            if (!claimed.insert(norm).second)
                fail("loader: two checkpoint tensors normalise to " + norm);
            const uint64_t file_off = 8 + hlen + t.off_begin;
            pending.push_back({std::move(t), std::move(norm), shard, file_off});
        }
    }
    st.parse_ms = ms_since(t_parse);
    if (pending.empty()) fail("loader: no text-path tensors found under " + opts.model_dir);

    // ---- phase 2: one hipMalloc + one hipMemcpy per tensor ---------------------
    // hipMalloc, never hipMallocManaged: managed memory degrades silently near the
    // pool limit on this box, hipMalloc fails loudly (devices doc, allocation rule).
    size_t max_tensor = 0;
    for (const Pending& p : pending)
        max_tensor = std::max(max_tensor, (size_t)(p.t.off_end - p.t.off_begin));
    st.stage_bytes = max_tensor;
    char* stage = nullptr;
    unsigned long long* dev_sum = nullptr;
    {
        const auto t_stage = Clock::now();
        HIP_TRY(hipHostMalloc((void**)&stage, max_tensor, hipHostMallocDefault));
        HIP_TRY(hipMalloc((void**)&dev_sum, sizeof(unsigned long long)));
        st.stage_alloc_ms = ms_since(t_stage);
    }
    // Snapshot after the pinned staging buffer exists: it is device-visible too, so
    // taking this here is what lets the report say what the weights actually cost.
    HIP_TRY(hipMemGetInfo(&free_bytes, &total_bytes));
    st.free_after_stage_bytes = free_bytes;

    auto t_alloc = Clock::now();
    for (const Pending& p : pending) {
        const size_t nbytes = (size_t)(p.t.off_end - p.t.off_begin);
        Slot slot;
        slot.w.name = p.norm;
        slot.w.storage_owner = p.norm;
        slot.w.checkpoint_name = p.t.name;
        slot.w.shape = p.t.shape;
        slot.w.dtype = parse_dtype(p.t.dtype_str, p.t.name);
        slot.w.nbytes = nbytes;
        slot.shard_path = p.shard;
        slot.file_off = p.file_off;
        HIP_TRY(hipMalloc(&slot.w.device, nbytes));
        store.index_.emplace(slot.w.name, store.slots_.size());
        store.ckpt_to_engine_.emplace_back(p.t.name, p.norm);
        store.slots_.push_back(std::move(slot));
        st.uploaded_bytes += nbytes;
        ++st.n_loaded;
        ++store.n_allocations_;
        store.resident_bytes_ += nbytes;
    }
    st.alloc_ms = ms_since(t_alloc);
    store.skipped_vision_ = st.n_skipped_vision;
    std::sort(store.ckpt_to_engine_.begin(), store.ckpt_to_engine_.end());

    // One pass of read-then-copy per tensor, and a single sync at the end: timing each
    // copy with its own sync measures the driver's page-population call per allocation,
    // not the transfer the engine actually pays for. Each shard is opened once, so the
    // timed loop moves bytes and nothing else.
    std::map<std::string, std::unique_ptr<FileCloser>> open_shards;
    const auto shard_fp = [&](const std::string& path) -> FILE* {
        auto it = open_shards.find(path);
        if (it == open_shards.end()) {
            auto fc = std::make_unique<FileCloser>(fopen(path.c_str(), "rb"));
            if (!fc->get()) fail("loader: cannot open " + path);
            it = open_shards.emplace(path, std::move(fc)).first;
        }
        return it->second->get();
    };
    {
        const auto t_loop = Clock::now();
        for (Slot& s : store.slots_) {
            if (fseeko(shard_fp(s.shard_path), (off_t)s.file_off, SEEK_SET) != 0)
                fail("loader: seek past end of " + s.shard_path);
            const auto t_read = Clock::now();
            if (fread(stage, 1, s.w.nbytes, shard_fp(s.shard_path)) != s.w.nbytes)
                fail("loader: short read while loading " + s.w.name);
            st.stage_read_ms += ms_since(t_read);
            HIP_TRY(hipMemcpy(s.w.device, stage, s.w.nbytes, hipMemcpyHostToDevice));
        }
        HIP_TRY(hipDeviceSynchronize());
        st.h2d_ms = ms_since(t_loop) - st.stage_read_ms;
    }
    HIP_TRY(hipMemGetInfo(&free_bytes, &total_bytes));
    st.free_after_upload_bytes = free_bytes;

    // The tied head (§01.1): the checkpoint carries no `lm_head.weight`, so the engine
    // registers a second *name* for the embedding allocation — one blob, two names,
    // and no second 485 MiB copy of the table.
    {
        const auto it = store.index_.find("lm.embed_tokens.weight");
        if (it == store.index_.end())
            fail("loader: no lm.embed_tokens.weight for lm_head to tie to (§01.1)");
        if (store.index_.count("lm_head.weight"))
            fail("loader: the checkpoint carries lm_head.weight, so the head is not tied "
                 "and this alias would be wrong (§01.1)");
        const Weight& emb = store.slots_[it->second].w;
        Slot alias;
        alias.w.name = "lm_head.weight";
        alias.w.storage_owner = "lm.embed_tokens.weight";
        alias.w.checkpoint_name = "";  // engine-side name; no checkpoint tensor behind it
        alias.w.shape = emb.shape;
        alias.w.dtype = emb.dtype;
        alias.w.nbytes = emb.nbytes;
        alias.w.device = emb.device;
        store.index_.emplace(alias.w.name, store.slots_.size());
        store.slots_.push_back(std::move(alias));
    }

    // ---- phase 3: one GPU pass over every resident page -------------------------
    auto t_warm = Clock::now();
    HIP_TRY(hipMemset(dev_sum, 0, sizeof(unsigned long long)));
    for (const Slot& s : store.slots_) {
        if (s.w.is_alias()) continue;  // same pages as its owner; already covered
        st.warm_bytes += s.w.nbytes;
        const size_t n16 = s.w.nbytes / 16;
        size_t blocks = (n16 + 255) / 256;
        if (blocks < 1) blocks = 1;
        if (blocks > 800) blocks = 800;
        warm_read_kernel<<<(unsigned)blocks, 256>>>((const unsigned char*)s.w.device,
                                                    s.w.nbytes, dev_sum);
        HIP_TRY(hipGetLastError());
    }
    HIP_TRY(hipDeviceSynchronize());
    st.warm_ms = ms_since(t_warm);
    HIP_TRY(hipMemcpy(&st.warm_checksum, dev_sum, sizeof(unsigned long long), hipMemcpyDeviceToHost));
    HIP_TRY(hipMemGetInfo(&free_bytes, &total_bytes));
    st.free_after_warm_bytes = free_bytes;

    // ---- phase 4 (optional): steady H2D with the destination pages committed ----
    // Re-uploads one large blob (the embedding) rather than the whole checkpoint: the
    // 76-82 GB/s band in devices/ was measured on big buffers, and per-tensor copies of
    // 32-byte norms measure call overhead, not bandwidth. Untimed re-read from page cache.
    if (opts.measure_steady_h2d) {
        const auto it = store.index_.find("lm.embed_tokens.weight");
        const Slot& emb = store.slots_[it->second];
        if (fseeko(shard_fp(emb.shard_path), (off_t)emb.file_off, SEEK_SET) != 0)
            fail("loader: seek past end of " + emb.shard_path);
        if (fread(stage, 1, emb.w.nbytes, shard_fp(emb.shard_path)) != emb.w.nbytes)
            fail("loader: short read in " + emb.shard_path);
        for (int rep = 0; rep < 3; ++rep) {
            auto t_copy = Clock::now();
            HIP_TRY(hipMemcpy(emb.w.device, stage, emb.w.nbytes, hipMemcpyHostToDevice));
            HIP_TRY(hipDeviceSynchronize());
            st.steady_h2d_ms += ms_since(t_copy);
            st.steady_bytes += emb.w.nbytes;
        }
        // The steady pass rewrote the embedding with the same bytes it already held;
        // nothing downstream can tell the difference, and verification (if any) re-reads
        // the device afterwards either way.
    }

    st.total_ms = ms_since(t_total);

    // ---- phase 5 (optional): SHA-256 the resident bytes against the file --------
    if (opts.verify_device_bytes) {
        st.device_verified = true;
        auto t_verify = Clock::now();
        std::vector<char> chunk(1u << 20);
        for (const Slot& s : store.slots_) {
            if (s.w.is_alias()) continue;
            Sha256 file_sha;
            if (fseeko(shard_fp(s.shard_path), (off_t)s.file_off, SEEK_SET) != 0)
                fail("loader: seek past end of " + s.shard_path);
            {
                size_t left = s.w.nbytes;
                while (left) {
                    const size_t take = std::min(left, chunk.size());
                    if (fread(chunk.data(), 1, take, shard_fp(s.shard_path)) != take)
                        fail("loader: short read in " + s.shard_path);
                    file_sha.update(chunk.data(), take);
                    left -= take;
                }
            }
            auto t_d2h = Clock::now();
            HIP_TRY(hipMemcpy(stage, s.w.device, s.w.nbytes, hipMemcpyDeviceToHost));
            HIP_TRY(hipDeviceSynchronize());
            st.d2h_ms += ms_since(t_d2h);
            const std::string want = file_sha.hex();
            const std::string got = sha256_hex(stage, s.w.nbytes);
            ++st.n_verified;
            if (got != want)
                st.verify_failures.push_back(s.w.name + " [" + s.w.checkpoint_name +
                                             "]: device " + got + " != file " + want);
        }
        st.verify_ms = ms_since(t_verify);
    }

    HIP_TRY(hipHostFree(stage));
    HIP_TRY(hipFree(dev_sum));
    return store;
}

void WeightStore::free_all() {
    for (Slot& s : slots_) {
        if (s.w.is_alias()) continue;  // owned by another slot; a second free would corrupt
        if (s.w.device) (void)hipFree(s.w.device);
        s.w.device = nullptr;
    }
}

WeightStore::~WeightStore() { free_all(); }

const Weight* WeightStore::find(std::string_view normalized_name) const {
    const auto it = index_.find(std::string(normalized_name));
    return it == index_.end() ? nullptr : &slots_[it->second].w;
}

const Weight& WeightStore::at(std::string_view normalized_name) const {
    const Weight* w = find(normalized_name);
    if (!w) throw std::out_of_range("WeightStore: no tensor named " + std::string(normalized_name));
    return *w;
}

}  // namespace engine
