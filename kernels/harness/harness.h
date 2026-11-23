// Benchmark harness shared by every kernel binary (PLAN.md section 3.7). Header only.
//
// A benchmark registers Ops (one Op = one whole operation, e.g. V1's three launches) and calls
// Runner::run_all(). Ops are grouped; a group's buffers must fit in device memory together. Per group:
//
//   1. setup every op; record device memory. A CUDA error (e.g. out of memory) marks the op "failed"
//      with its stage and message, and the run continues. If the error left the context unusable (an
//      illegal memory access, say), every op not yet timed is "skipped": on this driver a process cannot
//      create a new context after such an error, even after cudaDeviceReset (checked 2026-11-23), so
//      those ops must be rerun in a new process.
//   2. check every op against its reference; an op whose check fails is "check_failed" and never timed.
//   3. warm up all ops until the SM clock is steady (three readings within CLOCK_TOL, at least
//      WARMUP_MIN_S, at most WARMUP_MAX_S).
//   4. calibrate iteration counts once per op (a sample lasts about TARGET_SAMPLE_MS), then freeze them.
//   5. take SAMPLES samples per op, interleaved: each round runs the group's ops in a fresh order from a
//      seeded shuffle (rotation alone keeps each op behind the same neighbour, and a kernel's speed can
//      depend on what ran just before it: power headroom). Each sample records its predecessor.
//      Each sample measures, per iteration:
//        event_batch_ms  CUDA events around ITERS back-to-back iterations (kernel time incl. launch gaps)
//        host_sync_ms    host clock around launch + stream synchronise, one iteration at a time
//        phase_<name>_ms CUDA events around each declared phase (optional, for X2's breakdown)
//      NVML (SM and memory clock, temperature, clock-event reasons) is read after each sample,
//      outside the timed regions.
//   6. flag samples: a thermal or hardware-slowdown reason, or an SM clock more than THROTTLE_DROP below
//      the op's own median clock (not the group's: memory-bound ops clock higher than power-capped
//      compute-bound ones, and a group median would flag the latter wholesale). The software power cap is this laptop's steady state under compute load:
//      recorded, not flagged. An op with more than half its samples flagged is "invalid_throttled".
//   7. derive rates from declared work (FLOPs, DRAM bytes) and flag rates above the measured limits
//      (results/cuda/peak.json) as implausible: the kernel is probably not doing the declared work.
//   8. teardown.
//
// Output: one JSON file per benchmark binary, schema "airgap-bench/1" (kernels/harness/SCHEMA.md),
// with every sample, so figures can be regenerated from the file. A Python twin (for the PyTorch
// baselines) must write the same schema; C++ and Python ops cannot be interleaved in one process.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <nvml.h>

namespace harness {

// Protocol constants (also in SCHEMA.md).
constexpr int SAMPLES = 21;
constexpr double TARGET_SAMPLE_MS = 20.0;
constexpr double WARMUP_MIN_S = 1.0;
constexpr double WARMUP_MAX_S = 10.0;
constexpr double CLOCK_TOL = 0.01;
constexpr double THROTTLE_DROP = 0.03;
constexpr double RATE_SLACK = 1.05;
constexpr int PHASE_ITERS_MAX = 200;
constexpr unsigned SHUFFLE_SEED = 20261123;

struct CudaError : std::runtime_error {
    cudaError_t code;
    CudaError(cudaError_t c, const std::string& msg) : std::runtime_error(msg), code(c) {}
};

inline void check_cuda(cudaError_t e, const char* what, const char* file, int line) {
    if (e != cudaSuccess) {
        throw CudaError(e, std::string(cudaGetErrorName(e)) + ": " + cudaGetErrorString(e) + " in " + what + " at " +
                               file + ":" + std::to_string(line));
    }
}
#define HCHECK(x) ::harness::check_cuda((x), #x, __FILE__, __LINE__)

struct CheckResult {
    bool ok = true;
    double max_abs_err = 0, max_rel_err = 0;
    std::string note;
};

using Launch = std::function<void(cudaStream_t)>;

struct Op {
    std::string name;
    std::string group = "default";
    std::vector<std::pair<std::string, std::string>> params;
    double flops = 0;       // declared work per iteration
    double dram_bytes = 0;  // declared DRAM traffic per iteration
    std::function<void()> setup, teardown;
    Launch run;                                          // one whole iteration, asynchronous on the stream
    std::vector<std::pair<std::string, Launch>> phases;  // optional breakdown; together they must equal run
    std::function<CheckResult()> check;
};

struct Limits {
    double tflops = 0, dram_gb_per_s = 0;
};

// Reads the sustained FMA rate and DRAM read bandwidth from results/cuda/peak.json (no JSON library:
// looks for the keys this repository writes).
inline Limits read_limits(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    auto number_after = [&](const std::string& section, const std::string& key) -> double {
        size_t p = s.find("\"" + section + "\"");
        if (p == std::string::npos) return 0;
        p = s.find("\"" + key + "\"", p);
        if (p == std::string::npos) return 0;
        p = s.find(':', p);
        return std::atof(s.c_str() + p + 1);
    };
    Limits l;
    l.tflops = number_after("fma_sustained", "tflops_settled");
    l.dram_gb_per_s = number_after("dram_read", "gb_per_s");
    return l;
}

struct Snap {
    unsigned sm = 0, mem = 0, temp = 0;
    unsigned long long reasons = 0;
};

inline std::vector<std::string> reason_names(unsigned long long r) {
    static const std::pair<unsigned long long, const char*> names[] = {
        {nvmlClocksEventReasonGpuIdle, "idle"},
        {nvmlClocksEventReasonApplicationsClocksSetting, "app_clocks"},
        {nvmlClocksEventReasonSwPowerCap, "sw_power_cap"},
        {nvmlClocksThrottleReasonHwSlowdown, "hw_slowdown"},
        {nvmlClocksEventReasonSyncBoost, "sync_boost"},
        {nvmlClocksEventReasonSwThermalSlowdown, "sw_thermal"},
        {nvmlClocksThrottleReasonHwThermalSlowdown, "hw_thermal"},
        {nvmlClocksThrottleReasonHwPowerBrakeSlowdown, "hw_power_brake"},
        {nvmlClocksEventReasonDisplayClockSetting, "display_clock"},
        {nvmlClocksEventReasonBoardLimit, "board_limit"},
        {nvmlClocksEventReasonReliability, "reliability_voltage"},
    };
    std::vector<std::string> out;
    for (auto& [bit, name] : names)
        if (r & bit) out.push_back(name);
    return out;
}

constexpr unsigned long long THROTTLE_MASK = nvmlClocksEventReasonSwThermalSlowdown |
                                             nvmlClocksThrottleReasonHwThermalSlowdown |
                                             nvmlClocksThrottleReasonHwSlowdown |
                                             nvmlClocksThrottleReasonHwPowerBrakeSlowdown;

struct Sample {
    int round = 0;
    std::string prev;  // op sampled just before this one in the round ("" if first)
    double event_batch_ms = NAN, host_sync_ms = NAN;
    std::map<std::string, double> phase_ms;
    Snap snap;
    bool flagged = false;
    std::string flag_reason;
};

struct Stats {
    int n = 0;
    double median = NAN, p10 = NAN, p90 = NAN, min = NAN, max = NAN;
};

inline Stats stats_of(std::vector<double> v) {
    Stats s;
    v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return std::isnan(x); }), v.end());
    s.n = (int)v.size();
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    auto q = [&](double p) { return v[(size_t)std::lround(p * (v.size() - 1))]; };
    s.median = q(0.5);
    s.p10 = q(0.1);
    s.p90 = q(0.9);
    s.min = v.front();
    s.max = v.back();
    return s;
}

// Flags samples (see step 6). Returns the number flagged.
inline int flag_samples(std::vector<Sample>& samples) {
    std::vector<double> clocks;
    for (auto& m : samples) clocks.push_back(m.snap.sm);
    double med = stats_of(clocks).median;
    int n = 0;
    for (auto& m : samples) {
        m.flagged = false;
        m.flag_reason.clear();
        if (m.snap.reasons & THROTTLE_MASK) {
            m.flagged = true;
            m.flag_reason = "thermal or hardware slowdown";
        } else if (!std::isnan(med) && m.snap.sm < (1 - THROTTLE_DROP) * med) {
            m.flagged = true;
            m.flag_reason = "SM clock below the op's median";
        }
        n += m.flagged;
    }
    return n;
}

// An op is invalid if more than half its samples are flagged.
inline bool mostly_flagged(int flagged, size_t total) { return flagged * 2 > (int)total; }

// JSON helpers
inline std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\', o += c;
        else if (c == '\n') o += "\\n";
        else if ((unsigned char)c < 0x20) o += ' ';
        else o += c;
    }
    return o + "\"";
}
inline std::string jnum(double x) {
    if (std::isnan(x) || std::isinf(x)) return "null";
    char b[64];
    snprintf(b, sizeof b, "%.9g", x);
    return b;
}
inline std::string jlist(const std::vector<std::string>& v) {
    std::string o = "[";
    for (size_t i = 0; i < v.size(); ++i) o += (i ? ", " : "") + jstr(v[i]);
    return o + "]";
}
inline std::string jstats(const Stats& s) {
    return "{\"n\": " + std::to_string(s.n) + ", \"median\": " + jnum(s.median) + ", \"p10\": " + jnum(s.p10) +
           ", \"p90\": " + jnum(s.p90) + ", \"min\": " + jnum(s.min) + ", \"max\": " + jnum(s.max) + "}";
}

inline std::string read_first_line(const std::string& path) {
    std::ifstream f(path);
    std::string s;
    std::getline(f, s);
    return s;
}

class Runner {
  public:
    Runner(std::string benchmark, Limits limits) : bench_(std::move(benchmark)), lim_(limits) {
        if (lim_.tflops <= 0 || lim_.dram_gb_per_s <= 0) {
            warnings_.push_back("measured limits missing (results/cuda/peak.json not found or incomplete): "
                                "the plausibility guard is off");
            fprintf(stderr, "harness warning: %s\n", warnings_.back().c_str());
        }
        if (nvmlInit_v2() != NVML_SUCCESS) throw std::runtime_error("NVML init failed");
        if (nvmlDeviceGetHandleByIndex_v2(0, &dev_) != NVML_SUCCESS) throw std::runtime_error("NVML device 0");
        HCHECK(cudaSetDevice(0));
        HCHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        capture_env();
    }
    ~Runner() {
        if (stream_) cudaStreamDestroy(stream_);
        nvmlShutdown();
    }

    void add(Op op) {
        State s;
        s.op = std::move(op);
        ops_.push_back(std::move(s));
    }

    // Runs every group in the order its first op was added. Returns the number of ops that did not
    // produce a valid timing (failed, skipped, check_failed, invalid_throttled).
    int run_all() {
        std::vector<std::string> groups;
        for (auto& s : ops_)
            if (std::find(groups.begin(), groups.end(), s.op.group) == groups.end()) groups.push_back(s.op.group);
        for (auto& g : groups) {
            if (dead_) {
                for (auto& s : ops_)
                    if (s.op.group == g) {
                        s.status = "skipped";
                        s.error = "CUDA context lost in an earlier group (" + dead_group_ +
                                  "); this process cannot recover it: rerun in a new process";
                    }
                continue;
            }
            run_group(g);
        }
        int bad = 0;
        for (auto& s : ops_) bad += s.status != "ok";
        return bad;
    }

    void write_json(const std::string& path) const {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream f(path);
        f << "{\n \"schema\": \"airgap-bench/1\",\n \"benchmark\": " << jstr(bench_) << ",\n \"created\": " << jstr(created_)
          << ",\n \"warnings\": " << jlist(warnings_)
          << ",\n \"env\": " << env_json_ << ",\n \"limits\": {\"tflops\": " << jnum(lim_.tflops)
          << ", \"dram_gb_per_s\": " << jnum(lim_.dram_gb_per_s) << ", \"source\": \"results/cuda/peak.json\"},\n"
          << " \"protocol\": {\"samples\": " << SAMPLES << ", \"target_sample_ms\": " << jnum(TARGET_SAMPLE_MS)
          << ", \"warmup_s\": [" << jnum(WARMUP_MIN_S) << ", " << jnum(WARMUP_MAX_S) << "], \"clock_tol\": " << jnum(CLOCK_TOL)
          << ", \"throttle_drop\": " << jnum(THROTTLE_DROP) << ", \"rate_slack\": " << jnum(RATE_SLACK)
          << ", \"interleave\": \"seeded shuffle per round within a group\", \"shuffle_seed\": " << SHUFFLE_SEED
          << ", \"throttle_rule\": "
             "\"flag a sample on a thermal or hardware-slowdown reason, or an SM clock more than throttle_drop below the "
             "op's median; power cap recorded, not flagged; op invalid if more than half its samples are flagged\"},\n"
          << " \"groups\": [";
        for (size_t i = 0; i < groups_json_.size(); ++i) f << (i ? ", " : "") << groups_json_[i];
        f << "],\n \"ops\": [\n";
        for (size_t i = 0; i < ops_.size(); ++i) f << op_json(ops_[i]) << (i + 1 < ops_.size() ? ",\n" : "\n");
        f << " ]\n}\n";
    }

    struct State {
        Op op;
        std::string status = "pending", stage, error;
        CheckResult check;
        bool checked = false;
        double mem_mib = NAN;
        long iters = 0, host_iters = 0, phase_iters = 0;
        std::vector<Sample> samples;
        std::map<std::string, Stats> clocks;
        double tflops = NAN, dram_gb_per_s = NAN;
        std::vector<std::string> plausibility;
    };
    const std::vector<State>& results() const { return ops_; }

  private:
    std::string bench_, created_, env_json_;
    Limits lim_;
    nvmlDevice_t dev_{};
    cudaStream_t stream_{};
    std::vector<State> ops_;
    std::vector<std::string> groups_json_;
    bool dead_ = false;
    std::vector<std::string> warnings_;
    std::string dead_group_;

    Snap snap() const {
        Snap s;
        nvmlDeviceGetClockInfo(dev_, NVML_CLOCK_SM, &s.sm);
        nvmlDeviceGetClockInfo(dev_, NVML_CLOCK_MEM, &s.mem);
        nvmlTemperature_t t{};
        t.version = nvmlTemperature_v1;
        t.sensorType = NVML_TEMPERATURE_GPU;
        if (nvmlDeviceGetTemperatureV(dev_, &t) == NVML_SUCCESS) s.temp = (unsigned)t.temperature;
        nvmlDeviceGetCurrentClocksEventReasons(dev_, &s.reasons);
        return s;
    }

    static double now_ms() {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // true if the CUDA context is still usable after an error
    static bool context_ok() {
        cudaGetLastError();
        return cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == cudaSuccess;
    }

    void capture_env() {
        cudaDeviceProp p;
        HCHECK(cudaGetDeviceProperties(&p, 0));
        int rt = 0, drv = 0;
        cudaRuntimeGetVersion(&rt);
        cudaDriverGetVersion(&drv);
        char dver[96] = {0};
        nvmlSystemGetDriverVersion(dver, sizeof dver);
        std::string ac = "unknown";
        try {
            for (auto& e : std::filesystem::directory_iterator("/sys/class/power_supply"))
                if (read_first_line(e.path().string() + "/type") == "Mains")
                    ac = read_first_line(e.path().string() + "/online") == "1" ? "on" : "off";
        } catch (...) {
        }
        std::string profile = read_first_line("/sys/firmware/acpi/platform_profile");
        std::vector<std::string> procs;
        for (int kind = 0; kind < 2; ++kind) {
            unsigned n = 64;
            nvmlProcessInfo_t info[64];
            nvmlReturn_t r = kind == 0 ? nvmlDeviceGetComputeRunningProcesses(dev_, &n, info)
                                       : nvmlDeviceGetGraphicsRunningProcesses(dev_, &n, info);
            if (r != NVML_SUCCESS) continue;
            for (unsigned i = 0; i < n; ++i) {
                if (info[i].pid == (unsigned)getpid()) continue;
                char name[256] = {0};
                nvmlSystemGetProcessName(info[i].pid, name, sizeof name);
                std::string nm = std::filesystem::path(name).filename().string();
                if (std::find(procs.begin(), procs.end(), nm) == procs.end()) procs.push_back(nm);
            }
        }
        char tbuf[64];
        std::time_t t = std::time(nullptr);
        std::strftime(tbuf, sizeof tbuf, "%Y-%m-%dT%H:%M:%S%z", std::localtime(&t));
        created_ = tbuf;
        unsigned max_sm = 0;
        nvmlDeviceGetMaxClockInfo(dev_, NVML_CLOCK_SM, &max_sm);
        std::ostringstream o;
        o << "{\"gpu\": " << jstr(p.name) << ", \"cc\": \"" << p.major << "." << p.minor << "\", \"sms\": "
          << p.multiProcessorCount << ", \"memory_mib\": " << (p.totalGlobalMem >> 20) << ", \"max_sm_mhz\": " << max_sm
          << ", \"driver\": " << jstr(dver) << ", \"cuda_runtime\": " << rt << ", \"cuda_driver_api\": " << drv
          << ", \"nvcc\": \"" << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__
          << "\", \"host_compiler\": " << jstr(__VERSION__) << ", \"ac\": " << jstr(ac)
          << ", \"platform_profile\": " << jstr(profile) << ", \"other_gpu_processes\": " << jlist(procs) << "}";
        env_json_ = o.str();
    }

    // Event time of `iters` back-to-back runs, per iteration (ms).
    double event_batch(const Launch& f, long iters) {
        cudaEvent_t a, b;
        HCHECK(cudaEventCreate(&a));
        HCHECK(cudaEventCreate(&b));
        HCHECK(cudaEventRecord(a, stream_));
        for (long i = 0; i < iters; ++i) f(stream_);
        HCHECK(cudaEventRecord(b, stream_));
        HCHECK(cudaEventSynchronize(b));
        HCHECK(cudaGetLastError());
        float ms = 0;
        HCHECK(cudaEventElapsedTime(&ms, a, b));
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        return ms / iters;
    }

    // Host time of launch + synchronise, one iteration at a time, per iteration (ms).
    double host_sync(const Launch& f, long iters) {
        double t0 = now_ms();
        for (long i = 0; i < iters; ++i) {
            f(stream_);
            HCHECK(cudaStreamSynchronize(stream_));
        }
        double t1 = now_ms();
        HCHECK(cudaGetLastError());
        return (t1 - t0) / iters;
    }

    std::map<std::string, double> phase_times(const Op& op, long iters) {
        size_t n = op.phases.size();
        std::vector<cudaEvent_t> ev(n + 1);
        for (auto& e : ev) HCHECK(cudaEventCreate(&e));
        std::vector<double> acc(n, 0.0);
        for (long it = 0; it < iters; ++it) {
            HCHECK(cudaEventRecord(ev[0], stream_));
            for (size_t k = 0; k < n; ++k) {
                op.phases[k].second(stream_);
                HCHECK(cudaEventRecord(ev[k + 1], stream_));
            }
            HCHECK(cudaEventSynchronize(ev[n]));
            for (size_t k = 0; k < n; ++k) {
                float ms = 0;
                HCHECK(cudaEventElapsedTime(&ms, ev[k], ev[k + 1]));
                acc[k] += ms;
            }
        }
        HCHECK(cudaGetLastError());
        for (auto& e : ev) cudaEventDestroy(e);
        std::map<std::string, double> out;
        for (size_t k = 0; k < n; ++k) out[op.phases[k].first] = acc[k] / iters;
        return out;
    }

    void fail(State& s, const std::string& stage, const std::exception& e) {
        s.status = "failed";
        s.stage = stage;
        s.error = e.what();
    }

    void run_group(const std::string& g) {
        std::vector<State*> grp;
        for (auto& s : ops_)
            if (s.op.group == g) grp.push_back(&s);
        bool lost = false;
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        double used_before = (total_b - free_b) / 1048576.0;

        // 1. setup
        for (auto* s : grp) {
            if (lost) {
                s->status = "skipped";
                s->error = "context lost earlier in the group";
                continue;
            }
            try {
                size_t f0 = 0, f1 = 0, t = 0;
                cudaMemGetInfo(&f0, &t);
                if (s->op.setup) s->op.setup();
                HCHECK(cudaDeviceSynchronize());
                cudaMemGetInfo(&f1, &t);
                s->mem_mib = ((double)f0 - (double)f1) / 1048576.0;
                s->status = "ok";
            } catch (const std::exception& e) {
                fail(*s, "setup", e);
                if (!context_ok()) lost = true;
            }
        }
        cudaMemGetInfo(&free_b, &total_b);
        double used_after = (total_b - free_b) / 1048576.0;

        // 2. check
        for (auto* s : grp) {
            if (s->status != "ok" || !s->op.check || lost) continue;
            try {
                s->check = s->op.check();
                s->checked = true;
                if (!s->check.ok) s->status = "check_failed";
            } catch (const std::exception& e) {
                fail(*s, "check", e);
                if (!context_ok()) lost = true;
            }
        }
        if (lost)
            for (auto* s : grp)
                if (s->status == "ok") {
                    s->status = "skipped";
                    s->error = "context lost during setup or check";
                }
        std::vector<State*> active;
        for (auto* s : grp)
            if (s->status == "ok" && !lost) active.push_back(s);

        // An error in one op: mark it failed; if the context is lost, skip the rest of the group.
        auto drop = [&](State* s, const std::string& stage, const std::exception& e) {
            fail(*s, stage, e);
            if (!context_ok()) {
                lost = true;
                for (auto* o : grp)
                    if (o != s && (o->status == "ok" || o->status == "pending")) {
                        o->status = "skipped";
                        o->error = "context lost after " + s->op.name + " failed in " + stage;
                    }
            }
        };
        auto refresh_active = [&] {
            std::vector<State*> a;
            for (auto* s : active)
                if (s->status == "ok") a.push_back(s);
            active.swap(a);
        };

        // 3. warm-up until the SM clock is steady
        double warm_s = 0;
        std::vector<unsigned> warm_clocks;
        double t_warm = now_ms();
        while (!active.empty()) {
            for (auto* s : active) {
                try {
                    double t = event_batch(s->op.run, 1);
                    long k = std::max(1L, (long)(50.0 / std::max(t, 1e-4)));
                    event_batch(s->op.run, std::min(k, 100000L));
                } catch (const std::exception& e) {
                    drop(s, "warmup", e);
                    if (lost) break;
                }
            }
            refresh_active();
            if (active.empty()) break;
            warm_clocks.push_back(snap().sm);
            warm_s = (now_ms() - t_warm) / 1000;
            size_t m = warm_clocks.size();
            bool steady = m >= 3 && std::abs((double)warm_clocks[m - 1] - warm_clocks[m - 3]) <= CLOCK_TOL * warm_clocks[m - 1] &&
                          std::abs((double)warm_clocks[m - 2] - warm_clocks[m - 3]) <= CLOCK_TOL * warm_clocks[m - 1];
            if ((warm_s >= WARMUP_MIN_S && steady) || warm_s >= WARMUP_MAX_S) break;
        }

        // 4. calibrate iteration counts, then freeze
        for (auto* s : active) {
            try {
                double t = event_batch(s->op.run, 1);
                long probe = std::max(1L, (long)(2.0 / std::max(t, 1e-4)));
                t = event_batch(s->op.run, probe);
                s->iters = std::clamp((long)std::ceil(TARGET_SAMPLE_MS / std::max(t, 1e-5)), 1L, 2000000L);
                double h = host_sync(s->op.run, std::min(probe, 200L));
                s->host_iters = std::clamp((long)std::ceil(TARGET_SAMPLE_MS / 2 / std::max(h, 1e-4)), 1L, 200000L);
                s->phase_iters = s->op.phases.empty() ? 0 : std::min<long>(s->iters, PHASE_ITERS_MAX);
            } catch (const std::exception& e) {
                drop(s, "calibrate", e);
                if (lost) break;
            }
        }
        refresh_active();

        // 5. interleaved samples: a seeded shuffle per round
        std::mt19937 rng(SHUFFLE_SEED);
        for (int r = 0; r < SAMPLES && !active.empty(); ++r) {
            std::vector<State*> order = active;
            std::shuffle(order.begin(), order.end(), rng);
            std::string prev;
            for (auto* s : order) {
                if (s->status != "ok") continue;
                try {
                    Sample smp;
                    smp.round = r;
                    smp.prev = prev;
                    prev = s->op.name;
                    smp.event_batch_ms = event_batch(s->op.run, s->iters);
                    smp.host_sync_ms = host_sync(s->op.run, s->host_iters);
                    if (s->phase_iters) smp.phase_ms = phase_times(s->op, s->phase_iters);
                    smp.snap = snap();
                    s->samples.push_back(smp);
                } catch (const std::exception& e) {
                    drop(s, "timing", e);
                    if (lost) break;
                }
            }
            refresh_active();
        }

        // 6. flags, 7. stats and rates
        refresh_active();
        std::vector<double> clocks;
        for (auto* s : active)
            for (auto& smp : s->samples) clocks.push_back(smp.snap.sm);
        double med_clock = stats_of(clocks).median;
        for (auto* s : active) {
            int flagged = flag_samples(s->samples);
            std::vector<double> ev, hs;
            std::map<std::string, std::vector<double>> ph;
            for (auto& smp : s->samples) {
                if (smp.flagged) continue;
                ev.push_back(smp.event_batch_ms);
                hs.push_back(smp.host_sync_ms);
                for (auto& [k, v] : smp.phase_ms) ph[k].push_back(v);
            }
            s->clocks["event_batch_ms"] = stats_of(ev);
            s->clocks["host_sync_ms"] = stats_of(hs);
            for (auto& [k, v] : ph) s->clocks["phase_" + k + "_ms"] = stats_of(v);
            if (mostly_flagged(flagged, s->samples.size())) s->status = "invalid_throttled";
            double t = s->clocks["event_batch_ms"].median * 1e-3;
            if (s->op.flops > 0 && t > 0) {
                s->tflops = s->op.flops / t / 1e12;
                if (lim_.tflops > 0 && s->tflops > RATE_SLACK * lim_.tflops)
                    s->plausibility.push_back("FLOP rate above the measured FMA peak");
            }
            if (s->op.dram_bytes > 0 && t > 0) {
                s->dram_gb_per_s = s->op.dram_bytes / t / 1e9;
                if (lim_.dram_gb_per_s > 0 && s->dram_gb_per_s > RATE_SLACK * lim_.dram_gb_per_s)
                    s->plausibility.push_back("DRAM rate above the measured bandwidth");
            }
        }

        // 8. teardown
        for (auto* s : grp) {
            if (!s->op.teardown || lost) continue;
            try {
                s->op.teardown();
            } catch (const std::exception& e) {
                if (s->status == "ok") fail(*s, "teardown", e);
            }
        }
        if (lost) {
            dead_ = true;
            dead_group_ = g;
        } else {
            cudaDeviceSynchronize();
        }

        std::ostringstream o;
        o << "{\"group\": " << jstr(g) << ", \"device_used_mib_before\": " << jnum(used_before)
          << ", \"device_used_mib_after_setup\": " << jnum(used_after) << ", \"warmup_s\": " << jnum(warm_s)
          << ", \"warmup_sm_mhz\": [";
        for (size_t i = 0; i < warm_clocks.size(); ++i) o << (i ? ", " : "") << warm_clocks[i];
        o << "], \"median_sm_mhz\": " << jnum(med_clock) << ", \"context_lost\": " << (lost ? "true" : "false") << "}";
        groups_json_.push_back(o.str());
    }

    std::string op_json(const State& s) const {
        std::ostringstream o;
        o << "  {\"name\": " << jstr(s.op.name) << ", \"group\": " << jstr(s.op.group) << ", \"params\": {";
        for (size_t i = 0; i < s.op.params.size(); ++i)
            o << (i ? ", " : "") << jstr(s.op.params[i].first) << ": " << jstr(s.op.params[i].second);
        o << "},\n   \"status\": " << jstr(s.status);
        if (!s.stage.empty()) o << ", \"stage\": " << jstr(s.stage);
        if (!s.error.empty()) o << ", \"error\": " << jstr(s.error);
        if (s.checked)
            o << ", \"check\": {\"ok\": " << (s.check.ok ? "true" : "false") << ", \"max_abs_err\": "
              << jnum(s.check.max_abs_err) << ", \"max_rel_err\": " << jnum(s.check.max_rel_err)
              << ", \"note\": " << jstr(s.check.note) << "}";
        o << ", \"mem_mib\": " << jnum(s.mem_mib) << ", \"work\": {\"flops\": " << jnum(s.op.flops)
          << ", \"dram_bytes\": " << jnum(s.op.dram_bytes) << "}, \"iters\": {\"event_batch\": " << s.iters
          << ", \"host_sync\": " << s.host_iters << ", \"phases\": " << s.phase_iters << "},\n   \"clocks\": {";
        size_t i = 0;
        for (auto& [k, v] : s.clocks) o << (i++ ? ", " : "") << jstr(k) << ": " << jstats(v);
        o << "},\n   \"rates\": {\"tflops\": " << jnum(s.tflops) << ", \"dram_gb_per_s\": " << jnum(s.dram_gb_per_s)
          << "}, \"plausibility\": " << jlist(s.plausibility) << ",\n   \"samples\": [";
        for (size_t k = 0; k < s.samples.size(); ++k) {
            const Sample& m = s.samples[k];
            o << (k ? ",\n    " : "\n    ") << "{\"round\": " << m.round << ", \"prev\": " << jstr(m.prev)
              << ", \"event_batch_ms\": " << jnum(m.event_batch_ms)
              << ", \"host_sync_ms\": " << jnum(m.host_sync_ms);
            if (!m.phase_ms.empty()) {
                o << ", \"phase_ms\": {";
                size_t j = 0;
                for (auto& [pk, pv] : m.phase_ms) o << (j++ ? ", " : "") << jstr(pk) << ": " << jnum(pv);
                o << "}";
            }
            o << ", \"sm_mhz\": " << m.snap.sm << ", \"mem_mhz\": " << m.snap.mem << ", \"temp_c\": " << m.snap.temp
              << ", \"clock_events\": " << jlist(reason_names(m.snap.reasons)) << ", \"flagged\": "
              << (m.flagged ? "true" : "false");
            if (m.flagged) o << ", \"flag_reason\": " << jstr(m.flag_reason);
            o << "}";
        }
        o << "]}";
        return o.str();
    }
};

}  // namespace harness
