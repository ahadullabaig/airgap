// K0 CPU benchmark: reference serial vs OpenMP at several frame sizes and
// thread counts. Configurations are interleaved in rounds so slow drift
// (thermals, turbo) spreads across all of them instead of biasing one.
//
// Frames: by default each size has a ring of frames totalling at least 4x the
// L3 cache (RING_BYTES), and each call compares the next consecutive pair, so
// frames come from DRAM as a stream of new frames would (the GPU benchmark does
// the same with its L2). --hot re-reads one pair, which stays in cache for
// sizes up to 1080p (the behaviour before 2026-11-23).
//
// Usage: k0_bench [--hot] [output.json]   (JSON goes to stdout if no path is given)
#include "k0.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef K0_FLAGS
#define K0_FLAGS "unknown"
#endif

namespace {

struct Config {
    int h, w, c, block;
    bool omp;
    int threads;  // ignored for the reference
};

struct Frames {
    int h, w, c;
    std::vector<std::vector<uint8_t>> ring;  // consecutive pairs (k, k + 1)
    size_t next = 0;
};

constexpr double RING_BYTES = 64.0 * 1048576;  // 4 x the 16 MB L3 of the Ryzen 7 5800H

std::string read_first_line(const char *path) {
    std::ifstream f(path);
    std::string line;
    if (f && std::getline(f, line)) return line;
    return "";
}

std::string cpu_model() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("model name", 0) == 0) return line.substr(line.find(':') + 2);
    return "";
}

std::string ac_online() {
    for (const char *p : {"/sys/class/power_supply/AC/online", "/sys/class/power_supply/ACAD/online",
                          "/sys/class/power_supply/ADP0/online", "/sys/class/power_supply/ADP1/online"}) {
        std::string v = read_first_line(p);
        if (!v.empty()) return v;
    }
    return "";
}

std::string json_str(const std::string &s) {
    std::string out = "\"";
    for (char ch : s) {
        if (ch == '"' || ch == '\\') out += '\\';
        out += ch;
    }
    return out + "\"";
}

double pct(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    const double idx = q * (v.size() - 1);
    const size_t lo = static_cast<size_t>(idx), hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

double run_once(const Config &k, Frames &f, std::vector<float> &out) {
    const size_t i = f.next++ % (f.ring.size() - 1);
    const uint8_t *a = f.ring[i + 1].data(), *b = f.ring[i].data();
    const auto t0 = std::chrono::steady_clock::now();
    if (k.omp)
        k0_changemap_omp(a, b, f.h, f.w, f.c, k.block, out.data());
    else
        k0_changemap_serial(a, b, f.h, f.w, f.c, k.block, out.data());
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

}  // namespace

int main(int argc, char **argv) {
    bool hot = false;
    const char *out_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--hot") hot = true;
        else out_path = argv[i];
    }
    const int sizes[][2] = {{512, 512}, {720, 1280}, {1080, 1920}, {2160, 3840}};
    const int thread_counts[] = {1, 2, 4, 8, 16};
    const int c = 3, block = 16, rounds = 5, warmup = 3;
    const double batch_budget_us = 20000.0;  // per config per round
    const int batch_min = 10, batch_max = 200;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<Frames> frames;
    for (auto &s : sizes) {
        Frames f{s[0], s[1], c, {}, 0};
        const size_t n = static_cast<size_t>(f.h) * f.w * c;
        const size_t count = hot ? 2 : std::max<size_t>(3, static_cast<size_t>(std::ceil(RING_BYTES / n)) + 1);
        f.ring.assign(count, std::vector<uint8_t>(n));
        for (auto &fr : f.ring)
            for (auto &v : fr) v = static_cast<uint8_t>(dist(rng));
        frames.push_back(std::move(f));
    }

    std::vector<Config> configs;
    std::vector<size_t> frame_of;
    for (size_t i = 0; i < frames.size(); ++i) {
        configs.push_back({frames[i].h, frames[i].w, c, block, false, 1});
        frame_of.push_back(i);
        for (int t : thread_counts) {
            configs.push_back({frames[i].h, frames[i].w, c, block, true, t});
            frame_of.push_back(i);
        }
    }

    std::vector<std::vector<double>> samples(configs.size());
    std::vector<float> out(static_cast<size_t>((2160 + block - 1) / block) * ((3840 + block - 1) / block));

    for (size_t i = 0; i < configs.size(); ++i) {
        k0_set_num_threads(configs[i].threads);
        for (int r = 0; r < warmup; ++r) run_once(configs[i], frames[frame_of[i]], out);
    }
    for (int r = 0; r < rounds; ++r) {
        for (size_t i = 0; i < configs.size(); ++i) {
            k0_set_num_threads(configs[i].threads);
            run_once(configs[i], frames[frame_of[i]], out);  // first call after a thread-count change
            double spent = 0;
            for (int n = 0; n < batch_max && (n < batch_min || spent < batch_budget_us); ++n) {
                const double us = run_once(configs[i], frames[frame_of[i]], out);
                samples[i].push_back(us);
                spent += us;
            }
        }
    }

    std::ostringstream js;
    js << "{\n  \"benchmark\": \"k0_changemap_cpu\",\n";
    js << "  \"env\": {\n";
    js << "    \"cpu\": " << json_str(cpu_model()) << ",\n";
    js << "    \"logical_cpus\": " << std::thread::hardware_concurrency() << ",\n";
    js << "    \"compiler\": " << json_str(__VERSION__) << ",\n";
    js << "    \"flags\": " << json_str(K0_FLAGS) << ",\n";
    js << "    \"ac_online\": " << json_str(ac_online()) << ",\n";
    js << "    \"governor\": " << json_str(read_first_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")) << ",\n";
    js << "    \"platform_profile\": " << json_str(read_first_line("/sys/firmware/acpi/platform_profile")) << ",\n";
    const char *bind = std::getenv("OMP_PROC_BIND");
    js << "    \"OMP_PROC_BIND\": " << json_str(bind ? bind : "") << "\n  },\n";
    js << "  \"protocol\": {\"rounds\": " << rounds << ", \"warmup_calls\": " << warmup
       << ", \"batch_budget_us\": " << batch_budget_us << ", \"batch_min\": " << batch_min
       << ", \"batch_max\": " << batch_max << ", \"data\": \"uniform random bytes, seed 42\", \"frames\": "
       << json_str(hot ? "hot: one pair re-read (cache-resident up to 1080p)"
                       : "ring >= 64 MB per size (4x L3), consecutive pairs") << "},\n";
    js << "  \"results\": [\n";
    for (size_t i = 0; i < configs.size(); ++i) {
        const Config &k = configs[i];
        const double med = pct(samples[i], 0.5);
        const double bytes = 2.0 * k.h * k.w * k.c;
        js << "    {\"h\": " << k.h << ", \"w\": " << k.w << ", \"c\": " << k.c << ", \"block\": " << k.block
           << ", \"impl\": " << json_str(k.omp ? "omp" : "serial_ref") << ", \"threads\": " << k.threads
           << ", \"ring_frames\": " << frames[frame_of[i]].ring.size() << ", \"n\": " << samples[i].size()
           << ", \"median_us\": " << med
           << ", \"p10_us\": " << pct(samples[i], 0.1) << ", \"p90_us\": " << pct(samples[i], 0.9)
           << ", \"read_gb_per_s\": " << bytes / med / 1e3 << "}" << (i + 1 < configs.size() ? "," : "") << "\n";
    }
    js << "  ]\n}\n";

    if (out_path) {
        std::ofstream f(out_path);
        f << js.str();
        std::fprintf(stderr, "wrote %s\n", out_path);
    } else {
        std::fputs(js.str().c_str(), stdout);
    }
    return 0;
}
