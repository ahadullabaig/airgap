// The attention versions the generic test, benchmark and profiling drivers know about. A version with
// three phases (scores, softmax, output; N x N stored in its workspace) fills in the phase functions;
// a fused version leaves them null and has only `full`.
#ifndef ATTN_VERSIONS_H
#define ATTN_VERSIONS_H

#include <cstring>

#include "attn.h"

struct AttnVersion {
    const char* name;   // "v1"
    const char* label;  // "V1"
    size_t (*workspace_bytes)(int heads, int n);
    int (*full)(const float* q, const float* k, const float* v, float* o, float* workspace, int heads, int n,
                cudaStream_t stream);
    int (*scores)(const float* q, const float* k, float* s, int heads, int n, cudaStream_t stream);
    int (*softmax)(float* s, int heads, int n, cudaStream_t stream);
    int (*output)(const float* p, const float* v, float* o, int heads, int n, cudaStream_t stream);
};

static const AttnVersion ATTN_VERSIONS[] = {
    {"v1", "V1", attn_v1_workspace_bytes, attn_v1, attn_v1_scores, attn_v1_softmax, attn_v1_output},
    {"v2", "V2", attn_v2_workspace_bytes, attn_v2, attn_v2_scores, attn_v2_softmax, attn_v2_output},
    {"v3", "V3", attn_v3_workspace_bytes, attn_v3, nullptr, nullptr, nullptr},
};

static inline const AttnVersion* attn_version(const char* name) {
    for (const auto& v : ATTN_VERSIONS)
        if (std::strcmp(v.name, name) == 0) return &v;
    return nullptr;
}

#endif  // ATTN_VERSIONS_H
