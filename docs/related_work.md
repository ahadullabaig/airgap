# Related work for the Part B paper

Literature search, 2026-11-25. Three research agents covered:
1. attention kernels and GPU measurement;
2. temporal redundancy and gated computation in video models;
3. frame filtering and sampling, video VLMs, and aerial video.

Bibliographic fields come from fetched records only: the arXiv API, Crossref, OpenAlex, the CVF, NeurIPS, ICLR,
MLSys and ACL proceedings pages, publisher pages, or official documentation. DBLP, OpenReview and MDPI blocked
automated access, Semantic Scholar rate-limited, and Google Scholar, IEEE Xplore and the ACM DL were not
searched. Coverage is therefore "not found in these sources", not "does not exist".

Each entry ends with an evidence level:
- **[checked]**: I confirmed the cited statement in the paper's text, or by our own measurement.
- **[agent: full text]**: an agent read the text; I have not checked it.
- **[agent: abstract]**: abstract or metadata only.

An author must still open every cited work before submission.

## A. Attention kernels and GPU measurement

- **FlashAttention** (`dao2022flashattention`): Dao, Fu, Ermon, Rudra, Ré, NeurIPS 35 (2022), pp. 16344–16359.
  - Tiling plus an online softmax avoids storing N x N.
  - Fig. 2 (A100, N = 1024, d = 64): larger blocks cut HBM traffic, but beyond a point runtime is "bottlenecked by other factors (e.g., arithmetic operations)".
  - Relevance: the algorithm V3 follows, and prior evidence that removing traffic stops paying once compute dominates. Our 13% recovery is a measured instance of this; it is not new as a phenomenon. [agent: full text]
- **FlashAttention-2** (`dao2024flashattention2`): Dao, ICLR 2024, pp. 35549–35562.
  - Reduces non-matmul work, because tensor-core matmul throughput is up to 16x the non-matmul FP32 throughput on A100.
  - Relevance: that asymmetry does not exist on CUDA cores in FP32, part of why V3 is compute-limited. [agent: full text]
- **Online normaliser** (`milakov2018online`): Milakov, Gimelshein, arXiv:1805.02867 (2018).
  - The one-pass running max and sum that V2's fallback softmax and V3 use. [agent: full text]
- **Memory-efficient attention** (`rabe2021selfattention`): Rabe, Staats, arXiv:2112.05682.
  - O(sqrt n) memory in practice. Do not claim PyTorch's backend descends from it. [agent: abstract]
- **PyTorch's FP32 "memory-efficient" SDPA is a tensor-core kernel on this GPU** (`pytorchsdpa`, `xformers2022`):
  - It launches `fmha_cutlassF_f32_aligned_64x64_rf_sm80` (`results/attention/b1_kernels.json`).
  - Nsight Compute counts 4.7 M tensor-pipe instructions, tensor pipe 38% active, FMA pipe 7% (`results/attention/b1_efficient_ncu.txt`). PyTorch's source builds it with CUTLASS `OpMultiplyAddFastF32`: FP32 emulated with TF32 parts.
  - The math backend uses cuBLAS SGEMM on CUDA cores.
  - The paper must call V3 versus B1's memory-efficient backend a CUDA-core kernel against a tensor-core FP32-emulating kernel. [checked: our own measurement; the source reading is agent: full text]
- **Faster Neighborhood Attention** (`hassani2024faster`): Hassani, Hwu, Shi, NeurIPS 37 (2024), pp. 64717–64734.
  - A naive, GEMM-based and fused ladder in full precision (FP32/TF32, A100).
  - Sec. 4, Tab. 3: "fused kernels only have a very minimal edge over unfused".
  - Relevance: the closest peer-reviewed ladder with a full-precision arm. Neighbourhood attention, not dense; data-centre GPU; no clock control. [checked]
- **ELSA** (`hsu2026elsa`): Hsu et al., CVPR 2026 Findings, pp. 2988–2997.
  - An exact FP32 attention kernel without tensor cores, 1.3–3.5x faster than memory-efficient SDPA on A100 and 1.5–1.6x on Jetson TX2.
  - Relevance: "an FP32 CUDA-core kernel beating PyTorch's FP32 SDPA" is not new. [checked: abstract and Sec. 5 lines]
- **Data Movement Is All You Need** (`ivanov2021data`): Ivanov et al., MLSys 3 (2021), pp. 711–732.
  - Transformer training is memory-bound; fusion cuts data movement.
  - Relevance: the usual expectation, against which our "removing traffic buys little" stands out. [agent: abstract]
- **EfficientViT** (`liu2023efficientvit`, CVPR 2023): ViT speed is often bounded by memory-inefficient operations in multi-head self-attention. [agent: abstract]
- **FlashAttention-T** (PPoPP 2026): softmax on CUDA cores leaves tensor cores idle in fused attention. [agent: abstract]
- **Dispatch-Aware Ragged Attention** (`abdellatif2026dispatch`): arXiv:2604.15408v2 (preprint, 2026).
  - At ≤197 tokens, host dispatch of about 50 µs exceeds GPU compute (FP16, RTX 4000 Ada, no clock control).
  - The author list differs between v1 and v2: cite v2.
  - Relevance: launch cost depends on shape. Present our <1% as specific to N = 1024 and 12 heads. [agent: full text]
- **Pruning One More Token is Enough** (`eliopoulos2025pruning`): WACV 2025, pp. 7153–7162.
  - ViT latency is non-linear in token count because of partial waves.
  - Locks CPU and GPU clocks on Jetson (Sec. 4.1); the A100's clocks "could not be locked". [agent: full text]
- **Clock and power methodology**:
  - Sinha et al., "Not All GPUs Are Created Equal" (`sinha2022notall`, SC22): power management causes about 8% (up to 22%) variation between GPUs of the same model. [agent: abstract]
  - NVIDIA CUTLASS measurement guidelines (`nvidiacutlassmethodology`): fixed-frequency runs measure architectural efficiency; fixed-power runs are "the most realistic". This matches our pinned-plus-as-deployed design. [agent: full text]
  - Nsight Compute Profiling Guide (`nvidiancuguide`), Clock Control: by default the tool sets the GPU clocks itself, and hardware or driver throttling overrides any setting. Our profiles used `--clock-control none`, so their durations are at the clock the GPU chose. [agent: full text]

## B. Temporal redundancy and gated computation

**The equal-cost "run less often" baseline has been tested before, and the gate won every time on per-frame
tasks:**

- **Event Neural Networks** (`dutson2022evnets`): Dutson, Li, Gupta, ECCV 2022, pp. 276–293.
  - Supp. B, "Comparison Against Output Interpolation" (OpenPose, JHMDB): running every 16 frames and interpolating reaches 68.52% PCK at 6.764e9 ops per frame. The EvNet reaches 76.37% at 6.780e9. [checked: arXiv full text]
- **Skip-Convolutions** (`habibian2021skipconv`): CVPR 2021.
  - An "EfficientDet-interleaved" baseline propagates predictions from keyframes to the following frames without processing them (Sec. 4.1, Fig. 4). Skip-Conv cuts compute 3.3–3.5x at about the same AP, while the interleaved baseline drops accuracy sharply. [checked]
- **ResidualViT** (`soldan2025residualvit`): ICCV 2025, pp. 22305–22315.
  - The supplement's "Frame Rate Ablation" (Charades-STA): ResidualViT at 2 FPS "outperforms CLIP at FPS=1, with similar computational cost". Its Limitations section expects less benefit where sparse sampling suits the task. [checked: arXiv]
- **Reducto** (`li2020reducto`): SIGCOMM 2020, pp. 359–376.
  - Pixel-difference frame filtering against Chameleon, whose knobs include 30–1 fps (Table 7, car counting at a 90% target): 90.08% accuracy at 103.40 fps backend against 92.00% at 93.75 fps, "37% better backend processing speeds". [checked]

**No reduced-frame-rate baseline:**

- **Eventful Transformers** (`dutson2023eventful`): Dutson, Li, Gupta, ICCV 2023.
  - The direct ancestor of our token-delta design: top-r token gates, including inside q kᵀ and A v. It reports 2–4x fewer FLOPs for a small accuracy loss, but only up to 1.74x GPU speedup (Table 2).
  - Baselines: the base model, a simplified STGT, and token-wise-only acceleration. It never reduces the frame rate, skips frames or holds the previous output: the text mentions interpolation only for position embeddings. [checked]
- **DeltaCNN** (`parger2022deltacnn`): CVPR 2022.
  - Equal-cost comparison only against a smaller model.
  - Limitations: temporally sparse CNNs "only work well on fixed camera input". [checked]
- **MotionDeltaCNN** (`parger2023motiondeltacnn`): ICCV 2023; the published title differs from arXiv.
  - Moving cameras handled through homography alignment and spherical buffers. Our gates do no alignment. [agent: full text]
- **Run-Length Tokenization** (`choudhury2024rlt`): NeurIPS 37 (2024).
  - Drops repeated patches before the model. Limitations: "with constant camera motion, few tokens will be removed". Its 7.5/15/30 fps comparison is about training time, at unmatched cost. [checked: camera-motion line]
- **Déjà Vu** (`hwang2025dejavu`): PVLDB 18(10), 2025.
  - Learned inter-frame reuse inside video-language models; every method samples at a fixed 2 FPS, with no frame-rate baseline.
  - "CMC and Eventful Transformer propose kernels to reuse computations in the attention layer. In our experiments, these kernels did not yield speedups because the attention layers have relatively" small cost. [checked]
- Other reuse and merging work, none with a rate baseline found [agent]:
  - STC-Cacher (CVPR 2026);
  - Temporal Token Reuse on UAV video (IEEE JSTARS 2026): the only gated UAV evaluation found;
  - MaskVD (WACV 2025): finds Eventful gives no significant latency benefit;
  - VLA-Cache (NeurIPS 2025), CMC (ASPLOS 2024), vid-TLDR, DyCoke, STTS, STTM;
  - STGT (NeurIPS 2021 workshop).
- **Low-frame-rate input as a strong baseline for VLMs** (preprints, small samples; cite with care):
  - FrameMogging, arXiv:2605.03351: "low-FPS dense remains a serious fixed-evidence baseline".
  - EVS, arXiv:2510.14624: frame subsampling nearly ties token pruning on some benchmarks.
  - SimpleStream, arXiv:2604.02317. [agent: full text / abstract]

## C. Frame sampling, video VLMs and the cost of the encoder

- **Few frames often suffice for clip-level tasks**:
  - Apollo (`zohar2025apollo`, CVPR 2025): the centre frame alone answers much of most benchmarks (Sec. 2.1). [agent: full text]
  - Singularity (`lei2023singularity`, ACL 2023). [agent: abstract]
  - ATP (`buch2022atp`, CVPR 2022). [agent: abstract]
  - PickNet (ECCV 2018): 6–8 frames for captioning. [agent: abstract]
- **Counter-evidence: more frames help captioning on dynamic data**:
  - SwinBERT (`lin2022swinbert`, CVPR 2022), Table 4a: MSR-VTT CIDEr 36.6 → 55.3 from 2 to 64 frames. [agent: full text]
  - LLaVA-Video (arXiv:2410.02713). [agent: full text]
- **CIDEr can be blind to temporal input**: TemporalBench (`cai2024temporalbench`, arXiv:2410.10818). "GPT-4o obtains similar performance with one compared to 64 frames on both CIDEr and BLEU scores", and the authors recommend text-embedding similarity. Our null caption result needs this caveat, or an embedding-similarity metric. [checked]
- **Aggregate ties can hide item-level flips**: Stable Curves (arXiv:2608.07014). [agent: full text]
- **Typical frame counts**: Video-MME (CVPR 2025): 4–384 frames, 1 fps for Gemini. [agent: full text]
- **The encoder may not dominate end-to-end cost**:
  - "Seeing is Free, Speaking is Not" (arXiv:2607.09520): on Jetson Orin NX, output tokens dominate energy. [agent: abstract]
  - "Seeing at the Edge" (arXiv:2610.20136). [agent: abstract]
  - So state the encoder's share of our own pipeline's latency, which E1's timings contain.
- **SmolVLM** (`marafioti2025smolvlm`), arXiv:2504.05299: no published version found. [agent: full text]
- **SmolVLM2-500M on edge NPUs** (`yang2026ondevice`, arXiv:2610.40039): full attention takes 66.5–70.3% of prefill time on Hexagon. That contrasts with attention's roughly 18% FLOP share on our GPU. [agent: full text]
- **Frame-rate knobs and filtering in video analytics**:
  - Chameleon (`jiang2018chameleon`, SIGCOMM 2018): 5 fps instead of 30 cuts resources 6x without hurting slow-traffic counting, but hurts fast cars; the best frame rate changes over time. [agent: full text]
  - NoScope (`kang2017noscope`, PVLDB 2017). [agent: full text]
  - Glimpse (SenSys 2015), FilterForward (MLSys 2019), Focus (OSDI 2018), InFi (MobiCom 2022). [agent: abstract]

## D. Aerial video

- **ERA** (`mou2020era`): IEEE GRSM 8(4), 2020: 2,864 five-second clips at 24 fps, 25 classes. The single middle frame (DenseNet-201) reaches 62.3% overall accuracy against 64.3% for the best video model (TRN): on our data, one frame carries nearly all the event signal. [checked]
- **CapERA** (`bashmal2023capera`): Remote Sensing 15(8), 2023. MDPI blocked the full text, so its frame counts and CIDEr tables are unread. [agent: abstract]
- **Other UAV datasets** [agent: abstract]:
  - UrbanVideo-Bench (ACL 2025)
  - UAV-Human (CVPR 2021)
  - Okutama-Action (CVPRW 2017)
  - Drone-Action (Drones 2019)
  - AirVT-15k (arXiv 2026)
- **Onboard or edge UAV video and VLMs** (preprints) [agent: abstract]:
  - BLIP-2 with keyframe selection (arXiv:2601.08408)
  - LiteVLA-H (arXiv:2605.00884)
  - AVERY (arXiv:2511.18151)

## E. What the paper may and may not claim

**Not new; do not claim:**
- comparing gating against reduced-frame-rate or held output at equal cost (EvNets, Skip-Conv, ResidualViT, Reducto);
- token caching in a VLM's encoder (Déjà Vu, STC, VLA-Cache);
- gating on UAV video (TTR);
- the camera-motion caveat (DeltaCNN, RLT, MotionDeltaCNN);
- few frames sufficing for clip-level video-language tasks;
- an FP32 CUDA-core attention kernel beating PyTorch's FP32 SDPA (ELSA);
- fusion paying off less than its traffic savings once compute dominates (FlashAttention Fig. 2; Faster Neighborhood Attention);
- clock locking as methodology (CUTLASS guide, WACV 2025).

**Not found in the sources searched, so these can be stated as such:**
- **The gate losing the equal-cost comparison.** Every published equal-cost comparison found favours the gate, on dense per-frame tasks (pose, detection, grounding, counting). Ours favours frame-rate reduction, on clip-level captioning from a moving drone camera inside a VLM's encoder, at both the representation and the caption level.
- **Caption quality when each sampled frame is replaced by an older one** (dense at a low frame rate, output held).
- **The fused-kernel study on a small consumer GPU:** a naive → tiled → fused FP32 dense-attention ladder on a small consumer GPU, compared at equal clocks, with launch, compute and memory floors, showing each kernel settling at its own clock under a laptop's software power cap and how much of the removed N x N traffic fusion recovers. The closest are two non-peer-reviewed T4 projects (a course project and a personal repository) without clock control.

**Actions for the draft:**
- Describe B1's memory-efficient backend as tensor-core FP32 emulation (B1 math is CUDA-core SGEMM).
- Present launch cost as dependent on shape.
- Attribute the online softmax to Milakov and Gimelshein.
- Cite FlashAttention's Fig. 2 and Faster Neighborhood Attention beside the 13% result.
- State the CIDEr caveat (TemporalBench), or add an embedding-similarity caption metric.
- State the encoder's share of end-to-end latency (from E1's timings).
- Discuss Reducto and EvNets as opposite results, and give the reason: task granularity and camera motion.
