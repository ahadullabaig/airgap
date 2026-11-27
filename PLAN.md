# PLAN: Onboard Drone Perception

Two outcomes from one codebase:

- **Part A (NLP course):** a working offline demo plus an evaluation of its outputs. MVP quality is enough for the demo; the evaluation is required.
- **Part B (Parallel Programming course):** a research paper.

**Done means:**
- Part A: the acceptance test (2.4) passes and the evaluation (2.5) is written up.
- Part B: a complete paper with the floor contributions (3.3), every number traceable to a result file. The ceiling is included only if the decision gate passed.

## 0. Confirmed and assumed

**Confirmed**
- Both instructors have approved two deliverables built on one codebase.
- The paper is a course paper in conference format. No venue submission is required.
- There is no deadline. Week numbers in this plan give the order of work, not dates.
- The NLP instructor expects an evaluation of Part A (section 2.5).
- Target device (checked 2026-11-21): RTX 3050 Laptop GPU, 4 GB VRAM, driver 620.70, Fedora 44. Toolchain (checked 2026-11-23, `results/cuda/toolchain.txt`): CUDA toolkit 13.5 from NVIDIA's repository, which accepts the system GCC 16.2 (its host-compiler limit is 16); PyTorch 2.15 built for CUDA 13.2 in a uv environment on Python 3.12. Kernels are standalone binaries, so the toolkit and PyTorch's CUDA versions need not match.
- Model shape (from the SmolVLM2-500M config): the vision encoder sees 512 px frames in 16 px patches, so N = 1024 tokens per frame, hidden size 768, 12 heads, head size d = 64. Built-in video sampling is 1 fps. Confirmed on first load (2026-11-21): the encoder outputs 1024 tokens per frame. Without image splitting, a 16:9 frame is stretched (not cropped or padded) to 512x512, so all 1024 patches are real image content. Other aspect ratios are unchecked. Each frame costs 64 tokens on the language side.

**Still assumed (correct these first)**

| Assumption | If it is wrong |
|---|---|
| Team of 3-4 | - |
| No drone, no flying. Recorded footage replayed from disk. | - |
| "Offline" means zero network calls at inference time | - |

## 1. How the two parts relate

```
Part A: demo                               Part B: paper
video -> sample frames -> VLM captions     attention kernels + benchmarks
      -> mission report                            ^
            |                                      |
            +--- token traces from real footage ---+
```

**Dependency rule:** neither part may block the other. The demo uses library code only and never waits for a custom kernel. The paper's kernels are benchmarked standalone. The only link is one-way: the demo exports traces (sequence lengths and per-frame token changes on real aerial footage) that make the paper's evaluation realistic.

---

## 2. Part A - NLP demo

### 2.1 What it does

A user points the tool at an aerial video. With the network off, it produces timestamped scene descriptions and a short mission report.

### 2.2 Scope

| MVP (must work) | Later (only if time) |
|---|---|
| CLI: `run_mission.py --video X --offline` | Simple web UI (Gradio or similar) with a clickable timeline |
| Uniform frame sampling, one caption per window | Change-based frame skipping |
| SmolVLM2-500M, default precision | 2.2B model quantised to fit 4 GB |
| Report as Markdown: timeline, notable events, uncertain items | Questions over the mission ("were there vehicles near the river?") answered from the captions |
| Three prepared demo videos | Any user-supplied video |
| Evaluation E1-E4 (section 2.5) | Ablations (model size, frames per window, image splitting); check that the uncertainty flag predicts errors; fine-tuning on CapERA train |

### 2.3 Build steps

1. Load the model from local files; caption one clip with the network disabled.
2. Windowing: split a video into fixed windows, sample a few frames per window, caption each window with a fixed prompt.
   - Decode frames ourselves with PyAV (`demo/sampling.py`) rather than using the processor's 1 fps video path. The same decoder then serves the trace export.
   - Decide image splitting on purpose. With splitting on, a 16:9 720p frame becomes 13 tiles, and a one-frame prompt grows from 76 to 871 tokens: better for small objects, but roughly 13x the encoder work.
3. Report generation: a fixed Markdown template holds the structure (timeline table, notable events, uncertain items), filled from the timestamped captions. The language model (the VLM's text side, or a small local text model loaded after the VLM is unloaded) writes only the summary paragraph. The 500M model's text side is small (hidden size 960).
   - "Uncertain items" needs a defined source, for example a caption's low mean token log-probability, or disagreement between neighbouring windows.
4. Wrap in the CLI; log latency per window and peak VRAM.
5. Trace export for Part B: run the vision encoder on its own at a higher frame rate than the captioning (5 fps up to the clip's native rate; 1 fps is not the regime gating targets). It needs only the encoder, not the finished demo, so it runs before the decision gate. Export per-token change magnitudes for every layer, plus full token tensors for a few frames only. Full tensors for every frame would be roughly 20 MB per frame (estimate).

### 2.4 Acceptance test

- Fresh terminal, one command, no network: `HF_HUB_OFFLINE=1`, and either Wi-Fi off or, for a repeatable check, run inside `unshare -rn` or `podman --network=none`. (Checked 2026-11-21: the GPU is visible inside `unshare -rn` and outbound connections fail.)
- A 2-3 minute aerial video produces a report in a time you are happy to show live (decide the budget after the first run).
- The report names the main events in each of the three prepared videos.
- A screen recording of a successful run exists as a fallback for demo day.

### 2.5 Evaluation (required)

The NLP instructor expects an evaluation but prescribes no form (asked, 2026-11-23), so the design below is ours. It is sized to finish, not to be exhaustive. All results are zero-shot: nothing is trained on CapERA.

**Data: CapERA test split** (checked 2026-11-21)
- 1,391 clips in 25 classes: 24 events plus NonEvent (155 clips). Class sizes range from 19 to 155. The class is in each clip's file name.
- 5 captions per clip, about 9 words on average. One is the manual annotation; the other four are generated from it. Treat this as one human reference with paraphrases, not five independent references.
- The repo holds captions only. The videos come from the ERA dataset, downloaded separately (DATA.md); every caption id matches a clip.

| ID | What | How | Output |
|---|---|---|---|
| E1 | Caption quality | A fixed, class-stratified subset of the test split; set its size after timing the first 50 clips. The prompt asks for one short sentence to match the reference style. | BLEU-4, METEOR, ROUGE-L and CIDEr from a standard COCO caption scorer; mean output length |
| E2 | Event recognition | Give the model the 25 class names and ask it to pick one per clip. Match the answer to a class name; count unparseable answers as wrong and report how often that happens. NonEvent is scored as an ordinary class. | Accuracy, macro-F1 (classes are uneven), confusion matrix |
| E3 | The actual report | Before looking at any output, write down the main events (type and rough time) in each of the three demo videos. Score each generated report against that list. If two people can score independently, report their agreement. | Events found; events claimed that are not in the video; timestamps in the right window |
| E4 | Error analysis | About 50 clips, spread across classes: E2 misclassifications plus the lowest-scoring E1 captions, tagged by failure type (small objects, counting, viewpoint, invented objects, wrong event) | Failure-type counts; five good and five bad examples for the slides |

Report latency and peak VRAM next to the quality numbers. The CapERA paper's benchmark models are trained on CapERA, so do not put our zero-shot numbers in the same table without saying so.

*The quantisation, frame-budget and token-budget experiments stay in the Later column of 2.2.*

### 2.6 Checklist

Items marked (team) need a person; the rest is code and GPU runs. Order matters: E3's event lists must exist
before any report is generated, and E2-E4 run on the frozen demo.

**Done**
- [x] Model captions a clip with the network off (`demo/caption_clip.py`); frames decoded with PyAV (`demo/sampling.py`)
- [x] E1 on the whole test split: 1,391 clips, CIDEr 0.210, BLEU-4 0.059, METEOR 0.168, median 0.66 s per clip
  (`results/e1/scores_test.json`; prompt P1 chosen on 100 training clips, `eval/e1_config.json`)

**Demo inputs (before building the report)**
- [ ] (team) Choose three 2-3 minute aerial videos that we may show in class. ERA clips last about 5.75 s, too short
  for this; record each video's source and terms in DATA.md
- [ ] (team) For each video, write down the main events (type, rough time) and commit the list before running the
  demo on it. This is E3's reference; two people writing separately is better
- [ ] (team) Name E3's scorers (two, if possible, for agreement)

**Demo build (2.3, 2.4)**
- [x] Windowing: window length, frames per window and prompt. Start from E1's frozen settings (P1, 4 frames, no
  image splitting, greedy, 48 new tokens). Decided 2026-11-27: E1's settings unchanged; the video is cut into
  round(duration / 6 s) equal windows, so no short tail is left
- [x] "Uncertain items": choose the source (low mean token log-probability, or disagreement between neighbouring
  windows) and its threshold. Decided 2026-11-27: mean token log-probability below -0.879, the 10th percentile on
  E1's 100 train selection clips (`demo/uncertainty.json`; the calibration reproduced E1's captions exactly)
- [x] Report: fixed Markdown template (timeline table, notable events, uncertain items); the language model writes
  only the summary paragraph. Notable events: captions containing a word from a fixed list built from ERA's 25
  event classes (`EVENTS` in `demo/run_mission.py`), runs of windows joined
- [x] `run_mission.py --video X --offline`: one command; logs latency per window and peak VRAM
  (`demo/run_mission.py`; `--offline` also makes Python-level internet connections fail; `unshare -rn` is the
  guarantee)
- [ ] First full run on one video; set the time budget for a live run. Development runs (2026-11-27, under
  `unshare -rn`), not demo videos: Sheboygan Falls (1:38, 720p) in 16.4 s, 0.82 s per window of which 0.23 s is
  decoding; 25 ERA train clips joined (2:23, 640x640) in 18.7 s, 0.62 s per window; 1.1 GB peak GPU memory; 17-19%
  of windows flagged uncertain. Decoding 1080p or 4K is unmeasured. The summary paragraph invents and reorders
  events on both, which E3 will count
- [ ] Runs on all three videos
- [ ] Acceptance test (2.4): fresh terminal, `unshare -rn`, `HF_HUB_OFFLINE=1`; each report names its video's main
  events
- [ ] Freeze the demo (tag the commit)
- [ ] (team) Screen recording of a successful run, as the demo-day fallback

**Evaluation (2.5), on the frozen demo**

E2 runs before the freeze (decided 2026-11-27): it uses E1's frozen settings, not the demo's windowing, so it does
not wait for the demo. Rerun it if the demo changes the model or those settings.
- [x] E2 script: the 25 class names in the prompt, answer matched to a class name, unparseable answers counted wrong
  and reported. Parsing rule and metrics committed before the run (`96fcd01`; prompt P3, one option per line,
  chosen on 200 train clips, `eval/e2_config.json`)
- [x] E2 run on the test split (8 min, 0.35 s per clip): accuracy 0.410 [0.384, 0.436], macro-F1 0.370 [0.353,
  0.386]; 54 of 1,391 answers match no name; 8 classes are never predicted (`results/e2/scores_test.json`)
- [ ] (team) E3: score the three reports against the event lists: events found, events claimed that are not in the
  video, timestamps in the right window; agreement if two scorers
- [x] E4: pick about 50 clips across classes (E2's misclassifications plus E1's lowest-CIDEr captions). 50 clips,
  two per class (`eval/e4.py select`, rule in its docstring); 40 of them are E2 errors. The team tags
  `results/e4/tags.csv`; `eval/e4.py show` prints each clip's video path, references and our outputs
- [ ] (team) E4: tag each by failure type (small objects, counting, viewpoint, invented objects, wrong event); counts;
  five good and five bad examples
- [ ] Latency and peak VRAM reported next to the quality numbers

**Write-up and demo day**
- [ ] Evaluation write-up, E1-E4. Results are zero-shot: never in the same table as CapERA's trained models without
  saying so. The references are one human caption plus four paraphrases
- [ ] (team) Slides. CapERA has no stated licence: confirm before showing its reference captions outside the course
- [ ] (team) Demo rehearsal

---

## 3. Part B - Parallel Programming paper

### 3.1 Research question

Frozen 2026-11-25 (the authors' choice, after the gate decision and the literature search):

> Is skipping unchanged tokens ever worth more than skipping whole frames, and where does attention's time go
> on a laptop GPU?

The original question, written for the gated kernel:

> Consecutive video frames barely differ, so in principle only the tokens that
> changed need new attention outputs. On a small, memory-limited GPU, how much of
> that theoretical saving does a fused CUDA kernel actually deliver, and which
> overheads consume the rest?

Paper title (the authors' choice): "Fewer Frames Beat Smarter Kernels: Attention, Clocks and Temporal Redundancy on
a Laptop GPU".

Hypotheses:

- **H1.** At the encoder's real sequence length (N = 1024, d = 64, 12 heads), dense attention on this GPU is limited by compute and memory traffic, not by fixed per-launch costs, so fusion helps mainly by removing the N x N traffic. Fixed costs only dominate once gating has made the work small, below a keep-rate that H3 predicts.
- **H2.** A gated kernel that does compaction and attention in one launch (G1, G2) realises more of the theoretical saving than the same steps as separate launches of our own kernels (B4), and than selecting tokens in PyTorch and calling library attention (B2).
- **H3.** A cost model (a fixed launch cost plus a work term that steps up with each wave of thread blocks, not a smooth M x N term) predicts the keep-rate at which gating starts to pay off.

N is the number of tokens; M is the number that changed since the last frame.

**Predicted times** (written 2026-11-23, before any attention kernel; `bench/predict_times.py`, full output in `results/cuda/predicted_times.json`). Inputs are this GPU's measured limits (`kernels/microbench`, `results/cuda/peak.json`): FP32 FMA 7.46 TFLOPS sustained over 10 s (about 1.9 GHz, at the software power cap; a short burst gives 7.5), DRAM 188 GB/s read and 183 GB/s copy, L2 reads at least 0.73 TB/s (512 KiB working set), empty-kernel launch 1.7 us back to back and 4.6 us with a sync. Shape: N = 1024, 12 heads, d = 64, FP32. Ranges come from assumed efficiencies (stated in the script), not from measurements.

| Kernel | Predicted time | What sets it |
|---|---|---|
| V1 (three kernels, N x N in DRAM) | 2.0-4.9 ms | Untiled matmuls at 10-30% of peak, assuming 16 x 16 blocks reuse operand rows from L1 (without that reuse, L2 traffic alone would take about 9 ms per matmul), plus about 0.5 ms of softmax traffic |
| V2 (tiled, N x N in DRAM) | 1.2-1.6 ms | Four N x N passes through DRAM (about 1 ms) plus tiled matmuls at 40-70% of peak |
| V3 (fused, online softmax) | 0.66-1.2 ms | Compute: 3.2 GFLOP at 35-65% of peak; DRAM traffic only 13 MB |
| B1 (library attention) | 0.5-1.8 ms | 35-85% of peak (the low end is the rate the whole PyTorch encoder reaches, 2.7 TFLOPS); the math backend also round-trips N x N |

Added 2026-11-25, after V2 and before V3 existed. V2 ran at about 1.5 GHz under the software power cap (`results/attention/v2.json`), not the 1.9 GHz of the FMA peak above. If V3 also sits at that clock, the same efficiencies predict **0.84-1.56 ms** for V3 (peak scaled to 5.92 TFLOPS). V3 moves much less DRAM traffic than V2, so it may run faster than that; both ranges stand.

Gated, at keep-rate k = M / N (G1 against our V3, B2 against B1):
- G1 recomputing changed rows only (design option 1): break-even near k = 0.8-0.9. At small M the time is set by wave quantisation, 0.05-0.22 ms for k below 0.1, which split-KV would reduce.
- G1 with the exact update (option 3): break-even near k = 0.42, since updating every unchanged row costs about 2k(1 - k) of dense on top of the changed rows.
- B2 (PyTorch selection plus library attention): break-even near k = 0.9 against B1, with 50-150 us of fixed cost (the sync to learn M, index operations, dispatch). Rows-only G1 and B2 are the like-for-like pair (both leave unchanged rows stale); their predicted ranges overlap at every keep-rate, so the model does not order them. Whether H2 holds depends on how close our kernels get to library efficiency, which B1's measured time will show.
- Launch costs are microseconds against dense times near a millisecond, as H1 expects.
- Attention is about 18% of an encoder layer's FLOPs at this shape, so gating attention alone bounds the encoder-level saving.

The decision gate compares measured keep-rates on real footage with these break-evens; the measured kernel times replace the predictions as they arrive.

Notes on the hypotheses:
- **H1 measured (X2, 2026-11-25;** `bench/x2_breakdown.py`, `results/attention/x2_breakdown.json`, `plots/x2_time.png`, `plots/x2_stalls.png`). The criteria were written after V1-V3 were timed, so they are not blind.
  - **First clause holds.** Empty kernels with each version's own launches cost 0.11-0.82% of its time. Each phase's launch is timed alone; timed back to back as one call they cost a little less.
  - **Second clause fails.** V3 saves 0.40 ms over V2, and 0.14-0.22 ms with V2 brought to V3's clock. V2's N x N traffic is worth 1.07 ms at full DRAM bandwidth.
  - **Where the time goes:**
    - V2's softmax sits on its DRAM floor.
    - V2's matmuls take 2.1-2.2x their floors, where the DRAM and compute floors nearly tie; their warps mostly wait on the issue side.
    - V3 takes 3.1x its compute floor; its warps wait on MIO results (shared memory, shuffles, exp) and the issue side.
    - Floors use the bytes each kernel actually moved, so V1's softmax floor includes its two avoidable extra reads.
    - V1 waits on global memory.
  - **Causes:**
    - V2's shared-memory store conflicts are bounded, not tested. They are 10% of the scores kernel's shared-memory wavefronts, and all MIO waiting is 7.5% of its warp cycles.
    - Doubling V2's output-kernel occupancy was no faster, but that shape also halves the register tile, so occupancy is confounded.
    - V3's occupancy is untested, because no swept shape raises it.
    - The clock is untested, because pinning it needs root.
  - **For reference:** copying Q, K and V to the GPU and O back costs more than V3 itself (1.42 + 0.47 ms).
  - **At a pinned clock (1,395 MHz, `nvidia-smi -lgc`, 2026-11-25;** `bench/clock_pinned.py`, `results/attention/clock1395/summary.json`). Every op held the pin, the memory clock was unchanged, and the definitions were written before the run.
    - V3 saves 0.14 ms over V2 (1.77 against 1.92 ms), 13% of the N x N floor. The second clause's failure stands, and the unpinned 22% gap was mostly clock.
    - Against predictions rescaled to 1,395 MHz: V2 lands inside (1.38-2.01 ms), V3 is 6% above (0.91-1.68 ms), V1 is far above.
    - Our kernels' cycle counts barely change with the clock (pinned against unpinned, 0.98).
- H1 replaces an earlier version that said fixed per-launch costs dominate at ViT lengths. That result (Dispatch-Aware Ragged Attention) was measured at 197 tokens or fewer on a much faster GPU (RTX 4000 Ada), with about 50 µs of host-side dispatch per call.
- Estimate, unmeasured: dense attention at N = 1024 with 12 heads is about 3.2 GFLOP, on the order of 1 ms on this GPU, against a few µs per raw kernel launch. Write predicted times for every kernel before the first benchmark; a large miss means a broken kernel or a wrong model.
- The memory wall (where V1 and V2 run out of memory) sits near N = 30,000 for one head, a separate regime from the gating tests. It belongs to the kernel ladder (X1), not to H1.
- B4 is what makes H2 fair. G1 and G2 are C++ binaries while B2 goes through PyTorch, so without B4 a win could come from skipping Python dispatch rather than from fusion.

### 3.2 Position against prior work

Literature search done 2026-11-25. The annotated list is `docs/related_work.md`; it records each statement's evidence level, and which statements were checked against the papers' text.

| Work | What it established | Where we stand |
|---|---|---|
| FlashAttention (NeurIPS 2022), FlashAttention-2 (ICLR 2024); online softmax (Milakov & Gimelshein 2018) | Tiling plus an online softmax avoids N x N storage. At N = 1024, d = 64 on A100, cutting traffic stops paying once arithmetic dominates (Fig. 2). | V3 is this algorithm in FP32 on CUDA cores. Our "fusion recovers 13% of the removed N x N traffic" is a measured instance on a small GPU, not a new phenomenon. |
| Faster Neighborhood Attention (NeurIPS 2024); ELSA (CVPR 2026 Findings) | Naive/GEMM/fused ladder in full precision: fused kernels have "a very minimal edge over unfused". An FP32 CUDA-core kernel beats PyTorch's FP32 SDPA, including on Jetson. | Neither is dense attention on a small consumer GPU at equal clocks with a launch/compute/memory breakdown. "Our kernel beats PyTorch FP32" is not new. |
| PyTorch SDPA source; our profile (`results/attention/b1_kernels.json`, `b1_efficient_ncu.txt`) | On compute capability 8.0+ the FP32 "memory-efficient" backend is a CUTLASS tensor-core kernel emulating FP32 with TF32 parts. | Describe B1 accordingly: memory-efficient is tensor-core FP32 emulation; math is CUDA-core SGEMM. |
| Dispatch-Aware Ragged Attention (2026 preprint); Pruning One More Token (WACV 2025) | Host dispatch (about 50 µs) hides savings at 197 tokens or fewer. Latency is non-linear in token count. Clocks were locked on Jetson. | Our launch cost under 1% is specific to N = 1024 and 12 heads. Clock locking is established methodology (also the CUTLASS measurement guide). |
| Eventful Transformers (ICCV 2023) | Token gates including inside attention: 2-4x fewer FLOPs, up to 1.74x GPU speedup. No reduced-frame-rate or held-output baseline (checked). | The direct ancestor of our token-delta design. Its missing baseline is the gap our equal-cost comparison fills for this design family. |
| Event Neural Networks (ECCV 2022), Skip-Convolutions (CVPR 2021), ResidualViT (ICCV 2025), Reducto (SIGCOMM 2020) | Equal-cost comparisons against running less often, interpolating or holding output, or frame-rate adaptation. The gate won in every case, on per-frame tasks. | Our comparison is not new; our result is the opposite. The reasons to test in the paper are task granularity (clip-level captioning) and camera motion. |
| DeltaCNN (CVPR 2022), MotionDeltaCNN (ICCV 2023), Run-Length Tokenization (NeurIPS 2024) | Frame-difference sparsity works on fixed cameras; moving cameras need alignment; with constant camera motion few tokens repeat. | Consistent with our token traces on drone footage. Our gates do no motion alignment, so say so. |
| Déjà Vu (PVLDB 2025), STC-Cacher (CVPR 2026), VLA-Cache (NeurIPS 2025), Temporal Token Reuse on UAV (JSTARS 2026) | Inter-frame reuse inside VLM encoders and on UAV video, all at fixed sampling rates with no rate baseline. Attention-layer reuse kernels gave no speedup, because attention is a small share of cost. | Token caching in a VLM encoder and gating on UAV video are not new. Caption-level evaluation against a rate baseline was not found. |
| Apollo (CVPR 2025), Singularity (ACL 2023), ATP (CVPR 2022); ERA (GRSM 2020); TemporalBench (preprint); SwinBERT (CVPR 2022) | Few frames often suffice for clip-level tasks. On ERA one frame reaches 62.3% against 64.3% for the best video model (checked). CIDEr can be flat between 1 and 64 frames (checked). More frames help captioning on dynamic data. | Supports our caption result. It also obliges a metric caveat, or an embedding-similarity metric, and a discussion of when frames matter. |

Not found in the sources searched (DBLP, OpenReview and MDPI blocked automated access; Google Scholar, IEEE Xplore and the ACM DL were not searched):
- an equal-cost comparison in which gating loses to frame-rate reduction;
- caption quality measured with each sampled frame replaced by an older one;
- a naive, tiled and fused FP32 dense-attention ladder on a small consumer GPU compared at equal clocks, with a launch, compute and memory breakdown and the effect of the software power cap.

### 3.3 Contributions: a floor and a ceiling

**Floor (always achievable, and it meets the course requirement on its own):**
1. A from-scratch ladder of attention kernels (naive, tiled, fused) on a 4 GB GPU.
2. A measured breakdown of where time goes at realistic sequence lengths: launch and dispatch, compute, memory traffic, transfers.
3. The memory wall shown directly: the sequence length where each version fails. It follows from N x N x 4 bytes, so present it as a check of that simple memory model rather than as a finding.

**Ceiling (optional; what would make it worth submitting beyond the course):**
4. A gated attention kernel that computes outputs only for changed queries.
5. A cost model for the break-even keep-rate, validated against measurements.
6. Evaluation on change traces from real aerial footage.

**Decision gate, end of week 5:** pursue the ceiling only if both hold:
- G1 passes its correctness tests.
- Real aerial footage reaches keep-rates clearly below the break-even estimated from the predicted times (section 3.1), at some frame rate. X0 (week 2) is a pixel-level proxy and the early warning. The token-level traces (week 5) are the real check, because a pixel mask is exact only at the patch embedding (section 3.4).

Otherwise write the floor paper and treat the gated kernel as future work. G1 losing to the baselines does not fail the gate: a measured and explained loss is a result (section 6).

**Gate decided, 2026-11-25 (the authors' decision, after the token traces, the caption-level check and X2):** write the floor paper, with a measured section on why temporal gating does not pay for this captioning pipeline. G1 and the rest of the ceiling become future work. Paper: IEEE conference format (IEEEtran), 8 pages, in `paper/`.

**Token traces, 2026-11-24** (`bench/tokens_era.py`, `results/tokens/era_tra/summary.json`, `plots/tokens_era_error.png`, `plots/tokens_era_spread.png`). 36 ERA training clips, 12 each with the camera still, moving and untrackable, plus 3 clips that look still but the hover rule rejected. Error is measured on the connector output, i.e. what the language model sees, at 24 fps.
- **Written criterion: met by every group once the threshold is high enough, so on its own it does not discriminate.** Keep-rate says nothing about the error at that threshold. Against the predicted break-evens (0.8-0.9 for rows-only G1, 0.42 for the exact update):
  - Pixel mask at 24 fps, thresholds 1/4/8: still 0.48/0.18/0.10, moving 0.76/0.55/0.41, untrackable 0.85/0.62/0.47.
  - Token deltas at 24 fps, thresholds 0.1/0.3/0.5: still 0.30/0.09/0.03, moving 0.68/0.36/0.18, untrackable 0.78/0.44/0.23.
- **A baseline the gate did not include: dense at a lower frame rate, with the output held.**
  - At equal cost, per clip, against the chord between the strides run (counts over the three main groups; the 3 still-by-eye clips are not in them):
    - Stale rows have the lower error in 30 of 412 settings.
    - Token deltas with exact attention have it in 3 of 360.
    - Stale input (a floor no kernel reaches) wins 213 of 412, mostly by 1-2 points. That margin is smaller than what ±1 grey level of noise does (2.8%).
  - The cost axis flatters the gates: FLOP shares, whole-layer skipping, no fixed costs. The loss is therefore conservative.
- Connector-output distance is not caption change; the caption-level check below tests that.

**Caption-level check, 2026-11-24** (`eval/gate_captions.py`, `results/gate_captions/scores.json`, `plots/gate_captions.png`). 100 CapERA test clips (E1's first 100) through E1's frozen captioner. Each condition replaces only the connector outputs at E1's four frames. Metrics were fixed before the run (commit 1f6d9b0).
- **No condition changes caption quality detectably.**
  - Every condition's CIDEr-D is within 0.03 of dense's (0.267), with 95% intervals of ±0.02-0.06.
  - Resolution: scoring dense captions against another clip's references costs 0.218 [0.142, 0.305]. Losses smaller than about a quarter of that cannot be ruled out.
  - The informative cases are corrupted outputs with no detected loss:
    - stale rows at threshold 8: 27% connector error, +0.001 [-0.038, +0.040]
    - token deltas at 0.2: 21%, -0.000 [-0.041, +0.041]
  - Held output at 1 fps (36% connector error, +0.003 [-0.039, +0.043]) is close to a null by construction. It still feeds four real frames, each under a second older.
  - Stale input at threshold 2: -0.029 [-0.059, -0.006]. That is one of 12 unadjusted intervals, and with CIDEr's length penalty off it becomes -0.030 [-0.064, +0.003].
- **Equal cost: no gate differs from held output** (every interval includes 0). This covers only the costs the gates reached, 9-20 full-frame equivalents per second on average. Held output reaches cost 1 with no detected loss, so on caption quality it dominates every gate tested.
- **Wording changes even when quality does not.**
  - bf16 rounding alone (1% connector error) changes 35% of captions.
  - The identical-caption rate falls with connector error for every method; see the matched-error table in `scores.json`.
- **The trace ordering holds on test clips.** At the caption frames, connector error over held output at equal cost is +4.4 and +11.5 points for stale rows, and +2.2 and +12.2 for token deltas.
- **Scope.** Clip-level CapERA captions from four frames per clip. A task that needs sub-second detail could behave differently; nothing here tests that.
- **Still open at the time.** G1 did not exist, and the gate was undecided. It was decided on 2026-11-25 (above).

### 3.4 Kernels

Standalone CUDA C++, one benchmark binary each. Every version passes a correctness test against a CPU reference before it is timed.

| ID | Kernel | Purpose |
|---|---|---|
| K0 | Block change map: mean absolute difference per 16x16 block between two frames (serial, OpenMP, CUDA) | Warm-up for the toolchain; measures real keep-rates early (X0); later produces change masks. Run it on the frame stretched to 512x512, as the model sees it, so each block is one encoder patch. |
| V1 | Attention as three kernels (scores, row softmax, output), storing N x N | Naive baseline |
| V2 | Tiled matrix multiplies with shared memory | Middle rung |
| V3 | One fused kernel with online softmax, no N x N storage | Dense fused baseline |
| G1 | Gated attention in one launch: takes a change mask, compacts the changed queries (prefix sum), computes M x N attention, scatters results into the cached output | Core of the ceiling |
| G2 | G1 with change detection fused in (token delta against a stored reference, thresholded), so the whole step is one launch | Tests H2 |

Design question to settle in week 1 after reading Eventful Transformers section 4. G1 as described recomputes changed rows only. The error in the other rows is structural, not just drift: every unchanged query also attends to the keys and values that changed, so its cached output is wrong as soon as any token changes. A pixel-level mask is also exact only at the patch embedding. After the first attention layer, change reaches every token, so deeper layers need token-delta thresholds. Options:

1. Accept stale rows and measure the error.
2. Add a periodic full refresh, which bounds the drift.
3. Exact update (novelty unchecked). Cache each row's online-softmax state, which V3 computes anyway: running max m, denominator l, unnormalised output o. When the tokens in set C change, update every unchanged row exactly: remove the old exp(s_ij - m_i) terms and their v_j contributions for j in C, then add the new ones. Old scores come from q_i and a cached copy of the old K rows. Estimated work is k(3 - 2k) of dense attention at keep-rate k (the changed rows, plus about 4d multiply-adds per unchanged row and changed key), so about 3x option 1 at small k and break-even with dense near k = 0.5 before fixed costs; O(N x d) memory and no N x N cache. If a row put most of its weight on changed keys, the subtraction loses precision; recompute that row instead.

Measure before deciding.

**Measured on the ERA token traces** (2026-11-24; status note under the decision gate, section 3.3):
- **Option 1 (stale rows): loses to dense at a lower frame rate at equal cost on 382 of 412 settings.**
  - At 24 fps and pixel threshold 1, the group-mean errors of the two clip halves agree within 0.4 points, so there is no drift there.
  - At higher thresholds the error drifts. Over the first half of each clip, stale rows still lose to hold on 363 of 412 settings (`equal_cost.B_first_half`).
  - Option 2 (periodic refresh) was not run, and it adds cost.
- **Tier C (option 3's exact attention under token-delta gates): its attention is exact.** Its error comes from the delta thresholds on the q/k/v and MLP inputs, the layers that hold most of the FLOPs.
  - An exact attention update alone adds no error.
  - It caps the saving at attention's share, about 18% of a layer's FLOPs.
- **Settled by the gate decision (2026-11-25):** G1 is future work, so the choice between the options goes with it.

Single head and FP32 first, for correctness. Add multiple heads early, since it is one more grid dimension and timings must use the real shape (12 heads). FP16 only after everything is correct.

For small M, G1 should also split the key range across blocks and merge the partial results (the Flash-Decoding approach). Otherwise one block sweeps all N keys alone while most of the GPU's 16 SMs sit idle.

### 3.5 Baselines

- **B1:** library attention on all N tokens (PyTorch `scaled_dot_product_attention`).
- **B2:** select changed queries, then library attention on M x N. This is the "standard operators" approach. Boolean selection forces a GPU-to-CPU sync to learn M.
- **B3:** our own V3 on all N tokens.
- **B4:** our own kernels for the gated steps as separate launches, in C++: mask, prefix-sum compaction, V3 on M x N, scatter. G1 against B4 isolates fusion; B2 against B4 isolates Python dispatch and the sync.

Time B1 and B2 with and without CUDA Graphs. Pin the `scaled_dot_product_attention` backend and compare at matched precision (in FP32 the Flash backend is not available).

### 3.6 Experiments

| ID | Question | Sweep | Output |
|---|---|---|---|
| X0 | Is there redundancy to exploit? (gate, section 3.3) | K0 on 2-3 hovering and 2-3 moving aerial clips (need not be the demo videos: ERA clips, or the Stanford Drone Dataset for hovering footage, terms to check), from 1 fps up to the clips' native rate, several thresholds. Two modes: consecutive-sample pairs, and a per-block reference refreshed only when the block crosses the threshold (what a gate actually pays; pairs understate it). Shots hand-labelled hover / moving / non-aerial, with transitions excluded (`bench/x0_segments/`) | Keep-rate per sample and full-frame equivalents per second of video, vs frame rate and threshold, per footage type |
| X1 | How do V1-V3 scale? | N from 128 to 32,768, d = 64 | Runtime and peak memory vs N; where each version fails. At N = 32,768 the FP32 score matrix alone is 4 GiB. V1 and V2 both store N x N, so they fail at the same N. |
| X2 | Where does the time go at realistic N? (H1) | N = 1024, d = 64, 12 heads | Stacked breakdown: launch, compute, transfer; an empty-kernel launch as the fixed-cost floor |
| X3 | When does gating pay off? (H2) | Keep-rate M/N from 5% to 100%; N = 1024 and others; 12 heads | Latency vs keep-rate for G1, G2, B1, B2, B3, B4; break-even points |
| X4 | Can we predict break-even? (H3) | Fit on some N values, test on held-out N values | Predicted vs measured break-even table |
| X5 | Does it hold on real footage? | Replay change traces from aerial videos | Keep-rate distribution, per-frame latency distribution, output error against full recomputation over time |
| X6 (stretch) | Does it generalise? | Second GPU (cloud or borrowed); kernel placed inside the vision encoder | Same plots on another device; end-to-end timing |

### 3.7 Measurement protocol

- Laptop plugged in, performance mode, nothing else on the GPU.
- Warm-up runs discarded. At µs scale, time many back-to-back iterations per sample; take at least 20 samples and report the median with spread.
- Interleave configurations instead of running each one in a block.
- Log SM clock, temperature and throttle reasons with every run (laptop GPUs throttle and share power with the CPU). Discard throttled runs.
- Time kernel-only and end-to-end separately. H1 and H2 are about the difference.
- Record GPU model, driver, CUDA toolkit and compiler versions in every result file. Lock the kernel and driver packages (`dnf versionlock`) during measurement weeks.
- Use a profiler (Nsight Compute or equivalent) for the explanations, not just wall-clock time. Nsight Compute's counters are admin-only on Linux by default, and it locks clocks to base by default, so use it for counters, not headline timings.
- Every figure regenerated by script from raw result files.

### 3.8 Paper outline

1. Introduction: video redundancy, small GPUs, the gap between theoretical and realised savings. Why attention: the projections and MLP dominate a ViT block's FLOPs and are easy to gate (gather, GEMM, scatter); once they are gated, ungated attention becomes the bottleneck. Measure the attention share with a profiler in week 2 rather than quoting an estimate.
2. Background: attention cost, fusion and tiling, temporal gating
3. Kernel designs: V1-V3, G1-G2
4. Cost model
5. Experimental setup
6. Results: X0-X5
7. Limitations: one GPU family, forward pass only, approximation error, camera motion. The demo captions at 1 fps, where the encoder is likely a small share of run time next to text generation (unmeasured), so do not claim demo speed-ups. The paper's setting is high-frame-rate encoding.
8. Related work, conclusion

Planned figures: keep-rate vs frame rate (X0); runtime vs N; memory vs N; time breakdown; latency vs keep-rate; break-even predicted vs measured; keep-rate distribution on aerial footage.

Start the document in week 2 with the outline and empty figure slots. Fill it as results arrive.

---

## 4. Timeline

There is no deadline, so week numbers give the order of work, not dates.

| Week | Part A: demo | Part B: paper | Exit criterion |
|---|---|---|---|
| 1 | Model captions one clip offline; ERA videos downloaded | Literature search; question frozen; toolchain; K0; predicted time per kernel | One caption and one timed kernel on the laptop |
| 2 | CLI works end to end on one video | V1; benchmark harness; B1, B2; X0 keep-rate curves; attention share profiled; paper skeleton with intro and related-work drafts | First report file; first runtime plot; keep-rate curves |
| 3 | Report template tuned; three demo videos chosen; E1 and E2 timed on 50 clips | V2; V3 started; X2 at N = 1024, 12 heads | Demo runs on all three videos |
| 4 | **Demo frozen**; acceptance test passes; fallback recording made | V3 done; X1; G1 prototype | Acceptance test passes |
| 5 | Evaluation E1-E4 on the frozen demo; trace export for Part B (encoder only) | G1 correct; B4 | **Decision gate** (section 3.3) |
| 6 | Evaluation written up | Ceiling: X3; G2; X4. Floor: profiler analysis for X1 and X2; draft starts | Evaluation written up; ceiling: latency-vs-keep-rate plot exists |
| 7 | UI polish only if spare time | Ceiling: X5. Both: second GPU if available | All experiments have raw results |
| 8 | - | Final figures; full draft | Complete draft, every number traceable to a file |
| 9 | Demo rehearsal | Internal review and revision | Both parts done (top of this file) |

## 5. Team split

- **3 people:** one on the demo for weeks 1-4, then moves to traces, benchmarking and writing. Two on kernels throughout.
- **4 people:** the fourth owns the benchmark harness, measurement protocol and the paper document from week 1.

The paper is the harder goal, so it gets most of the people for most of the time.

## 6. Risks

| Risk | Response |
|---|---|
| The gated-kernel idea is already published | Week-1 literature search. The floor paper is a measurement study and survives this. |
| Aerial footage changes too much between frames for gating to pay off (camera motion; the demo captions at 1 fps) | X0 in week 2, before the gate. If moving footage shows little redundancy, limit the claim to hovering segments and say so. MotionDeltaCNN is the prior work for moving cameras. |
| G1 or G2 is not finished | Decision gate in week 5; floor paper. |
| Gating never beats the baselines on this GPU | That is a publishable negative result if the breakdown explains why. Report it as measured. |
| Stale outputs from row-only gating drift too far | Measure in X5; add periodic refresh or the exact update (section 3.4); report the trade-off. |
| Results from one GPU look narrow | Add a second device in X6, or state it plainly under limitations. |
| Custom dense kernel is slower than the library | Expected. The paper is about the gated case and the overhead breakdown. |
| Demo eats paper time | Hard freeze in week 4. UI work is optional after that. |
| Model too slow for a live demo | Shorter video, fewer frames per window, or the 256M model. Fallback recording regardless. |
| One GPU for the whole team | Develop on CPU or a free cloud GPU; book the laptop for measurement runs; all reported numbers come from a named device. |
| Driver, CUDA toolkit and PyTorch versions disagree on Fedora | Resolved in week 1: CUDA 13.5 accepts GCC 16.2, and PyTorch runs from a uv environment on Python 3.12 (`results/cuda/toolchain.txt`). Lock the driver and toolkit packages during measurement weeks. |
| CapERA has no stated licence | Fine for coursework and internal checks; confirm terms before putting its content in a published paper. Part B does not depend on it. |
| ERA videos unavailable (CapERA ships captions only) | Check in week 1. Fallback: ask the ERA authors; otherwise hand-caption about 100 public aerial clips and evaluate on those. |

## 7. Repository layout

```
demo/          run_mission.py, sampling, captioning, report, trace export
eval/          CapERA evaluation (E1, E2, E4); demo-video event lists and scoring (E3)
kernels/
  k0_changemap/   serial, openmp, cuda
  attention/      v1, v2, v3, g1, g2, b4, reference, tests
bench/         harness, configs, protocol scripts
traces/        exported from the demo (generated)
results/       raw timings and logs (generated)
plots/         scripts and figures (generated)
paper/         source for the paper
```

## 8. Week-1 checklist

- [x] Both instructors approve the plan
- [x] Conference template chosen: IEEE conference (IEEEtran), 8 pages (`paper/`)
- [x] Ask the NLP instructor what form the evaluation should take (section 2.5): no prescribed form; we use E1-E4
- [ ] Roles written into this file
- [x] NVIDIA driver, CUDA toolkit and PyTorch working together; versions committed (`results/cuda/toolchain.txt`)
- [x] Nsight Compute can read performance counters (`NVreg_RestrictProfilingToAdminUsers=0`; `make -C kernels/smoke profile`)
- [x] SmolVLM2-500M captions one aerial clip with the network off; N = 1024 confirmed (`demo/caption_clip.py`)
- [x] ERA videos downloaded; all 2,864 match CapERA; per-clip scan in `results/era/` (DATA.md)
- [x] A trivial CUDA kernel compiles, runs and is timed (`kernels/smoke`, `results/cuda/smoke.json`)
- [x] Predicted time per kernel written down (section 3.1; `results/cuda/predicted_times.json`)
- [ ] Everyone on Part B has read the papers in section 3.2
- [x] Literature search done; novelty table updated; research question frozen (2026-11-25; section 3.1, section 3.2, `docs/related_work.md`)
- [x] Stale, refresh or exact-update design question (section 3.4): settled by the gate decision, G1 is future work

## 9. References

- SmolVLM paper: https://arxiv.org/abs/2504.05299
- SmolVLM2 announcement: https://huggingface.co/blog/smolvlm2
- SmolVLM2-500M model card: https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
- CapERA dataset: https://github.com/yakoubbazi/CapEra
- CapERA paper: https://www.mdpi.com/2072-4292/15/8/2139
- ERA dataset (CapERA's videos): https://lcmou.github.io/ERA_Dataset
- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashAttention-2 as a fused CUDA kernel, case study: https://arxiv.org/abs/2312.11918
- From-scratch FlashAttention in CUDA: https://sanket-pixel.github.io/projects/1_project
- Eventful Transformers: https://arxiv.org/abs/2308.13494
- Dispatch-Aware Ragged Attention for Pruned Vision Transformers: https://arxiv.org/abs/2604.15408
- Pruning One More Token is Enough: https://arxiv.org/abs/2407.05941
- DeltaCNN: https://arxiv.org/abs/2203.03996
- MotionDeltaCNN: https://arxiv.org/abs/2210.09887
- Skip-Convolutions: https://arxiv.org/abs/2104.11487
- Run-Length Tokenization: https://arxiv.org/abs/2411.05222
- On-device VLM inference measurement study on edge NPUs: https://arxiv.org/abs/2610.40039
