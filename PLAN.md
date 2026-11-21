# PLAN: Onboard Drone Perception

Two outcomes from one codebase:

- **Part A (NLP course):** a working offline demo. MVP quality is enough.
- **Part B (Parallel Programming course):** a research paper.

## 0. Confirmed and assumed

**Confirmed**
- Both instructors have approved two deliverables built on one codebase.
- The paper is a course paper in conference format. No venue submission is required.

**Still assumed (correct these first)**

| Assumption | If it is wrong |
|---|---|
| Team of 3-4, about 8 weeks | Weeks are relative. Cut Part B's ceiling (section 3.3) before anything else. |
| Target device: one laptop GPU, RTX 3050 with 4 GB VRAM, Fedora Linux | Re-check which models fit. |
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

### 2.3 Build steps

1. Load the model from local files; caption one clip with the network disabled.
2. Windowing: split a video into fixed windows, sample a few frames per window, caption each window with a fixed prompt.
3. Report generation: feed the timestamped captions to the language model (the same VLM's text side, or a small local text model) with a fixed report template.
4. Wrap in the CLI; log latency per window and peak VRAM.
5. Trace export for Part B: dump the vision encoder's sequence length and per-layer token tensors for a few videos.

### 2.4 Acceptance test

- Wi-Fi off, fresh terminal, one command.
- A 2-3 minute aerial video produces a report in a time you are happy to show live (decide the budget after the first run).
- The report names the main events in each of the three prepared videos.
- A screen recording of a successful run exists as a fallback for demo day.

### 2.5 Light evaluation (half a day, optional but cheap)

Run 50 CapERA clips, report one caption metric plus five good and five bad examples. This is for the write-up and for honest slides, not for research claims.

*The quantisation, frame-budget and token-budget experiments from the earlier plan are dropped from Part A. They can return as backlog if the demo finishes early.*

---

## 3. Part B - Parallel Programming paper

### 3.1 Research question

> Consecutive video frames barely differ, so in principle only the tokens that
> changed need new attention outputs. On a small, memory-limited GPU, how much of
> that theoretical saving does a fused CUDA kernel actually deliver, and which
> overheads consume the rest?

Hypotheses:

- **H1.** A fused attention kernel removes the memory wall on a 4 GB GPU (it runs where the naive kernel cannot) but gives little latency benefit at vision-transformer sequence lengths, where fixed per-launch costs dominate.
- **H2.** A gated kernel that does change detection, compaction and attention in one launch realises more of the theoretical saving than selecting tokens first and then calling library attention.
- **H3.** A simple cost model (fixed launch cost plus a term in M x N) predicts the keep-rate at which gating starts to pay off.

N is the number of tokens; M is the number that changed since the last frame.

### 3.2 Position against prior work

| Work | What it established | What it leaves open for us |
|---|---|---|
| FlashAttention (2022) and successors | Tiling plus kernel fusion avoids storing the N x N matrix; tuned for long sequences on data-centre GPUs | Behaviour at short sequences on a 4 GB consumer GPU |
| Eventful Transformers (ICCV 2023) | Re-processing only changed tokens in video saves compute, built from standard operators | A purpose-built fused kernel for the gated case; behaviour under tight memory |
| Dispatch-Aware Ragged Attention (2026) | At ViT-scale lengths, host-side dispatch overhead hides token-pruning savings; a custom kernel helps | Temporal gating (subset of queries against all keys) rather than pruned single images |
| Pruning One More Token is Enough (2024) | Latency does not fall smoothly with workload size on edge GPUs | The same effect for gated attention kernels |

**This table is a starting point, not a novelty guarantee.** A quick search did not turn up a fused gated-attention kernel evaluated on a memory-limited GPU, but week 1 must include a proper literature search before the question is frozen.

### 3.3 Contributions: a floor and a ceiling

**Floor (always achievable, and it meets the course requirement on its own):**
1. A from-scratch ladder of attention kernels (naive, tiled, fused) on a 4 GB GPU.
2. A measured breakdown of where time goes at realistic sequence lengths: launch and dispatch, compute, memory traffic, transfers.
3. The memory wall shown directly: the sequence length where each version fails.

**Ceiling (optional; what would make it worth submitting beyond the course):**
4. A gated attention kernel that computes outputs only for changed queries.
5. A cost model for the break-even keep-rate, validated against measurements.
6. Evaluation on change traces from real aerial footage.

**Decision gate, end of week 4:** if the gated kernel is not yet correct and at least matching the select-then-library baseline at low keep-rates, write the floor paper and treat the gated kernel as future work.

### 3.4 Kernels

Standalone CUDA C++, one benchmark binary each. Every version passes a correctness test against a CPU reference before it is timed.

| ID | Kernel | Purpose |
|---|---|---|
| K0 | Block change map: mean absolute difference per 16x16 block between two frames (serial, OpenMP, CUDA) | Warm-up for the toolchain; later produces change masks |
| V1 | Attention as three kernels (scores, row softmax, output), storing N x N | Naive baseline |
| V2 | Tiled matrix multiplies with shared memory | Middle rung |
| V3 | One fused kernel with online softmax, no N x N storage | Dense fused baseline |
| G1 | Gated attention: takes a change mask, compacts the changed queries (prefix sum), computes M x N attention, scatters results into the cached output | Core of the ceiling |
| G2 | G1 with change detection fused in (token delta against a stored reference, thresholded), so the whole step is one launch | Tests H2 |

Design question to settle in week 1 after reading Eventful Transformers section 4: G1 as described recomputes changed rows only and leaves unchanged rows stale, which is an approximation. Either accept it and measure the error, or add a periodic full refresh. Measure before deciding.

Single head and FP32 first. Add multiple heads and FP16 only after everything is correct.

### 3.5 Baselines

- **B1:** library attention on all N tokens (PyTorch `scaled_dot_product_attention`).
- **B2:** select changed queries, then library attention on M x N. This is the "standard operators" approach.
- **B3:** our own V3 on all N tokens.

### 3.6 Experiments

| ID | Question | Sweep | Output |
|---|---|---|---|
| X1 | How do V1-V3 scale? (H1) | N from 128 to 32,768, d = 64 | Runtime and peak memory vs N; where each version fails. At N = 32,768 the FP32 score matrix alone is 4 GiB. |
| X2 | Where does the time go at realistic N? (H1) | N taken from the demo's traces | Stacked breakdown: launch, compute, transfer |
| X3 | When does gating pay off? (H2) | Keep-rate M/N from 5% to 100%, several N | Latency vs keep-rate for G1, G2, B1, B2, B3; break-even points |
| X4 | Can we predict break-even? (H3) | Fit on half the configurations, test on the rest | Predicted vs measured break-even table |
| X5 | Does it hold on real footage? | Replay change traces from aerial videos | Keep-rate distribution, per-frame latency distribution, output error against full recomputation over time |
| X6 (stretch) | Does it generalise? | Second GPU (cloud or borrowed); kernel placed inside the vision encoder | Same plots on another device; end-to-end timing |

### 3.7 Measurement protocol

- Laptop plugged in, performance mode, nothing else on the GPU.
- Warm-up runs discarded; median of at least 20 runs with spread shown.
- Time kernel-only and end-to-end separately. H1 and H2 are about the difference.
- Record GPU model, driver, CUDA toolkit and compiler versions in every result file.
- Use a profiler (Nsight Compute or equivalent) for the explanations, not just wall-clock time.
- Every figure regenerated by script from raw result files.

### 3.8 Paper outline

1. Introduction: video redundancy, small GPUs, the gap between theoretical and realised savings
2. Background: attention cost, fusion and tiling, temporal gating
3. Kernel designs: V1-V3, G1-G2
4. Cost model
5. Experimental setup
6. Results: X1-X5
7. Limitations: one GPU family, forward pass only, approximation error
8. Related work, conclusion

Planned figures: runtime vs N; memory vs N; time breakdown; latency vs keep-rate; break-even predicted vs measured; keep-rate distribution on aerial footage.

Start the document in week 2 with the outline and empty figure slots. Fill it as results arrive.

---

## 4. Timeline

| Week | Part A: demo | Part B: paper | Exit criterion |
|---|---|---|---|
| 1 | Model captions one clip offline | Literature search; question frozen; toolchain; K0 | One caption and one timed kernel on the laptop |
| 2 | CLI works end to end on one video | V1; benchmark harness; paper skeleton with intro and related-work drafts | First report file; first runtime plot |
| 3 | Report template tuned; three demo videos chosen | V2; V3 started; X2 measurements at realistic N | Demo runs on all three videos |
| 4 | **Demo frozen**; acceptance test passes; fallback recording made | V3 done; X1; G1 prototype | **Decision gate** (section 3.3) |
| 5 | Trace export for Part B | G1 correct; G2; X3 | Latency-vs-keep-rate plot exists |
| 6 | UI polish only if spare time | X4; X5; second GPU if available | All experiments have raw results |
| 7 | - | Final figures; full draft | Complete draft, every number traceable to a file |
| 8 | Demo rehearsal | Internal review, revision, submission | Both submitted. Keep half the week as buffer. |

## 5. Team split

- **3 people:** one on the demo for weeks 1-4, then moves to traces, benchmarking and writing. Two on kernels throughout.
- **4 people:** the fourth owns the benchmark harness, measurement protocol and the paper document from week 1.

The paper is the harder goal, so it gets most of the people for most of the time.

## 6. Risks

| Risk | Response |
|---|---|
| The gated-kernel idea is already published | Week-1 literature search. The floor paper is a measurement study and survives this. |
| G1 or G2 is not finished | Decision gate in week 4; floor paper. |
| Gating never beats the baselines on this GPU | That is a publishable negative result if the breakdown explains why. Report it as measured. |
| Stale outputs from row-only gating drift too far | Measure in X5; add periodic refresh; report the trade-off. |
| Results from one GPU look narrow | Add a second device in X6, or state it plainly under limitations. |
| Custom dense kernel is slower than the library | Expected. The paper is about the gated case and the overhead breakdown. |
| Demo eats paper time | Hard freeze in week 4. UI work is optional after that. |
| Model too slow for a live demo | Shorter video, fewer frames per window, or the 256M model. Fallback recording regardless. |
| One GPU for the whole team | Develop on CPU or a free cloud GPU; book the laptop for measurement runs; all reported numbers come from a named device. |
| Driver, CUDA toolkit and PyTorch versions disagree on Fedora | Pin and commit versions in week 1. |
| CapERA has no stated licence | Fine for coursework and internal checks; confirm terms before putting its content in a published paper. Part B does not depend on it. |

## 7. Repository layout

```
demo/          run_mission.py, sampling, captioning, report, trace export
kernels/
  k0_changemap/   serial, openmp, cuda
  attention/      v1, v2, v3, g1, g2, reference, tests
bench/         harness, configs, protocol scripts
traces/        exported from the demo (generated)
results/       raw timings and logs (generated)
plots/         scripts and figures (generated)
paper/         source for the paper
```

## 8. Week-1 checklist

- [x] Both instructors approve the plan
- [ ] Conference template chosen (ask the Parallel Programming instructor which one)
- [ ] Deadline and roles written into this file
- [ ] NVIDIA driver, CUDA toolkit and PyTorch working together; versions committed
- [ ] SmolVLM2-500M captions one aerial clip with the network off
- [ ] A trivial CUDA kernel compiles, runs and is timed
- [ ] Everyone on Part B has read the four papers in section 3.2
- [ ] Literature search done; novelty table updated; research question frozen
- [ ] Row-only versus refresh design question (section 3.4) answered

## 9. References

- SmolVLM paper: https://arxiv.org/abs/2504.05299
- SmolVLM2 announcement: https://huggingface.co/blog/smolvlm2
- SmolVLM2-500M model card: https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
- CapERA dataset: https://github.com/yakoubbazi/CapEra
- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashAttention-2 as a fused CUDA kernel, case study: https://arxiv.org/abs/2312.11918
- From-scratch FlashAttention in CUDA: https://sanket-pixel.github.io/projects/1_project
- Eventful Transformers: https://arxiv.org/abs/2308.13494
- Dispatch-Aware Ragged Attention for Pruned Vision Transformers: https://arxiv.org/abs/2604.15408
- Pruning One More Token is Enough: https://arxiv.org/abs/2407.05941
- On-device VLM inference measurement study on edge NPUs: https://arxiv.org/abs/2610.40039
