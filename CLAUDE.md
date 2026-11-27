# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Two course projects on one codebase, both on one RTX 3050 Laptop GPU (4 GB, 16 SMs, `sm_86`):

- **Part B (Parallel Programming): a paper**, "Fewer Frames Beat Smarter Kernels" (`paper/`, full draft). FP32 attention kernels V1 (naive), V2 (tiled) and V3 (fused, online softmax) at the SmolVLM2 vision encoder's shape (12 heads, N = 1024, d = 64), plus a measured section on why temporal gating does not pay.
- **Part A (NLP): an offline aerial-video captioner** on SmolVLM2-500M, with evaluations E1-E4. The demo (`demo/run_mission.py`), E1, E2 and E3 exist (E3 on the three demo videos in `eval/e3_events/`, before any freeze), and E4's 50 clips are chosen; `PLAN.md` section 2.6 is the checklist for the rest (the freeze decision, E4 tagging, the write-up).

`PLAN.md` holds every decision with its date, including the gate decision (section 3.3): the gated kernel G1 is future work. `DATA.md` lists the data sources and their terms.

## Which machine

**Assume you are on the measurement laptop**, the machine every committed measurement comes from, and check at the start of any session that needs the GPU, models or data:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader   # expect: NVIDIA GeForce RTX 3050 Laptop GPU
ls -d models/SmolVLM2-500M-Video-Instruct data/capera data/era/Videos
```

- **Both checks pass: you are on the laptop.** Everything in this file applies as written. That includes writing new measurements into `results/` when a run is meant to replace one.
- **Either check fails: you are on another machine.** Follow [Otherwise: another machine](#otherwise-another-machine). Where it differs from the rest of this file, it wins. Tell the user you are not on the laptop before running anything on the GPU.

About the laptop:
- RTX 3050 Laptop GPU (4 GB, 16 SMs, `sm_86`), Fedora 44.
- CUDA 13.5 (it accepts the system GCC 16), and PyTorch 2.15 built for CUDA 13.2. The kernels are standalone binaries, so the two CUDA versions need not match. Versions are in `results/cuda/toolchain.txt`.
- Nsight Compute's counters are enabled for normal users (`NVreg_RestrictProfilingToAdminUsers=0`).
- Liberation Serif and TeX Live are installed.

## Rules for this repository

- **Commits:** never add `Co-Authored-By`, `Claude-Session` or "Generated with Claude Code" lines to commits, PRs or releases. Commit only when asked.
- **Python:** run scripts as `.venv/bin/python -I <script>`. Scripts add their own directories to `sys.path`, so a module named like a stdlib module shadows it (`paper/numbers.py` once broke numpy; it is now `make_numbers.py`). Pass data paths as arguments. Scripts that load models set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` before importing transformers; keep that.
- **Data terms:**
  - Never commit, publish or send ERA frames anywhere: file names and numbers only.
  - Never write CapERA reference captions into committed files: only our model's captions and scores.
  - `models/`, `data/`, `traces/`, `build/` and `results/**/raw/` are git-ignored.
- **The user does some steps themselves:**
  - Root-only commands, such as `sudo nvidia-smi -lgc 1395,1395` to pin the clock and `sudo nvidia-smi -rgc` to reset it.
  - Large downloads: give the exact command. `hf download` reads `--include` patterns as file names, so list files explicitly.
- **Measurements:** run with the laptop on AC, in the performance profile, with nothing else on the GPU. Unpinned, each kernel settles at its own clock under the power cap, so compare kernels at a pinned clock (`bench/clock_pinned.py`). Use Nsight Compute for counters, not for headline timings.
- **Fix analyses before running them.** Write the metrics and decision rules in a docstring or config and commit them before the run (e.g. `eval/gate_captions.py`, `eval/e1_config.json`). Label any analysis designed after seeing data *post hoc*, in code and in the paper.

## Commands

```bash
uv sync                                            # Python 3.12 environment in .venv

# CUDA (nvcc from CUDA_HOME, default /usr/local/cuda; ARCH=sm_86)
make -C kernels/attention test                     # V1-V3 against the float64 reference; exit 0 = all pass
cd kernels/attention && make build/attn_test build/testdata/cases.txt && ./build/attn_test v3   # one version (reads build/testdata relative to this directory)
make -C kernels/attention bench VERSION=v3         # results/attention/v3.json
make -C kernels/attention tune VERSION=v2          # tile sweep at 12 heads, N = 1024
make -C kernels/attention x2                       # fixed costs and host-device copies
make -C kernels/attention x2-profile               # warp-stall profiles (needs perf-counter access)
make -C kernels/k0_changemap test test-cuda        # K0 change map: serial/OpenMP and CUDA
make -C kernels/harness selftest                   # C++ timing harness
make -C kernels/microbench run                     # FMA, DRAM and launch limits

# Python benchmarks and checks
.venv/bin/python -I bench/b1_sdpa.py [out.json] [--only h12_n1024] [--dtype fp32|bf16|fp16]
.venv/bin/python -I bench/harness_selftest.py
.venv/bin/python -I bench/check_bench_schema.py results/attention/*.json

# Evaluations: subcommands, resumable
# Demo: one offline report per video (default output results/demo/<video name>/report.md and run.json)
unshare -rn .venv/bin/python -I demo/run_mission.py --video data/clips/x.mp4 --offline [--out DIR]
.venv/bin/python -I eval/e1.py {precheck,select,run,memory,score}
.venv/bin/python -I eval/e2.py {select,run,score}
.venv/bin/python -I eval/e4.py {select,show,count}       # E4: 50 clips for the team to tag
.venv/bin/python -I eval/e3.py score                       # E3: tallies results/e3/*_judgments.json
bash demo/prepare_videos.sh                               # cut the three demo videos (data/clips/demo_*.mp4)
.venv/bin/python -I eval/gate_captions.py {check,run,score}

# Paper
.venv/bin/python -I paper/make_numbers.py          # paper/generated/numbers.tex, table_ladder.tex
.venv/bin/python paper/figures.py                  # paper/figures/*.pdf and *.png
cd paper && latexmk -pdf -outdir=build main.tex    # always -outdir=build, or build files land in paper/
```

Add Python dependencies with `uv add`. There is no linter configuration and no Python test suite. Correctness lives in the CUDA test drivers, the harness self-tests and the `*_check.py` scripts in `bench/`.

## Architecture

**Kernel side.**
- `kernels/attention/attn.h` documents every version's layout and launch shape. Q, K, V and O are `[H][N][D]` row-major FP32 with D = 64, indexed in `size_t`.
- `versions.h` is the registry that the generic drivers read: `attn_test`, `attn_bench`, `attn_profile` and `x2_bench`. A three-phase version fills in `scores`, `softmax` and `output` and keeps N×N in its workspace; a fused version provides only `full`.
- To add a version, implement it, register it in `ATTN_VERSIONS`, and add its `.cu` file to `KERNELS` in the Makefile.
- The test reference comes from `bench/attn_testdata.py` (float64, written to `build/testdata`). Versions with alternative internal paths (tile shapes, softmax variants) are tested once per path.

**One benchmark schema, two harnesses.**
- `kernels/harness/harness.h` (C++) and `bench/harness.py` (PyTorch baselines) follow the same protocol: steady-clock warm-up, frozen iteration counts, 21 interleaved samples, per-sample SM clock and throttle flags.
- Both write `airgap-bench/1` JSON (`kernels/harness/SCHEMA.md`), which `bench/check_bench_schema.py` validates.
- C++ and Python ops cannot run in one process, so cross-language comparisons are between separate runs.

**Model access.** Both parts use the same frame path:
- `demo/sampling.py` decodes with PyAV.
- `eval/vlm.py` does captioning: bf16, no image splitting, greedy decoding, prompt as a parameter, preprocessing kept separate from generation.
- `bench/visionstack.py` runs the vision tower and connector standalone, layer by layer, in FP32 with TF32 off. Gating error is measured there, on the connector output.
- `bench/x0core.model_view` gives the exact pixels the encoder sees, so dense and gated runs match.

**Provenance chain: results → macros → paper.**
- Result files in `results/` are committed. Benchmarks and evaluations write there by default, so rerunning one replaces its committed result. The paper follows the next time `make_numbers.py` runs, so rerun only when replacing is the intent. Then regenerate, rebuild, and update the root README's numbers.
- `paper/make_numbers.py` turns them into TeX macros with `put(name, value, src, fmt)`. Every macro records its source file, and the name must be letters only. Signed formats (`{:+...}`) render through `\ensuremath`.
- `paper/derived.py` holds derivations shared by the macros and the figures, such as `pinned_breakdown()`.
- `paper/figures.py` draws the figures from the same files: IEEE widths, Liberation Serif TrueType.
- Never type a number into `main.tex` that a result file holds. Never edit `paper/generated/` or `paper/figures/` by hand.
- `paper/README.md` maps each section, figure and table to its source. The root README copies its numbers from `numbers.tex`, so update it when they change.
- `plots/` holds the working plots made during the experiments; the paper's figures supersede them.

**Gating experiments** emulate the gates in PyTorch and model their costs; no gated kernel exists:
- `bench/tokens_era.py` measures error against cost on ERA clips, compared at equal cost with dense encoding at a lower frame rate, output held.
- `eval/gate_captions.py` measures caption quality (CIDEr-D).
- `eval/caption_embed.py` adds the post hoc embedding metric.

## Otherwise: another machine

Use this section when either check under [Which machine](#which-machine) fails.

- **Never overwrite committed results.**
  - Benchmarks and evaluations write into `results/` by default: the Makefile targets `bench`, `tune`, `x2` and `x2-profile`, `bench/b1_sdpa.py`, and `eval/*.py`. A run on another GPU would replace the laptop's measurements, and `make_numbers.py` would then change the paper's numbers without warning.
  - Give every run an explicit output path outside `results/`, e.g. `./build/attn_bench v3 /tmp/v3.json` or `bench/b1_sdpa.py /tmp/b1.json`.
  - Make no changes under `results/`, `paper/generated/` or `paper/figures/` unless the user says the laptop's numbers should be replaced.
- **The paper rebuilds anywhere.** A fresh clone regenerates `paper/generated/`, `paper/figures/` (byte for byte) and the PDF from the committed results, with no GPU or data: run `uv sync`, then the three paper commands.
  - It needs TeX Live with `latexmk`, `cite`, `microtype`, `enumitem` and `booktabs`. The IEEEtran files are in `paper/`.
  - It needs the Liberation Serif fonts (`liberation-serif-fonts` on Fedora, `fonts-liberation` on Debian and Ubuntu). Without them matplotlib falls back to another font, and the figures change.
- **Another GPU:**
  - Build with `make ARCH=sm_XX`.
  - The 1395 MHz pin is the laptop's; another GPU needs its own.
  - `uv sync` installs a PyTorch built for CUDA 13.2, which needs a recent enough driver.
  - Profiling targets need performance-counter access (`NVreg_RestrictProfilingToAdminUsers=0`, or root).
- **No models or data.** Download them as `DATA.md` describes:
  - SmolVLM2-500M goes into `models/`.
  - CapERA goes into `data/capera/`.
  - The ERA videos are a manual Google Drive download into `data/era/`, for academic use only.
  - Until they are there, only the kernel tests and benchmarks and the paper build can run.

## Building on what exists

**Part A: reuse these, don't rewrite them.**
- `eval/capera.py`: `load(split)` returns `Clip` (split, id, class, path, duration, references). Clips are keyed on (split, id), because ids repeat across splits. `stratified_order` gives a class-balanced order.
- `demo/sampling.py`: `probe` and `sample_window(path, start_s, window_s, k)`.
- `eval/vlm.py`: `Captioner`.
- `eval/e1.py`:
  - `sanitise`: the caption clean-up every metric uses;
  - `caption_clips`: resumable JSONL output;
  - `coco_scores`: BLEU, METEOR, ROUGE-L and CIDEr via pycocoevalcap.
- `eval/e2.py`: `parse`, the frozen answer-matching rule. E4 draws E2's misclassifications from `results/e2/per_clip_test.json`.
- `eval/gate_captions.py`: `interval` (bootstrap, 10,000 resamples, seed 0) and `chord` (equal-cost interpolation).
- `demo/run_mission.py` reads E1's frozen settings from `eval/e1_config.json` (prompt P1, 4 frames, no image splitting, greedy decoding, 48 new tokens) and the uncertainty cutoff from `demo/uncertainty.json`. Its event words (`EVENTS`) and cutoff were fixed before any demo video was run; changing either after a demo video has been run is post hoc.
- `Captioner.generate(..., logprobs=True)` adds the caption's mean token log-probability without changing the caption; `Captioner.text_inputs` builds a text-only prompt.
- Develop the demo on footage that is not a demo video (`data/clips/sheboygan_falls.webm`, or ERA train clips joined in a scratch folder), and send development output outside `results/`.
- For E3, each demo video's event list must be written and committed before any report is generated for it.

**Paper edits.**
- The limit is 8 pages. The draft is 6, with the last page filled to the bottom of both columns, and the authors want the last page kept nearly full. After any edit, check the page count and look at the last page.
- `\pending{...}` marks something that needs an author's decision; it renders in red.
- Macros swallow the following space, so write `\NxN{}` before text. Signed macros are already in math mode, so don't wrap them in another `$...$`.
- `refs.bib` was built only from fetched records (Crossref, arXiv), with venue names abbreviated by hand in IEEE style. Fetch and check new references instead of writing them from memory. Cite keys have no underscores. Requests to outside services use a generic User-Agent and no personal details.
- The PDF is published as a GitHub release; publish one only when asked:
  - tag `paper-draft-YYYY-MM-DD`;
  - asset `fewer-frames-beat-smarter-kernels.pdf`;
  - no attribution lines.

  The README links to `releases/latest`, so a new release replaces the link automatically.
