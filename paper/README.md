# Paper: *Fewer Frames Beat Smarter Kernels*

*Fewer Frames Beat Smarter Kernels: Attention, Clocks and Temporal Redundancy on a Laptop GPU.* IEEE conference
format (`IEEEtran.cls` v1.8b, conference mode, US Letter), 8-page limit. Scope, decided 2026-11-25: the floor
paper (kernel ladder, where the time goes, memory wall) plus a measured section on why temporal gating does not
pay for this captioning pipeline.

Status: **full draft**, six pages; every citation and the author details checked by the authors (2026-11-27).
`main.tex` marks anything still needing an author's decision with a red `[pending: ...]` note; none are left.

## Build

```bash
.venv/bin/python -I paper/make_numbers.py          # generated/numbers.tex and generated/table_ladder.tex
.venv/bin/python paper/figures.py                  # figures/*.pdf, and *.png for the repository README
cd paper && latexmk -pdf -outdir=build main.tex
```

`build/main.pdf` is the output (`build/` is git-ignored). `generated/` is written by `make_numbers.py` from the
committed result files: do not edit it by hand, and never type into `main.tex` a number that a result file
holds. `figures.py` draws the figures at paper width (3.5 in for one column, 7.16 in for two) as vector PDFs in
`figures/`, in Liberation Serif (TrueType, Times metrics; OpenType Times clones embed with a CFF mismatch that PDF
checkers flag). Do not edit `figures/` by hand.

Format decisions: `microtype` with `protrusion=false`
(protrusion pushed punctuation into the column gutter, which EDAS measures), and the IEEEtran "et al." control
entry for more than six authors.

## Where the numbers come from

Every macro in `generated/numbers.tex` names its source file in a comment. By section:

| Paper | Source |
|---|---|
| Measured limits (Setup) | `results/cuda/peak.json` (`kernels/microbench`) |
| Predictions written before any attention kernel | `results/cuda/predicted_times.json` (`bench/predict_times.py`, 2026-11-23) |
| Table I, pinned clock and rescaled predictions | `results/attention/clock1395/summary.json` (`bench/clock_pinned.py`) |
| Table I, as deployed | `results/attention/v1.json`, `v2.json`, `v3.json`, `b1.json` |
| Scaling and the memory wall (X1) | the same files, groups `x1_n*` (`plots/x1_plot.py`) |
| Where the time goes (X2) | `results/attention/x2_breakdown.json` (`bench/x2_breakdown.py`), `x2.json`, `x2_ncu_*.txt` |
| Token traces | `results/tokens/era_tra/summary.json` (`bench/tokens_era.py`) |
| Caption-level check | `results/gate_captions/scores.json` (`eval/gate_captions.py`) |
| Caption check, embedding similarity (post hoc) | `results/gate_captions/embed_scores.json` (`eval/caption_embed.py`) |
| Pixel-level redundancy (X0) | `results/x0/era_tra/summary.json` (`bench/x0_era.py`) |
| Where a caption's time goes | `results/e1/encoder_share.json` (`eval/encoder_share.py`) |
| The encoder's GPU time by kernel kind (bf16) | `results/e1/encoder_profile.json` (`eval/encoder_profile.py`) |
| PyTorch attention in bf16 | `results/attention/b1_bf16.json` (`bench/b1_sdpa.py --dtype bf16`) |
| PyTorch's FP32 kernels (tensor cores, accuracy) | `results/attention/b1_kernels.json`, `b1_efficient_ncu.txt` (`bench/b1_kernels.py`), `b1.json` |
| Correctness suite | `results/attention/attn_test.txt` (`kernels/attention/attn_test.cu`) |
| V1's profile (L1 saturation) | `results/attention/v1_ncu.txt` |
| Repeat runs | `results/attention/{v1,v2,v3,b1}_repeat.json` |
| X2 at the pinned clock (Fig. 2, floors and fixed shares) | `paper/derived.py` from `clock1395/`, `x2.json`, `x2_breakdown.json` |
| Figures | `paper/figures.py` writes `figures/*.pdf` (and PNG copies for the README) from the files above |

## References

`refs.bib` was built on 2026-11-25 from fetched records (Crossref by DOI, the arXiv API by id), not typed from
memory. Each entry's comment names its record. Entries cited as arXiv preprints had no published version in the
sources searched; recheck them before submission. `docs/related_work.md` describes the search and gives each
cited statement's evidence level. Before submission an author must open every cited work and confirm the
statement it supports.

## Data

No ERA video frames and no CapERA reference captions go into the paper or this directory (DATA.md): ERA is
for academic use without redistribution, and CapERA states no licence.
