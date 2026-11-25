# Part B paper (Parallel Programming)

IEEE conference format (`IEEEtran.cls` v1.8b, conference mode, US Letter), 8 pages. Scope, decided
2026-11-25: the floor paper (kernel ladder, where the time goes, memory wall) plus a measured section on why
temporal gating does not pay for this captioning pipeline (PLAN.md section 3.3).

Status: **outline**. The red `[pending: ...]` notes in `main.tex` are the outline; prose waits on the
literature search (PLAN.md section 3.2 and the week-1 checklist).

## Build

```bash
.venv/bin/python -I paper/numbers.py          # generated/numbers.tex and generated/table_ladder.tex
cd paper && latexmk -pdf -outdir=build main.tex
```

`build/main.pdf` is the output (`build/` is git-ignored). `generated/` is written by `numbers.py` from the
committed result files: do not edit it by hand, and never type into `main.tex` a number that a result file
holds. Figures will be redrawn at paper width (3.5 in for one column, 7.16 in for two) into `figures/`
by a script; the current figures in `plots/` are 11-12 in wide and are placeholders here.

Format decisions carried over from the ICEE 2026 paper (neuroflux): `microtype` with `protrusion=false`
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
| Pixel-level redundancy (X0) | `results/x0/era_tra/summary.json` (`bench/x0_era.py`) |

## References

`refs.bib` was built on 2026-11-25 from fetched records (Crossref by DOI, the arXiv API by id), not typed from
memory. Each entry's comment names its record. Entries cited as arXiv preprints had no published version in the
sources searched; recheck them before submission. `docs/related_work.md` describes the search and gives each
cited statement's evidence level. Before submission an author must open every cited work and confirm the
statement it supports.

## Data

No ERA video frames and no CapERA reference captions go into the paper or this directory (DATA.md): ERA is
for academic use without redistribution, and CapERA states no licence.
