# Dead-code removal spec — committed NON-SOURCE artifacts + LOC report

AREA: committed non-source artifacts (nvcc intermediate dumps, scan output
dumps, the scan-prep helper, committed Claude session tool-result dumps) and
the recomputed LOC/language report.

Repo: `/workspace/SuperGrok1.5` (READ-ONLY analysis; this is an apply-ready
removal list, not an applied change).

Scope rule honored: the production L3-TC persistent wgmma megakernel, the 33
`_tc` cells, the prebuilt `_ops`, the 3 model layouts, and the gfx942/tpu trees
all STAY. Only PROVABLY-dead (reachability-checked, zero source/build/test
references) committed artifacts are listed for removal. `results/` is FLAGGED
TO KEEP (reference/curated benchmark results) — see §5.

---

## 0. Reachability verification (how each was proven dead)

Searched every `.py .cu .cuh .cpp .c .h .hpp .hip .sh .md .toml .txt .cfg .ini
.in .json .yaml .yml CMakeLists.txt` plus `setup.py`, `pyproject.toml`,
`MANIFEST.in`, `.gitignore`, `.github/**`, and `build.sh` for each artifact
path/name. Exclusions applied: `third_party/`, `.git/`, `__pycache__/`, and the
artifact dirs themselves.

Findings:

- **`_dectc_codegen/`** — the ONLY non-self reference is `_scan_prep.sh:7`
  (which is itself slated for removal here). Zero references from any
  `.py/.cu/.cuh/.cpp` source, from `setup.py`/`MANIFEST.in`/`pyproject.toml`,
  from any CMake, or from CI. Contents are nvcc `--keep` intermediates
  (`.cudafe1.cpp`, `.cpp1.ii`, `.cpp4.ii`, `.ptx`, `.gpu`, `.cubin`,
  `.fatbin`, `.fatbin.c`, `.module_id`, `.o`, `err.log`) plus two scratch
  probes (`sizeprobe.cpp`, `grid.cpp` + their compiled binaries). All are
  regenerable nvcc build droppings for `mega_decoder_real_adamw_tc`.
  No `__init__.py` → not an importable package.

- **`_scan/`** — referenced only by `_scan_prep.sh` (`OUT=/workspace/_scan`,
  which produces this dir) and by self-references inside `_scan/` dumps. The
  apparent matches for the token `_scan` in source (`_selective_scan`,
  `parallel_scan`, `y_scan`, `_scan_csrc_ifndef_macros`, `lax.associative_scan`)
  are unrelated identifiers, NOT references to this directory. Contents are the
  secret/PII scan-prep manifests + grep dumps (`all_files.txt`,
  `text_files.txt`, `text_unique.txt`, `text_hashes.tsv`, `binary_files.txt`,
  `review_candidates.txt`, `rc_chunk_00..29`, `hits_*`, `pat_*`).

- **`_scan_prep.sh` + `_scan_prep.log`** — `_scan_prep.sh` is the standalone
  generator of `_scan/` (a one-shot secrets/PII audit prep script that reads
  `/workspace/...` sibling dirs and writes `/workspace/_scan`). Nothing
  references `_scan_prep` (grep for `scan_prep` returns zero source/build/test
  hits). `_scan_prep.log` is its stdout capture. No other `_scan*` helper
  exists in the tree.

- **`claude_session_archive/`** — committed Claude session tool-result dumps
  (`.jsonl` transcripts, `.meta.json`, `tool-results/*.txt`, workflow `.js`,
  per-project `memory/*.md`). The ONLY non-self reference is `_scan_prep.sh:9`
  (a scan ROOT, itself being removed) and self-references inside `_scan/`
  dumps. Zero source/build/test/CI references. No `__init__.py` → not a
  package. NOTE: one stale path mention exists in
  `claude_session_archive/projects/.../memory/h100-mps-max-parallelism.md`
  pointing at `results/tuning/journal.log.lock` — that is content INSIDE the
  archive (a removed file), not an external dependency on the archive.

Packaging cross-check (so removal cannot break sdist/wheel):
`MANIFEST.in` ships only README/LICENSE/pyproject/setup.py/build.sh,
`csrc/** (*.cpp *.cu *.cuh *.h *.hpp *.hip *.inc)`,
`grokking_optimizers/*.cuh|*.hip.hpp`, `scripts/optimizer_math_manifest.json`,
and `third_party/cutlass/include/**`. `setup.py` uses `package_data` +
`include_package_data` which only reach files under importable packages. None
of the four artifact sets are shipped. Removal does not touch the build.

---

## 1. EXACT rm list (apply-ready)

All paths are git-tracked (committed). Delete the three directories and the two
top-level helper files. This removes exactly 528 tracked files (515 text + 13
binary).

```sh
# Decoder nvcc --keep intermediate dumps + scratch probes (64 tracked files)
git rm -r _dectc_codegen/

# Secret/PII scan-prep output manifests + grep dumps (43 tracked files)
git rm -r _scan/

# The scan-prep generator helper + its log (2 tracked files)
git rm _scan_prep.sh _scan_prep.log

# Committed Claude session tool-result/transcript dumps (419 tracked files)
git rm -r claude_session_archive/
```

Equivalent non-git form:

```sh
rm -rf _dectc_codegen/ _scan/ claude_session_archive/
rm -f  _scan_prep.sh _scan_prep.log
```

Per-set tracked file counts (verified via `git ls-files`):

| path                      | tracked files | wc -l (newlines) |
|---------------------------|--------------:|-----------------:|
| `_dectc_codegen/`         |            64 |        7,954,025 |
| `_scan/`                  |            43 |           91,145 |
| `claude_session_archive/` |           419 |           87,095 |
| `_scan_prep.sh`           |             1 |               86 |
| `_scan_prep.log`          |             1 |               27 |
| **total**                 |       **528** |    **8,132,378** |

(The `wc -l` total counts newlines across all 528 files including the few
binaries; the cloc-style content-line figure used in §3 is 8,089,083 text
lines over the 515 text files, excluding the 13 binaries.)

---

## 2. In-file OLD snippet removals (VERBATIM)

The only in-file reference to any removed artifact lives in `_scan_prep.sh`,
which is itself deleted in §1 — so NO surviving source file needs an in-place
edit. For completeness, the verbatim block in `_scan_prep.sh` that names the
removed dirs (deleted wholesale with the file) is:

OLD (`_scan_prep.sh`, lines 7-10):
```sh
ROOTS=(/workspace/_snap /workspace/tune11_out /workspace/_dectc_codegen \
       /workspace/nvcc_baseline_build /workspace/task11_bench_build \
       /workspace/fanout_patches /workspace/claude_session_archive \
       /workspace/SuperGrok1.5_repo.bundle)
```

OLD (`_scan_prep.sh`, line 3):
```sh
OUT=/workspace/_scan
```

No edits are required to any `.py/.cu/.cuh/.cpp/.h/.hpp/.hip/setup.py/
MANIFEST.in/pyproject.toml/CMake/.gitignore/.github` file. `.gitignore`
already ignores `results/tuning/` and build scratch; it does not mention any of
the four removed artifacts, so no `.gitignore` edit is needed.

---

## 3. Recomputed LOC / language report (cloc-style, by extension)

Scope: git-tracked files only (the committed footprint), excluding
`third_party/`, `.git/`, `__pycache__/` per spec. "lines" = text content lines
(binary files excluded from line counts; counted only as file counts).

### (i) WHOLE REPO AS-IS

total text lines: **8,450,093** — text files: 1,562 — binary files: 688 — all tracked files: 2,250

| ext        |        lines | files |  % lines |
|------------|-------------:|------:|---------:|
| .ii        |    5,649,695 |    10 |  66.86%  |
| .cpp       |    1,747,086 |    21 |  20.68%  |
| .gpu       |      331,496 |     6 |   3.92%  |
| .ptx       |      126,272 |     6 |   1.49%  |
| .log       |      123,916 |   161 |   1.47%  |
| .py        |       91,201 |   178 |   1.08%  |
| .txt       |       84,837 |    52 |   1.00%  |
| .jsonl     |       69,647 |   197 |   0.82%  |
| .c         |       58,072 |    15 |   0.69%  |
| .json      |       31,760 |   235 |   0.38%  |
| .tsv       |       25,257 |     1 |   0.30%  |
| .cuh       |       22,679 |    31 |   0.27%  |
| .md        |       15,690 |    84 |   0.19%  |
| .hpp       |       13,790 |    24 |   0.16%  |
| .d         |       13,038 |     2 |   0.15%  |
| (noext)    |       12,860 |   334 |   0.15%  |
| .diff      |       11,435 |    22 |   0.14%  |
| .patch     |        6,097 |    17 |   0.07%  |
| .cu        |        4,631 |    18 |   0.05%  |
| .h         |        3,309 |    20 |   0.04%  |
| .hip       |        2,309 |    48 |   0.03%  |
| (others)   |        ~4,000|   ~70 |  ~0.05%  |

Binary by ext: `(noext)`=589, `.o`=65, `.png`=10, `.so`=9, `.gz`=4, `.npy`=4,
`.cubin`=3, `.fatbin`=3, `.db`=1.

### (ii) TRUE SOURCE — after removing the confirmed artifacts

total text lines: **361,010** — text files: 1,047 — binary files: 675 — all tracked files: 1,722

| ext        |    lines | files |  % lines |
|------------|---------:|------:|---------:|
| .log       |  123,885 |   155 |  34.32%  |
| .py        |   91,201 |   178 |  25.26%  |
| .json      |   31,573 |    48 |   8.75%  |
| .cuh       |   22,679 |    31 |   6.28%  |
| .md        |   15,360 |    75 |   4.25%  |
| .hpp       |   13,790 |    24 |   3.82%  |
| .d         |   13,038 |     2 |   3.61%  |
| (noext)    |   11,925 |   304 |   3.30%  |
| .diff      |   11,435 |    22 |   3.17%  |
| .patch     |    6,097 |    17 |   1.69%  |
| .cu        |    4,621 |    17 |   1.28%  |
| .h         |    3,309 |    20 |   0.92%  |
| .txt       |    2,943 |    18 |   0.82%  |
| .hip       |    2,309 |    48 |   0.64%  |
| .cpp       |    2,100 |    16 |   0.58%  |
| .yml       |    1,666 |     3 |   0.46%  |
| .sh        |      987 |    14 |   0.27%  |
| .ninja     |      589 |    16 |   0.16%  |
| .toml      |      486 |     5 |   0.13%  |
| .inc       |      441 |     3 |   0.12%  |
| (others)   |     ~825 |   ~30 |  ~0.23%  |

Binary by ext: `(noext)`=587, `.o`=60, `.png`=10, `.so`=9, `.gz`=4, `.npy`=4,
`.db`=1.

NOTE on remaining `.log` (123,885 lines) and `.d`/`.ninja`/`.o`: these are
NON-artifact build/run scratch living under untracked-by-spec but git-committed
dirs (`nvcc_baseline_build/`, `task11_bench_build/`, `build/` deps, `.perf/`,
`fanout_patches/`, `archived_reports/`). They are OUT OF SCOPE for this area
(this area = the 4 named artifact sets only) and are NOT removed here; flag
separately if a deeper LOC cleanup is wanted.

### Delta (removed by this spec)

- text lines removed: **8,089,083**  (95.73% of repo text lines)
- text files removed: **515**
- binary files removed: **13**
- all tracked files removed: **528**
- TRUE source = **4.27%** of current committed repo lines.

The dominant removal is `_dectc_codegen/` (.ii=5,649,695 + .cpp=1,744,986 +
.gpu=331,496 + .ptx=126,272 + .c=58,072 ≈ 7.91M lines, 98% of the deletion).

---

## 4. ARTIFACTS-ONLY breakdown (the 528 removed files)

total text lines: 8,089,083 — text files: 515 — binary files: 13 — all files: 528

| ext        |        lines | files |  % of removed |
|------------|-------------:|------:|--------------:|
| .ii        |    5,649,695 |    10 |       69.84%  |
| .cpp       |    1,744,986 |     5 |       21.57%  |
| .gpu       |      331,496 |     6 |        4.10%  |
| .ptx       |      126,272 |     6 |        1.56%  |
| .txt       |       81,894 |    34 |        1.01%  |
| .jsonl     |       69,602 |   195 |        0.86%  |
| .c         |       58,072 |    15 |        0.72%  |
| .tsv       |       25,257 |     1 |        0.31%  |
| (noext)    |          935 |    30 |        0.01%  |
| .md        |          330 |     9 |        0.00%  |
| .js        |          227 |     6 |        0.00%  |
| .json      |          187 |   187 |        0.00%  |
| .sh        |           86 |     1 |        0.00%  |
| .log       |           31 |     6 |        0.00%  |
| .cu        |           10 |     1 |        0.00%  |
| .module_id |            3 |     3 |        0.00%  |

Binary removed: `.o`=5, `.cubin`=3, `.fatbin`=3, `(noext)`=2.

---

## 5. `results/` — FLAGGED TO KEEP (do NOT remove)

`results/` (101 tracked files: 60 `.log`, 15 `.json`, 10 `.png`, 10 `.md`,
5 `.csv`, 1 `.jsonl`) is CURATED REFERENCE BENCHMARK output and is actively
referenced by source/docs. Evidence it must stay:

- `.gitignore:27-30` explicitly documents `results/h100_grokking_race/` as
  "curated summaries ... deliverables" and ignores only the sub-path
  `results/tuning/` (operational scratch). The rest of `results/` is
  intentionally committed.
- Referenced as reference data by:
  - `README.md:405`, `HARDWARE_VALIDATION.md:156` → `results/h100_grokking_race/`
  - `AUTOTUNE_LINKAGE.md:203` → roofline outputs
  - `wiring_check.py:34`, `.gauntlet_plan.md:22` → `results/h100_grokking_race/wiring_check.json`
  - `tuning/roofline.py:29,223,848`, `tuning/precision_analysis.py:12`,
    `tuning/_mbtc_bypass_profile.py:94`, `tuning/_decoder_validate.py:51`,
    `tuning/tune_optimizers.py:5`, `PHASE1_CAMPAIGN.md:564,591`,
    `grokking_race_v2.py:719` → consumed/produced reference configs + verdicts.

Recommendation: KEEP all of `results/` as-is. (`results/tuning/` is already
gitignored; nothing to do.) If disk pressure later demands trimming, the only
candidates would be `results/h100_grokking_race/archive/*CONTAMINATED*` /
`*prebatchfix*` historical runs — but that is a SEPARATE judgment call for the
user, not a provably-dead removal.

---

## 6. Verification commands (post-apply sanity)

```sh
# 1. Confirm nothing surviving references the removed paths:
grep -rIn "_dectc_codegen\|claude_session_archive\|_scan_prep\|/_scan\b" \
  --include="*.py" --include="*.cu" --include="*.cuh" --include="*.cpp" \
  --include="*.c" --include="*.h" --include="*.hpp" --include="*.hip" \
  --include="*.sh" --include="*.toml" --include="*.in" --include="*.md" \
  . | grep -v "_selective_scan\|parallel_scan\|y_scan\|associative_scan\|_scan_csrc"
# expected: no output

# 2. Build/packaging untouched:
git diff --stat -- setup.py MANIFEST.in pyproject.toml build.sh .gitignore
# expected: no changes

# 3. Confirm exactly 528 files dropped:
git ls-files _dectc_codegen _scan claude_session_archive _scan_prep.sh _scan_prep.log | wc -l
# expected (pre-rm): 528 ; (post-rm): 0
```

The math-drift guard, parity/determinism gates, and `_ops` build are not
touched by any path in §1 — all removed files are regenerable nvcc droppings,
scan dumps, a one-shot audit helper, and session transcripts.
