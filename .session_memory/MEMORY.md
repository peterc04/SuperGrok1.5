# Memory index

- [SuperGrok working prefs](supergrok-working-prefs.md) — L3-TC-only, prebuilt binaries, caching, exhaustive reading, parallel agents, min GPU hours
- [ncu blocked on RunPod](ncu-blocked-runpod.md) — perf counters denied in-container; needs pod relaunch with CAP_SYS_ADMIN
- [NVSHMEM installed](nvshmem-installed.md) — 3.7.0 w/ sm_90 device bitcode; in-kernel device all-reduce buildable (-DSG_HAS_NVSHMEM=1)
- [Flagship distributed config](flagship-distributed-config.md) — TP8·ZeRO-3 saturates 8 H100s + fits all 11 opts (Nmax/8 shrinks SG2 scratch 509GB→58GB)
- [SuperGrok autonomy](supergrok-autonomy.md) — don't ask priority/what-next questions; proceed autonomously + parallel, user course-corrects
- [Adaptive parallelism + specialization](supergrok-adaptive-parallelism.md) — 3D–5D auto-selected from front-end (5th=expert parallelism); kernels self-specialize by size/config via if-constexpr templating
- [Front-end API](supergrok-frontend-api.md) — PyTorch-shaped: 3 models (any size) + 11 opts fixed, but datasets are PLUGGABLE (bring-your-own); backend self-specializes + compiles
- [Queued deliverables](supergrok-queued-deliverables.md) — post-build: 33-cell flagship roofline graph (ncu-free) + line-by-line dead-code cleanup + LOC/language report
- [ViT TC Fork-B already ported](vit-tc-forkb-already-ported.md) — task #31 premise stale: ViT TC megakernel has NO nCTA*total (only gated scalar does); only byte-id win is DW_SPLITK 4→1 (−25.5GB); flagship 80GB blocker is the acts buffer (~379GB), not grad partials
