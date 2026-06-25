# Transferring this session/work to the 8×H100 instance

## The key fact: the WORK is already portable; the SESSION process is not
- `/workspace` is a RunPod **network volume** (`<runpod-volume-endpoint>`, 311T, fuse). Mount it on the
  8×H100 pod and ALL project state is instantly there: the repo, git history, deliverables, every
  `/workspace/_analysis/` report, and the `.session_memory/` backup.
- A Claude Code **session** (this conversation) is a JSONL transcript on LOCAL disk
  (`~/.claude/projects/-/b9e57703-…jsonl`, ~1.7 MB) — it is DELETED on instance closure and does NOT move with
  the volume. Background tasks (the running analysis workflow) are OS processes tied to THIS pod and will NOT
  survive the move (resume does not re-establish them) — but their OUTPUTS are already on the volume.

## ⚠️ Hard requirement: mount at the SAME path
The session "project key" is derived from the working directory. This session's project slug is `-` (cwd `/`).
On the 8×H100, mount the volume at **`/workspace`** (same as here) and resume from the **same cwd (`/`)**, or the
project key won't match and `--resume` fails with "No conversation found".

## Option A (most robust — the project's intended pattern): durable-state handoff
The repo is built for instance-closure recovery. Don't rely on fragile process migration; let the new instance
read the state I've persisted.
1. On the 8×H100, mount the volume at `/workspace`.
2. Run the env-restore from `SuperGrok1.5/RESUME.md` §1:
   `pip install nvidia-nvshmem-cu12 optuna ruff nvidia-ml-py`; `git config …`; the NVSHMEM env vars;
   `cp .session_memory/*.md /root/.claude/projects/-/memory/`.
3. Start Claude Code and have it read `/workspace/_analysis/MASTER_STATE.md` (+ `RESUME.md`, `PROGRESS.md`).
   It resumes with full context — this is more reliable than a transcript copy and is how the prior teleports worked.

## Option B (resume THIS exact conversation): stage the transcript
`/teleport` is a CUSTOM command (not a built-in) that automates this; the manual equivalent:
1. Stage this session's transcript to the volume (I can do this for you — see below) BEFORE closing this pod.
2. On the 8×H100: copy it to `~/.claude/projects/-/b9e57703-…jsonl` (the `-` project slug, matching cwd `/`).
3. `claude --resume b9e57703-6aee-4a20-9e04-bc7623783b7d` run from cwd `/`.
   (Subagent/workflow transcripts under the session's `subagents/` dir are optional — only needed to re-inspect
   past agent runs; the main conversation resumes without them.)
Caveats per the Claude Code docs: MCP servers + auth re-init on the new pod (re-login), permission grants
re-prompt, in-flight tasks are gone.

## Why move at all: the 8 GPUs are REQUIRED for the next step
The #1 resume item — the real one-1.5B-model-across-8-GPUs TP training run (apply
`phase6/tp_datapath_fix_WIP.patch` → decoder gate → `tuning/_tp8_build.sh` →
`torchrun --nproc_per_node=8 tuning/_tp8_run.py` → close bug C with compute-sanitizer) — **cannot run on 1×H100.**

## Recommended sequence
1. Let the running analysis workflow finish (persists the final `MASTER_STATE.md` to the volume).
2. Stage this transcript to the volume (safety net).
3. Bring up the 8×H100, mount `/workspace`, run RESUME.md §1.
4. Resume via Option A (read MASTER_STATE) or Option B (`--resume` the staged transcript) — then start the 8-GPU run.
