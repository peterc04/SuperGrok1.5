#!/usr/bin/env bash
set -uo pipefail
SRC=/workspace/SuperGrok1.5
SNAP=/workspace/_snap
VENV=/workspace/venv_selfcontained
BRANCH=claude/custom-optimizer-analysis-HFYhg
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "clean any prior build dir"
rm -rf "$SNAP"; mkdir -p "$SNAP"

log "copy full working tree (including all .git; will strip next)"
cp -a "$SRC/." "$SNAP/" || { log "COPY FAILED"; exit 1; }

log "strip every nested .git (main + cutlass submodule + 36 worktrees)"
find "$SNAP" -name .git -prune -exec rm -rf {} +
log "remove stale .gitmodules (cutlass is now vendored as plain files)"
rm -f "$SNAP/.gitmodules"

log "capture python deps -> requirements-lock.txt"
"$VENV/bin/pip" freeze > "$SNAP/requirements-lock.txt" 2>/dev/null && \
  log "  froze $(wc -l < "$SNAP/requirements-lock.txt") packages" || \
  log "  (pip freeze failed; placeholder written)"

log "write ENVIRONMENT.md (rebuild recipe)"
PYV=$("$VENV/bin/python" --version 2>&1)
NVCC=$(nvcc --version 2>/dev/null | tail -1 || echo "nvcc not on PATH at snapshot time")
GCC=$(gcc --version 2>/dev/null | head -1 || echo "gcc n/a")
cat > "$SNAP/ENVIRONMENT.md" <<EOF
# Environment & Rebuild Guide

This repo is a **self-contained snapshot**: source + build cache + compiled
binaries + a vendored CUTLASS tree. A plain \`git clone\` gives you everything
except the Python virtualenv (which is large and machine-specific). Use the
steps below to recreate the venv.

## Toolchain captured at snapshot time
- Python: \`${PYV}\`
- CUDA:   \`${NVCC}\`
- GCC:    \`${GCC}\`
- Target GPU: NVIDIA H100 (\`sm_90\`)

## Recreate the Python environment
\`\`\`bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-lock.txt   # exact pinned versions
\`\`\`

## Build the extension
\`\`\`bash
./build.sh            # see BUILD_AND_VALIDATE.md for full options
# (CUTLASS is already vendored under third_party/cutlass — no submodule init needed)
\`\`\`

## Notes
- \`.build_cache/\` (sccache) and \`build/\` are included so rebuilds are fast on a
  matching toolchain; point sccache at \`.build_cache\` if needed.
- \`requirements-lock.txt\` is a pinned \`pip freeze\`; \`workspace_docs/env_requirements.txt\`
  is the original looser requirement notes.
- See \`BUILD_AND_VALIDATE.md\` and \`CODEBASE_EXPLAINED.md\` for the full picture.
EOF

log "copy /workspace planning docs -> workspace_docs/"
mkdir -p "$SNAP/workspace_docs"
n=0
for f in /workspace/.*.md /workspace/.*.txt; do
  [ -f "$f" ] || continue
  b=$(basename "$f"); b=${b#.}
  cp -a "$f" "$SNAP/workspace_docs/$b" && n=$((n+1))
done
log "  copied $n planning docs"

log "git init + single snapshot commit"
cd "$SNAP" || exit 1
git init -q
git config user.name "Peter Coy"
git config user.email "<owner-email>"
git add -A -f
log "  staged $(git diff --cached --name-only | wc -l) files; committing..."
git commit -q -m "snapshot: full self-contained environment (source + build cache + binaries + vendored CUTLASS + env recipe)

Clone-able snapshot of the SuperGrok1.5 H100 megakernel campaign: all source,
the build cache and compiled binaries, the vendored CUTLASS tree, the agent
worktrees, plus requirements-lock.txt and ENVIRONMENT.md so the environment
can be rebuilt from a clone.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_REDACTED"
git branch -m "$BRANCH"

echo
log "=== VERIFY ==="
echo "tracked files:      $(git ls-files | wc -l)"
echo "gitlinks (want 0):  $(git ls-files -s | awk '$1==160000' | wc -l)"
echo "cutlass files:      $(git ls-files third_party/cutlass | wc -l)"
echo "worktree files:     $(git ls-files .claude/worktrees | wc -l)"
echo "env recipe present: $(git ls-files | grep -cE '^(requirements-lock\.txt|ENVIRONMENT\.md)$')/2"
echo "planning docs:      $(git ls-files workspace_docs | wc -l)"
echo ".git object size:   $(du -sh "$SNAP/.git" | cut -f1)"
echo "branch:             $(git rev-parse --abbrev-ref HEAD)"
log "BUILD COMPLETE (nothing pushed yet)"
