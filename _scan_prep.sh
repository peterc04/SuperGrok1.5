#!/usr/bin/env bash
set -uo pipefail
OUT=/workspace/_scan
rm -rf "$OUT"; mkdir -p "$OUT"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

ROOTS=(/workspace/_snap /workspace/tune11_out /workspace/_dectc_codegen \
       /workspace/nvcc_baseline_build /workspace/task11_bench_build \
       /workspace/fanout_patches /workspace/claude_session_archive \
       /workspace/SuperGrok1.5_repo.bundle)

log "enumerate files (excluding snapshot .git)"
find "${ROOTS[@]}" -path '/workspace/_snap/.git' -prune -o -type f -print 2>/dev/null > "$OUT/all_files.txt"
log "  total files: $(wc -l < "$OUT/all_files.txt")"

log "classify text vs binary"
: > "$OUT/text_files.txt"; : > "$OUT/binary_files.txt"
while IFS= read -r f; do
  if grep -Iq . "$f" 2>/dev/null; then echo "$f" >> "$OUT/text_files.txt"; else echo "$f" >> "$OUT/binary_files.txt"; fi
done < "$OUT/all_files.txt"
log "  text: $(wc -l < "$OUT/text_files.txt")  binary: $(wc -l < "$OUT/binary_files.txt")"

log "dedup text by content hash (collapses worktree duplicates)"
while IFS= read -r f; do printf '%s\t%s\n' "$(sha256sum "$f" 2>/dev/null | cut -c1-16)" "$f"; done < "$OUT/text_files.txt" > "$OUT/text_hashes.tsv"
sort -u -k1,1 "$OUT/text_hashes.tsv" | cut -f2- > "$OUT/text_unique.txt"
log "  unique text files: $(wc -l < "$OUT/text_unique.txt") (of $(wc -l < "$OUT/text_files.txt"))"

# ---- pattern files ----
cat > "$OUT/pat_secrets.txt" <<'PAT'
gh[oprsu]_[0-9A-Za-z]{36}
github_pat_[0-9A-Za-z_]{30,}
sk-ant-[0-9A-Za-z_-]{20,}
sk-[A-Za-z0-9]{20,}
AKIA[0-9A-Z]{16}
ASIA[0-9A-Z]{16}
AIza[0-9A-Za-z_-]{35}
GOCSPX-[0-9A-Za-z_-]{20,}
hf_[0-9A-Za-z]{30,}
xox[baprs]-[0-9A-Za-z-]{10,}
sk_live_[0-9A-Za-z]{20,}
pk_live_[0-9A-Za-z]{20,}
eyJ[0-9A-Za-z_-]{10,}\.[0-9A-Za-z_-]{10,}\.[0-9A-Za-z_-]{10,}
-----BEGIN [A-Z ]*PRIVATE KEY-----
ssh-rsa AAAA[0-9A-Za-z+/]{50,}
ssh-ed25519 AAAA[0-9A-Za-z+/]{20,}
[a-zA-Z][a-zA-Z0-9+.-]*://[^/[:space:]:@]+:[^/[:space:]:@]+@
PAT
cat > "$OUT/pat_pii.txt" <<'PAT'
[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}
pcoy400
Peter Coy
[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}
\b(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)[0-9]{1,3}\.[0-9]{1,3}
([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}
runpodfs|podvolumes
PAT
cat > "$OUT/pat_kw.txt" <<'PAT'
(pass(word|wd)|secret|api[_-]?key|access[_-]?key|client[_-]?secret|auth[_-]?token|bearer |private[_-]?key|credential)["' ]{0,3}[:=]
PAT

log "mega-grep: secrets"
xargs -d'\n' grep -EnHI -f "$OUT/pat_secrets.txt" < "$OUT/text_unique.txt" > "$OUT/hits_secrets.txt" 2>/dev/null; log "  secret hits: $(wc -l < "$OUT/hits_secrets.txt")"
log "mega-grep: PII"
xargs -d'\n' grep -EnHI -f "$OUT/pat_pii.txt" < "$OUT/text_unique.txt" > "$OUT/hits_pii.txt" 2>/dev/null; log "  pii hits: $(wc -l < "$OUT/hits_pii.txt")"
log "mega-grep: keywords (case-insensitive)"
xargs -d'\n' grep -EnHiI -f "$OUT/pat_kw.txt" < "$OUT/text_unique.txt" > "$OUT/hits_kw.txt" 2>/dev/null; log "  keyword hits: $(wc -l < "$OUT/hits_kw.txt")"

log "strings-scan binaries for structured secrets/paths"
: > "$OUT/hits_binary.txt"
while IFS= read -r f; do
  case "$f" in *.png|*.jpg|*.jpeg|*.gif|*.pt|*.pth|*.npz|*.npy|*.idx*-ubyte|*-ubyte) continue;; esac
  strings -n 8 "$f" 2>/dev/null | grep -En -f "$OUT/pat_secrets.txt" 2>/dev/null | sed "s|^|$f:|" >> "$OUT/hits_binary.txt"
done < "$OUT/binary_files.txt"
log "  binary hits: $(wc -l < "$OUT/hits_binary.txt")"

# hits per area (for chunking the workflow)
log "=== hits by top-level area ==="
cat "$OUT/hits_secrets.txt" "$OUT/hits_pii.txt" "$OUT/hits_kw.txt" 2>/dev/null \
  | sed -E 's|^/workspace/_snap/([^/]+).*|_snap/\1|; s|^/workspace/([^/]+).*|\1|' \
  | cut -d: -f1 | sort | uniq -c | sort -rn | head -40

# human-authored files worth full LLM read (docs/notes/configs/source, small, unique)
grep -EiI '\.(md|txt|rst|toml|yaml|yml|cfg|ini|json|env|sh|py|cu|cuh|h|hpp|cpp|c)$' "$OUT/text_unique.txt" \
  | grep -vE '/third_party/cutlass/' > "$OUT/review_candidates.txt" 2>/dev/null
log "human-authored review candidates (excl cutlass): $(wc -l < "$OUT/review_candidates.txt")"
log "PREP DONE -> manifests in $OUT"
