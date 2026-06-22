#!/bin/bash
# Overnight campaign disk hygiene. SAFE: only scratch/cache, never repo/results/active builds.
# Run each watchdog cycle. Targets: /tmp scratch, old RAM build-variant dirs, pip cache, stale root torch_extensions.
find /tmp -maxdepth 1 -type f \( -name '*.out' -o -name '*.log' \) -mmin +30 -delete 2>/dev/null
find /dev/shm/torch_ext -mindepth 1 -maxdepth 1 -type d -mmin +90 -exec rm -rf {} + 2>/dev/null
rm -rf /root/.cache/pip/* 2>/dev/null
[ -d /root/.cache/torch_extensions ] && find /root/.cache/torch_extensions -mindepth 1 -maxdepth 1 -mmin +90 -exec rm -rf {} + 2>/dev/null
echo "[cleanup $(date -u +%H:%MZ)] /: $(df -h / | awk 'NR==2{print $4" free ("$5")"}')  shm: $(df -h /dev/shm | awk 'NR==2{print $4" free"}')  vol: $(df -h /workspace | awk 'NR==2{print $4" free"}')"
