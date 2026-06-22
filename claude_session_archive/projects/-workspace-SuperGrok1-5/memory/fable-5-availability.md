---
name: fable-5-availability
description: Claude Fable 5 (released 2026-06-09) — account HAS access; this session's harness can't run it for subagents (thinking-config bug); owner fallback = Opus 4.8 max effort
metadata:
  type: reference
---

Owner wants agents on **Claude Fable 5** (`claude-fable-5`, Anthropic's public
Mythos-class model, released 2026-06-09; aliases `fable`/`best`; CC ≥ v2.1.170;
$10/$50 per Mtok — included on Pro/Max/Team until 2026-06-22; adaptive-thinking
ONLY; safety classifiers reroute cyber/bio prompts to Opus 4.8).

**Empirical findings in this session (CC exactly 2.1.170):**
- `CLAUDE_CODE_SUBAGENT_MODEL` in `.claude/settings.local.json` env hot-applies
  to new agent launches. With `fable` → "model may not exist"; with
  `claude-fable-5` → API 400 `thinking.type.disabled not supported` ⇒ the
  account HAS Fable access but this build's subagent spawner sends a broken
  thinking config to adaptive-only models on the override/inherit paths
  (same bug hit inherited-model launches earlier).
- TRAP: removing the env key does NOT un-apply (host caches it; removal broke
  even plain opus launches). Fix = overwrite with a sane value. It is now
  PINNED to `"opus"` — leave it pinned; do not delete the key.
- Mid-session-created `.claude/agents/*.md` types do NOT register (registry
  loads at session start).

**How to apply:** subagents run `model: "opus"` (Opus 4.8, max effort) until
Fable works — owner-approved fallback. To get Fable: owner runs `claude update`
then restarts the session with `--model fable` (or `/model fable`); subagents
stay pinned to opus regardless, so the spawner bug doesn't bite. Re-test the
override after any CC update.
