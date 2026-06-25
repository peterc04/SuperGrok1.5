---
name: supergrok-autonomy
description: "On SuperGrok2, never pause to ask the user which task to prioritize — proceed autonomously on everything in parallel"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 6354dc07-b50f-40a0-8748-5189102539d3
---

The user denied an AskUserQuestion that asked them to pick the next focus among (TP flagship / TMA /
ViT-Mamba / datasets) at a milestone. They have repeatedly signalled: "continue with everything", "use as
many workflows as you can", "I don't want to be claude-bound, I want the hardware to be the constraint."

**Why:** they want maximal throughput; being asked to choose serializes on them and wastes their time. They
will course-correct if I pick wrong — that's cheaper to them than a blocking question.

**How to apply:** do NOT use AskUserQuestion to choose task priority or "what next" at milestones. Just
proceed — drive the north-star critical path as lead AND fan out the independent tracks as parallel
workflows (read-only spec producers to avoid edit conflicts, then apply serially). Only ask when genuinely
blocked on a decision only they can make (e.g. an external/destructive action), not for prioritization.
Report progress + ETAs, but keep moving. See [[supergrok-working-prefs]].
