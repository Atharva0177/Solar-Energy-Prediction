---
name: phase-gate-confirmations
description: User requires a summary + explicit confirmation gate after every completed PRD phase before starting the next
metadata:
  type: feedback
---

After each completed phase of the Solar_gemini project, summarize what shipped
(artifacts, measured results, doc updates) and ASK the user for confirmation
before beginning the next phase. Do not roll straight into the next phase.

**Why:** User wants checkpoints to review direction and catch issues early
(requested 2026-08-24 after Phase 2 completed).

**How to apply:** End the turn at a phase boundary with: completion table/summary
+ question "Start Phase N+1?". Exception: if user explicitly says "continue all"
or grants multi-phase permission in the current request. Related: [[project-state]].
