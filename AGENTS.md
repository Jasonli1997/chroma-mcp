# AGENTS.md

Practical workflow for AI/human contributors in this repo.

## Scope
- Prefer **small, reviewable QoL changes** over large refactors.
- Keep changes narrowly scoped to the issue/plan item.

## Local workflow
1. Create a feature branch from `main`.
2. Make focused edits.
3. Run tests before opening a PR:
   - `uv run pytest`
4. Open a PR with a concise summary and testing notes.

## Change guidelines
- Avoid changing public behavior unless required.
- Keep docs and implementation in sync.
- Add tests for behavior changes; keep test additions minimal and targeted.
- Prefer readability and maintainability over cleverness.

## PR quality bar
- Clear title (`docs: ...`, `chore: ...`, `fix: ...`).
- Include:
  - What changed
  - Why it changed
  - How it was tested
- Keep PR size small when possible.

## Tonight's constraint
Treat this pass as a quality-of-life sprint: docs polish, test/dev ergonomics, and small reliability improvements only.
