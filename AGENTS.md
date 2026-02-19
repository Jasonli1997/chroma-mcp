# AGENTS.md

Practical workflow for AI/human contributors in this repo.

## Scope
- Prefer **small, reviewable QoL changes** over large refactors.
- Keep changes narrowly scoped to the issue/plan item.
- If work grows beyond a small PR, split into multiple PRs.

## Local workflow
1. Sync from latest `main`.
2. Create a feature branch using a clear name (examples: `docs/...`, `chore/...`, `fix/...`).
3. Make focused edits tied to one purpose.
4. Run local checks before opening a PR:
   - `uv sync --extra test`
   - `uv run pytest tests/`
   - `uv run ruff check .`
5. Open a PR with concise summary + testing notes.

## Change guidelines
- Avoid changing public behavior unless required.
- Keep docs and implementation in sync.
- Add tests for behavior changes; keep test additions minimal and targeted.
- Prefer readability and maintainability over cleverness.
- Preserve backward compatibility unless the PR explicitly proposes a breaking change.

## PR quality bar
- Clear title (`docs: ...`, `chore: ...`, `fix: ...`).
- Include in PR description:
  - What changed
  - Why it changed
  - How it was tested
  - Any known limitations/blockers
- Keep PR size small when possible (easy to review in one pass).
- If CI/local checks are blocked by environment/tooling, state that explicitly.

## Review + follow-up expectations
- Address review comments in-thread with concrete responses.
- For small requested fixes (docs wording, tiny code tweaks), apply directly and push updates.
- For larger requests (multi-file refactor, design shifts), create a follow-up plan/issue instead of sneaking scope creep into a small PR.

## Current Sprint Focus (Temporary)
- For this sprint only, prioritize quality-of-life improvements: docs polish, dev/test ergonomics, and small reliability fixes.
- This temporary focus does not change the permanent repo guidelines above.
