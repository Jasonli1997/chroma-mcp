# PLAN.md — chroma-mcp QoL Sprint

Status: ACTIVE (Tonight)

## Goal
Ship small, low-risk quality-of-life improvements after a quick repo familiarization pass.

## Quick familiarization snapshot
- Package: `chroma-mcp` (Python, Hatchling)
- Runtime entrypoint: `chroma-mcp = chroma_mcp:main`
- Main implementation: `src/chroma_mcp/server.py`
- Test surface: `tests/test_server.py` (single file currently)

## Track A — Repo onboarding + guardrails
- [x] Clone repo locally under `research/chroma-mcp/repo`
- [x] Confirm write access and branch workflow readiness
- [x] Add `AGENTS.md` to document contributor/agent workflow

## Track B — Developer quality-of-life
- [ ] Run tests locally and capture baseline
- [ ] Add/verify a `make test` or equivalent one-liner workflow
- [ ] Add short “local dev quickstart” section if gaps found

## Track C — CI/test confidence
- [ ] Review GitHub Actions for test coverage on PRs
- [ ] Add lightweight checks only if missing (lint/tests)

## Track D — Small docs polish
- [ ] Tighten README around common setup pitfalls
- [ ] Ensure env var/docs references match current code behavior

## Track E — Wrap-up
- [ ] Open PR(s) with concise change summaries
- [ ] Keep scope to QoL only (no deep refactors tonight)
