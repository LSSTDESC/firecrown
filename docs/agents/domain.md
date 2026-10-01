# Domain Docs

## Layout and Consumer Rules

This repo uses the existing single-context layout:

- `CONTEXT.md` at the repository root holds domain terminology.
- `docs/adr/` holds architectural decisions.

Before exploring the codebase, read `CONTEXT.md` and the ADRs relevant
to the work. For CI signal-boundary work, read
`docs/adr/0001-pr-ci-signal-boundaries.md`.

Preserve these locations and their existing contents during setup.
No `CONTEXT-MAP.md` or per-package context layout is needed.

If a domain document is absent, proceed silently rather than proposing
placeholder documentation. `/domain-modeling` creates domain documents
when terminology or decisions are actually resolved.

## Use the Glossary's Vocabulary

Use concepts as defined in `CONTEXT.md` in ticket titles, proposals,
hypotheses, and test names. Avoid substituting synonyms that change
their meaning.

If a needed concept is absent, reconsider the terminology or note the
gap for `/domain-modeling`.

## Flag ADR Conflicts

Explicitly identify any proposal that contradicts an existing ADR,
including the ADR reference and the reason to reconsider it. Do not
silently override a recorded decision.
