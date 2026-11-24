# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`GLOSSARY.md`** at the repo root.
- **`docs/adr/`**: read ADRs that touch the area you're about to work in.
- For CI signal-boundary work, read `docs/adr/0001-pr-ci-signal-boundaries.md`.

If either location is absent, **proceed silently**. Don't flag its absence or suggest creating it upfront. The `/domain-modeling` skill creates domain documents lazily when terms or decisions are actually resolved.

## File structure

This is a single-context repository:

```
/
├── GLOSSARY.md
├── docs/adr/
│   └── 0001-pr-ci-signal-boundaries.md
└── firecrown/
```

No `GLOSSARY-MAP.md` or per-package glossary layout is needed.

## Use the glossary's vocabulary

When output names a domain concept, use the term as defined in `GLOSSARY.md`. Don't drift to synonyms that change its meaning.

If a needed concept isn't in the glossary yet, that's a signal: either the terminology doesn't match the project or there's a real gap to note for `/domain-modeling`.

## Flag ADR conflicts

If an output contradicts an existing ADR, surface the conflict explicitly, including the ADR and the reason to reconsider it. Don't silently override a recorded decision.
