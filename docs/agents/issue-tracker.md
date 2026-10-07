# Issue Tracker: Local Markdown

Track work locally in this checkout. "Publish to the issue tracker" means
write local Markdown files, not create remote issues or pull requests.

## Conventions

- Use one directory per feature: `.scratch/<feature-slug>/`.
- Write one file per ticket at
  `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`.
  Never combine all tickets into one file.
- Record triage state in a `Status:` line near the top of each ticket,
  using the vocabulary in `docs/agents/triage-labels.md`.
- Record dependencies in a `Blocked by:` line using ticket numbers within
  the same feature; use `none` for tickets without dependencies.
- Append discussion under a `## Comments` heading.
- Fetch a ticket by reading its referenced path. Resolve a bare ticket
  number within its feature directory; ask if the feature is ambiguous.

## Specifications

The authoritative CI specification is
`docs/pr-ci-signal-boundaries-spec.md`. Read and link to that file;
do not copy it into `.scratch/` or create a second authoritative version.

Use `.scratch/pr-ci-signal-boundaries/issues/` for tickets derived from
that specification.

For other features, preserve any existing authoritative specification
location. If a new specification is requested and no location exists,
use `.scratch/<feature-slug>/spec.md`.

## Ticket Planning Approval

Before writing tickets, present the proposed breakdown and dependencies
for approval. Write only the approved tickets locally. Creating tickets
does not authorize implementation.

## Wayfinding Operations

For `/wayfinder`, use `.scratch/<effort>/map.md` for the map and
`.scratch/<effort>/issues/<NN>-<slug>.md` for child tickets.

- Record the question in the child ticket body and its type in `Type:`
  (`research`, `prototype`, `grilling`, or `task`).
- Wayfinding uses lifecycle states in `Status:`: `open`, `claimed`,
  and `resolved`. These are distinct from triage roles.
- A ticket is unblocked when every ticket named in `Blocked by:` is
  `resolved`.
- Select the lowest-numbered open, unblocked ticket as the frontier.
- Claim a ticket by saving `Status: claimed` before work.
- Resolve a ticket by appending `## Answer`, setting `Status: resolved`,
  and adding a gist and link to the map's Decisions-so-far section.
