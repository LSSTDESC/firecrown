# Triage Labels

Use these canonical roles as the local tracker's label strings.

| Canonical role | Local value | Meaning |
| --- | --- | --- |
| `needs-triage` | `needs-triage` | Maintainer needs to evaluate the issue |
| `needs-info` | `needs-info` | Waiting on the reporter for more information |
| `ready-for-agent` | `ready-for-agent` | Fully specified and ready for an autonomous agent |
| `ready-for-human` | `ready-for-human` | Requires human implementation |
| `wontfix` | `wontfix` | Will not be actioned |

When a skill says to apply a triage label, set the ticket's `Status:`
line to the corresponding local value. No remote labels are created.

Wayfinding lifecycle states are defined separately in
`docs/agents/issue-tracker.md`.
