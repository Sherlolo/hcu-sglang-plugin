# Migration Contract

## Required Ledger

Create one row for every fork delta that affects runtime, build, packaging, tests, or invocation.

| Field | Meaning |
|---|---|
| ID | Stable migration item ID |
| Fork commit | Source commit or `WORKTREE` |
| Fork path | Source file path |
| Symbols | Functions, methods, classes, or constants |
| Behavior | User-visible or runtime behavior added by the fork |
| Official status | Missing, partial, equivalent, or superseded |
| Plugin status | Missing, inactive, partial, or active |
| Disposition | One classification from the Skill workflow |
| Plugin target | Hook target or plugin-owned module |
| Hook type | BEFORE, AFTER, AROUND, REPLACE, module patch, or N/A |
| Dependencies | LightOp, AITER, HCU runtime, kernel wheel, environment, etc. |
| Static evidence | Tests, signature checks, target resolution, source parity |
| Runtime evidence | Model, command, hardware, result, logs, metrics |
| Status | Pending, implemented, verified, blocked, or excluded |
| Risk | Compatibility and regression notes |

## Classification Rules

- Do not mark “already upstream” based on matching names alone; compare behavior and tests.
- Do not mark “already covered by plugin” unless the entry point reaches the Hook.
- Do not mark “verified” when only registration was tested.
- Treat uncommitted fork changes as first-class `WORKTREE` migration items.
- Keep native extension and packaging deltas visible even when Python Hooks are complete.
- Record an explicit upstream integration gap when the plugin framework cannot safely reproduce behavior.

## Required Evidence by Status

### Implemented

- source exists in the plugin;
- top-level registration reaches it;
- target path resolves against pinned official SGLang;
- unit tests cover optimized and fallback paths.

### Verified

- all “implemented” evidence;
- applicable integration tests pass;
- runtime command and logs demonstrate the intended branch;
- correctness comparison passes;
- performance evidence exists for performance claims.

### Excluded

- source-backed reason;
- proof that behavior is obsolete, upstream-equivalent, or intentionally out of scope;
- named owner or acceptance criterion when approval is required.

## Completion Metrics

Report at least:

- total ledger items;
- items by disposition;
- items by status;
- active Hook target count;
- unresolved target count;
- static test pass/fail count;
- runtime lane pass/pending count;
- native/package gaps remaining.
