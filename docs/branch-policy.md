# Branch policy

- `master` is release-only and remains empty until an explicit release decision.
- `develop` is the integration branch for active development.
- Feature work branches from `develop` and merges back into `develop` after CI passes.
- Release candidates are promoted from `develop` to `master` explicitly; release tags are created from commits reachable from `master`.

See [releases and container publication](releases.md) for the tag and GHCR policy.
