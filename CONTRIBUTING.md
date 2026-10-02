# Contributing to Fission

Fission is under active development. Keep changes small enough to review and preserve the separation between inference-core work and repository/deployment tooling.

## Branches

- `develop` is the integration branch.
- Create feature or maintenance branches from `develop`.
- `master` is reserved for explicit releases.
- Merge back to `develop` only after the relevant CI checks pass.

See [the branch policy](docs/branch-policy.md) and [release policy](docs/releases.md).

## Local validation

Use .NET 10 as pinned by `global.json`.

Build the full solution:

```bash
dotnet restore Fission.slnx
dotnet build Fission.slnx -c Release --no-restore
```

Run all sample executables that are registered in `Fission.slnx`:

```powershell
pwsh ./eng/run-projects.ps1 -Root samples -Configuration Release -NoBuild
```

Run every executable spec project under `tests/` that is registered in `Fission.slnx`:

```powershell
pwsh ./eng/run-projects.ps1 -Root tests -Configuration Release -NoBuild
```

The solution file is the project manifest used by CI. Adding a new spec means adding its project to `Fission.slnx`; the main workflow then discovers and executes it automatically. Do not maintain a second hard-coded test-project list in GitHub Actions.

Projects intentionally not registered in `Fission.slnx` are not treated as CI specs.

## Benchmarks

Scheduler hot-path changes should include a local before/after run of:

```bash
dotnet run --project benchmarks/Fission.Scheduler.Benchmarks/Fission.Scheduler.Benchmarks.csproj -c Release -- 20000
```

End-to-end serving work can be measured with:

```bash
dotnet run --project benchmarks/Fission.Serving.LoadGen/Fission.Serving.LoadGen.csproj -c Release -- --help
```

See [scheduler benchmark](docs/scheduler-benchmark.md) and [serving benchmark](docs/serving-benchmark.md). Do not use hosted-runner timing as a merge threshold.

## Container changes

For packaging or serving integration changes:

```bash
docker compose config --quiet
docker build -t fission:dev .
```

The container CI performs health, OpenAI-protocol, and small streaming-load checks. The current image still uses the deterministic backend; do not describe it as production GPU model serving.

## Pull requests

A focused PR should state:

- what behavior or infrastructure changes,
- what is intentionally out of scope,
- which validation was run,
- benchmark evidence when a hot path changes,
- container/release implications when packaging changes.

Avoid mixing unrelated scheduler/runtime changes with repository, CI, documentation, or packaging cleanup unless the changes are inseparable.
