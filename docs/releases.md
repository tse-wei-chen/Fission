# Releases and container publication

Fission keeps active development on `develop` and reserves `master` for explicit releases.

The repository does not publish an image for every development commit. Container publication is handled by `.github/workflows/image.yml` and is deliberately separate from normal build/container CI.

## Pull requests

Changes that affect the image run the image workflow in build-only mode.

Pull-request runs:

- build the Dockerfile with Buildx,
- validate the same packaging path used for publication,
- do not authenticate to GHCR,
- do not push an image.

The existing `container.yml` workflow remains responsible for starting the built server and exercising health/OpenAI-serving behavior.

## Manual snapshot images

The image workflow can be started manually from GitHub Actions.

The required `image_tag` input becomes the GHCR tag, for example:

```text
edge
scheduler-experiment
cuda-prep
```

A manual run publishes:

```text
ghcr.io/tse-wei-chen/fission:<image_tag>
```

Manual publication never updates `latest`.

Use manual snapshots for deployment/integration testing, not as release identifiers.

## Release images

A release tag must use:

```text
vMAJOR.MINOR.PATCH
```

or a prerelease suffix such as:

```text
v0.1.0-rc.1
```

Tag-triggered publication verifies that the tagged commit is reachable from `master`. This prevents a development-only commit from accidentally becoming a release image.

A stable tag such as `v0.1.0` publishes both:

```text
ghcr.io/tse-wei-chen/fission:v0.1.0
ghcr.io/tse-wei-chen/fission:latest
```

A prerelease tag publishes only its exact version tag and does not move `latest`.

Published images include OCI source/revision/version labels plus BuildKit provenance and SBOM attestations.

## First-release flow

Because `master` is intentionally release-only, the first release should be explicit:

1. choose a tested `develop` commit,
2. open a release PR from `develop` to `master`,
3. require the normal build/container checks to pass,
4. merge the release PR,
5. tag that `master` commit with `vMAJOR.MINOR.PATCH`,
6. let the image workflow publish the exact release tag and `latest`.

For later releases, repeat the same promotion path. Do not force-move or reuse published release tags.

## Current image capability

The current `Fission.Server` container still uses the deterministic backend/token codec. GHCR publication makes the packaging reproducible; it does not change the backend into production GPU model serving.

A CUDA/ONNX Runtime image should be introduced only after real backend/model selection is exposed by the serving composition root.
