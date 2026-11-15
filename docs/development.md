# Development

[Documentation](README.md) · [Contributing](../CONTRIBUTING.md)

All project tasks run through the root [.mise.toml](../.mise.toml).
The contributor environment requires a POSIX-compatible shell.
There is no Makefile or root Go module.

## Install the project tools

Install [mise](https://mise.jdx.dev/).
From the local repository root, run:

```sh
mise install --locked
mise run test
mise run check
```

| Tool | Configured version | Purpose |
| --- | --- | --- |
| Go | 1.27.3 | Compiler, tests, vet, module operations. |
| golangci-lint | 2.15.0 | Static analysis, gofumpt, goimports, whitespace checks. |
| goimports | Resolved version in mise.lock. | Standalone formatting tool. |
| govulncheck | 1.9.0 | Reachable-vulnerability analysis with Go 1.27 support. |
| Nancy | 1.2.0 through the original module path. | Optional Sonatype dependency scan. |
| mockgen | Resolved version in mise.lock. | Available for justified future mock generation. |

The lockfile pins tool resolutions for macOS and Linux on amd64 and arm64.
Keep generated dependency-lock sidecars when mise creates them.
Runtime module dependencies remain separate from developer tools.

Personal global mise tools are outside this repository's lockfile.
If those tools interfere with locked installation, target the project tools explicitly:

```sh
mise install --locked go golangci-lint \
  go:golang.org/x/tools/cmd/goimports \
  go:golang.org/x/vuln/cmd/govulncheck \
  go:github.com/sonatype-nexus-community/nancy \
  go:go.uber.org/mock/mockgen
```

## Choose a task

Scoped tasks use `action:target` names.
Aggregate tasks run the three ordinary module tasks.

| Purpose | Aggregate | Module example |
| --- | --- | --- |
| Build | `mise run build` | `mise run build:tensor` |
| Unit tests | `mise run test` | `mise run test:nn` |
| Lint | `mise run lint` | `mise run lint:simdops` |
| Benchmarks | `mise run bench` | `mise run bench:tensor` |
| Format and fix whitespace | `mise run fmt` | `mise run fmt:nn` |

Test does not invoke every task beginning with `test:`.
Specialized checks are separate because they have different costs or prerequisites.

| Task | What it does |
| --- | --- |
| `fmt:check` | Checks the configured formatters without changing files. |
| `check` | Runs format checks, build, vet, lint, and ordinary tests. |
| `ci` | Adds race tests and govulncheck to check. |
| `test:race` | Runs the Go race detector across all modules. |
| `test:simd` | Tests every module with `GOEXPERIMENT=simd`. |
| `test:coverage` | Reports package statement coverage. |
| `test:fuzz` | Runs bounded layout, broadcasting, and view fuzz checks. |
| `test:standalone` | Tests all modules with `GOWORK=off`. Requires published upstream versions. |
| `build:portable` | Cross-compiles Linux amd64/arm64, with CGo disabled, scalar and SIMD. |
| `bench:record` | Records hardware metadata and repeated scalar/SIMD measurements. |
| `docs:api` | Prints local GoDoc for the public packages. |

List every task with `mise tasks`.

## Daily workflow

For a tensor change, start with the focused tasks:

```sh
mise run fmt:tensor
mise run test:tensor
mise run lint:tensor
```

Before preparing a commit, run the normal local gate:

```sh
mise run check
```

For graph or parameter-lifecycle changes, also run `mise run test:race`.
For kernel changes, run `mise run test:simd` and compare representative benchmarks.
For indexing or shape changes, run `mise run test:fuzz`.

## Examples

| Example | Command | Data requirement |
| --- | --- | --- |
| Tensor basics and matrix multiplication | `mise run example:tensor` | None. |
| Linear regression | `mise run example:linear` | Generated constants. |
| XOR | `mise run example:xor` | Generated constants. |
| MNIST | `mise run example:mnist` | Explicitly downloaded local IDX files. |

Pass MNIST settings after the task separator:

```sh
mise run example:mnist -- -limit 10000 -epochs 5 -batch 64
```

See the [MNIST setup](../nn/examples/mnist/README.md) for local paths and downloads.

## Formatting and tests

Formatting uses golangci-lint's gofumpt and goimports integrations.
The wsl_v5 linter applies whitespace rules.
Use the formatter instead of compressing functions or correcting whitespace manually.

Implementation files have matching test files when they contain testable behavior.
Use operation-and-behavior names such as `TestDot_EmptyInput`.
Table subtests must identify meaningful input classes.

[CONTRIBUTING.md](../CONTRIBUTING.md) defines the code and review conventions.
The [validation guide](testing.md) explains numerical references and gradient checks.

## Dependency maintenance

| Task | Purpose |
| --- | --- |
| `deps:tidy` | Tidies all module manifests. |
| `deps:tidy:simdops`, `deps:tidy:tensor`, `deps:tidy:nn` | Tidies one module during an ordered release. |
| `deps:sync` | Synchronizes workspace dependency selections. |
| `deps:verify` | Verifies downloaded module checksums. |

After upstream versions exist, run:

```sh
mise run deps:tidy
mise run deps:sync
mise run deps:verify
mise run test:standalone
```

During bootstrap, downstream tidy and standalone tests can fail.
Their manifests require the intended upstream v0.1.0 tags.
Use workspace tests until those versions are published.
Do not add relative replace directives or fabricate checksum files.

## Tool upgrades

Update tool versions and lockfile entries deliberately:

1. Update the intended declaration in .mise.toml.
2. Run `mise lock` for the supported platforms.
3. Run `mise install --locked`.
4. Run `mise run ci`.
5. Review the configuration, lockfile, and generated sidecar changes together.

For a declaration using a floating selector, `mise lock --bump` requests a new resolution.
The existing lint and vulnerability-tool pins support the Go 1.27 toolchain.

Nancy stays on the original v1.2.0 module path.
A future v2 upgrade requires an explicit module-path and lockfile change.
Do not silently substitute a different major-version tool.

## Security tasks and network access

govulncheck is part of CI and can contact the vulnerability database.
Nancy is optional and can require configured Sonatype access.
Credentials belong outside the repository.

```sh
mise run security:vuln
mise run security:nancy
```

Ordinary unit tests use local data and do not download datasets.
Initial tool installation and optional data setup require network access.

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| Root `go test ./...` reports a workspace/module error | Use `mise run test`, which selects each module. |
| Locked installation reports an unrelated global tool | Use the explicit project-only installation command above. |
| Linter reports unsupported Go export data | Use the locked linter through mise. Check for an older tool on the shell PATH. |
| govulncheck reports no root go.mod | Use `mise run security:vuln`, which enters each module directory. |
| `GOEXPERIMENT=simd` is unknown | Check that mise selected Go 1.27.3. |
| Standalone tests report missing go.sum entries | Publish the upstream versions, then tidy that module. Workspace mode remains available meanwhile. |
| Benchmark results differ between runs | Compare the same CPU, toolchain, build mode, and sample method on an idle machine. |
| A task name conflicts with a mise command | Use the explicit `mise run <task>` form. |

## GitHub workflows

| Workflow | Trigger | Task |
| --- | --- | --- |
| [ci.yml](../.github/workflows/ci.yml) | Push to main and pull requests. | `mise run ci` |
| [simd.yml](../.github/workflows/simd.yml) | Push, pull request, or manual run. | `mise run test:simd` on amd64/arm64 runners. |
| [benchmark.yml](../.github/workflows/benchmark.yml) | Manual run. | `mise run bench:record` and artifact upload. |

The workflows install locked tools.
They are configured locally and have not run on GitHub yet.
