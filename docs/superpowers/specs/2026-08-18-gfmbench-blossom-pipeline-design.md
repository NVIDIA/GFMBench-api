# GFMBench Blossom CI Pipeline Design

## Objective

Provide a secure, reliable pull-request CI path for `NVIDIA/GFMBench-api`. An authorized GitHub comment starts the NVIDIA Blossom workflow, vulnerability scanning completes before Jenkins is contacted, and Jenkins validates the exact pull-request revision with fast, deterministic checks.

## Scope

The required pull-request gate will:

- accept the documented `/build` command only on pull requests and only from an explicit allowlist;
- run the Blossom authorization and vulnerability-scan stages;
- hand the validated repository and revision to the `GFMBench` Jenkins job;
- install GFMBench in an isolated Python environment;
- run linting, unit tests, and offline smoke tests;
- publish machine-readable test and coverage results;
- enforce timeouts, prevent overlapping builds, retain a bounded build history, and clean the workspace.

Network download tests and the heavy regression test are excluded from the required fast gate. They will be exposed through an explicit Jenkins `TEST_MODE=full` parameter so they can run on demand without slowing or destabilizing routine pull-request validation.

## Architecture

### GitHub workflow

The workflow lives at `.github/workflows/blossom-ci.yml`. It handles four responsibilities only:

1. authorize an exact `/build` pull-request comment from an allowlisted GitHub user;
2. run the NVIDIA vulnerability-scan runner and Blossom action;
3. ask Blossom to start the configured Jenkins job through the `CI_SERVER` secret;
4. relay Jenkins post-processing output back to the pull request through `workflow_dispatch`.

The workflow will declare least-privilege GitHub permissions and use current supported action versions. It will not execute pull-request code on the privileged Blossom authorization runner.

### Jenkins pipeline

A repository-owned `Jenkinsfile` will define the build so pipeline behavior is reviewed alongside application changes. Jenkins will use the SCM revision supplied by Blossom rather than selecting a branch independently.

The pipeline stages are:

1. **Checkout** — remove stale workspace state and check out the exact supplied revision.
2. **Environment** — create a workspace-local virtual environment and record Python and package-tool versions.
3. **Install** — install the project with its test dependencies.
4. **Lint** — run Ruff over application and test code.
5. **Fast tests** — run unit tests and offline smoke tests with JUnit and coverage output.
6. **Full tests** — only when `TEST_MODE=full`, run network download tests and the heavy regression test with a longer stage timeout.
7. **Publish** — publish JUnit and coverage artifacts even when tests fail.

The job will disable concurrent executions, apply an overall timeout, retain a bounded number of builds, add timestamps, and always clean its workspace.

## Test Selection

The default fast gate runs:

- `tests/unit/test_caching_utils.py`
- `tests/e2e/test_smoke.py`

The on-demand full mode additionally runs:

- `tests/e2e/test_download.py`
- `tests/e2e/test_heavy.py::test_heavy_sanity_regression`

The baseline-update test remains excluded because it intentionally rewrites reference data and is not a validation test.

## Failure Behavior

- Authorization failure stops before vulnerability scanning or Jenkins triggering.
- Vulnerability-scan failure stops before Jenkins triggering.
- Checkout failure reports the requested revision and fails immediately.
- Dependency, lint, or test failures fail the Jenkins build at their originating stage.
- JUnit and coverage publication runs after failures when result files exist.
- Cleanup runs for every terminal result, including timeout and abort.
- No test failure is converted into a successful build.

## Security and Credentials

- Repository secrets remain in GitHub or Jenkins credential storage and are never written into repository files or Git remote URLs.
- The GitHub workflow checks both that the event targets a pull request and that the command matches exactly.
- Pull-request code is checked out only on the vulnerability-scan runner and Jenkins worker, after authorization.
- Jenkins logs must not print tokens, secret values, or authenticated repository URLs.

## Verification

Before enabling the job as a required check:

1. validate the workflow YAML and Jenkins declarative syntax;
2. confirm the GitHub workflow exists under `.github/workflows`;
3. test an unauthorized comment and verify that neither scan nor Jenkins runs;
4. test `/build` from an authorized user and verify the exact PR revision is checked out;
5. introduce a controlled lint or unit-test failure and verify Jenkins and the PR report failure;
6. run `TEST_MODE=full` once and verify network, disk, cache, and timeout requirements on the selected Jenkins worker;
7. verify JUnit, coverage, log relay, retention, and cleanup behavior.

## Out of Scope

- release, package publication, or deployment;
- automatic execution of the full network-heavy suite on every pull request;
- model-specific GPU validation;
- changes to the GFMBench application API or benchmark implementation.
