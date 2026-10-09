# Contribution Rules

## Issue Tracking

All requests for enhancements, bug fixes, or features must begin with the creation of an [issue](https://github.com/NVIDIA/GFMBench-api/issues).

The issue request will be reviewed by the NVIDIA team and approved prior to pull request integration and code review.

## Coding Guidelines

- Avoid introducing unnecessary complexity into existing code so that maintainability and readability are preserved.

- Keep pull requests (PRs) as concise as possible:
  - Avoid committing commented-out code.
  - Wherever possible, each PR should address a single concern. If several otherwise unrelated changes are needed to reach a desired endpoint, open separate PRs and indicate their dependencies in the descriptions. The more complex the changes in a single PR, the more time it will take to review.

- Make sure that you can contribute your work to open source without introducing license or patent conflicts. You must [sign off](#signing-your-work) on your commits.

- Thanks in advance for your patience as we review your contributions; we appreciate them!

## Pull Requests

The developer workflow for code contributions is as follows:

1. First, [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the [upstream GFMBench-API repository](https://github.com/NVIDIA/GFMBench-api).

2. Clone the forked repository, commit your changes, and push them to your personal fork:

   ```bash
   git clone https://github.com/YOUR_USERNAME/GFMBench-api.git
   cd GFMBench-api
   # Check out a branch and commit your changes.
   # Push the commits to a branch on your fork.
   git push -u origin <local-branch>:<remote-branch>
   ```

3. Once the changes on your fork are ready for review, [create a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) to merge your branch into the selected branch of the upstream repository.
   - Exercise caution when selecting the source and target branches for the PR.
   - Creating a PR starts the code review process.
   - While the PR is under development, mark it as a draft.

## Signing Your Work

We require all contributors to sign off on their commits. This certifies that the contribution is your original work, that you have the right to submit it under the same license, or that it uses a compatible license. Contributions containing commits that are not signed off will not be accepted.

To sign off on a commit, use the `--signoff` (or `-s`) option:

```bash
git commit -s -m "Add cool feature."
```

This appends the following line to your commit message:

```text
Signed-off-by: Your Name <your@email.com>
```

### Full Text of the DCO

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
```

```text
Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```
