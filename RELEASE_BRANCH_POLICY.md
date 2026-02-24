# Release Branch Policy

- `main`: active development and release prep.
- `release/*`: stabilization branches for release candidates.
- tags `v*`: immutable release points used for artifact builds.

Hotfixes:
1. Branch from the last release tag.
2. Merge hotfix into `main` and the current release branch.
3. Cut patch tag.
