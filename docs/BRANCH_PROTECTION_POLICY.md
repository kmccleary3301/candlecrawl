# Branch Protection Policy

Target branch: `main`

Required protections (hosted git platform):
1. Require pull requests before merging.
2. Require at least 1 approving review.
3. Dismiss stale approvals on new commits.
4. Require status checks to pass:
- CandleCrawl CI / unit-and-api
- CandleCrawl CI / contract-validation
- CandleCrawl Security Scan / pip-audit
5. Require linear history.
6. Restrict force pushes and branch deletion.

Note: local bare staging remote enforces non-fast-forward and delete protections via server config.
