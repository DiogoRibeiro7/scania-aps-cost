# Security Policy

## Supported versions

This is a research codebase. Security fixes are applied to the latest release
on the `main` branch only.

| Version | Supported |
| ------- | --------- |
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Report it privately through GitHub's
[private vulnerability reporting](https://github.com/DiogoRibeiro7/scania-aps-cost/security/advisories/new),
or by email to **diogo.debastos.ribeiro@gmail.com**.

Include what you can: affected version or commit, reproduction steps, and the
impact you believe it has. You can expect an acknowledgement within 7 days and
an assessment within 30 days. If the report is valid, you will be credited in
the advisory unless you ask otherwise.

## Scope notes

Two things are worth calling out for a project of this kind:

- **Runtime data download.** `scania-aps download` fetches the dataset over
  HTTPS from the UCI Machine Learning Repository. Reports about that code path
  (URL handling, archive extraction, path traversal) are in scope.
- **Untrusted model artifacts.** Files under `artifacts/` are loaded with
  `joblib`, which uses `pickle` and can execute arbitrary code. Only load
  artifacts you produced yourself. This is inherent to the format rather than a
  vulnerability in this project, but a report showing an unexpected
  deserialization path is in scope.

Out of scope: vulnerabilities in third-party dependencies with no exploitable
path through this code (report those upstream), and anything requiring an
attacker to already control the machine running the study.
