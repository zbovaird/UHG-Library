# Security policy

## Supported versions

Security fixes are applied to the **latest minor release** on the **default branch** (`main`). Older versions may not receive backports unless agreed in a specific advisory thread.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for undisclosed security vulnerabilities.

Instead, open a **private security advisory** on GitHub:

1. Go to **Security** → **Advisories** → **Report a vulnerability** on [github.com/zbovaird/UHG-Library](https://github.com/zbovaird/UHG-Library).

Or email the repository owner via the address on their GitHub profile if advisories are unavailable.

Include:

- Description of the issue and impact
- Steps to reproduce (proof-of-concept if safe)
- Suggested fix (optional)

We aim to acknowledge reports within a few days and coordinate disclosure after a fix is available.

## Scope

This policy covers the **uhg** Python package and this repository’s code. Third-party dependencies are subject to their upstream policies; use Dependabot alerts and `pip audit` / `uv pip audit` where appropriate.
