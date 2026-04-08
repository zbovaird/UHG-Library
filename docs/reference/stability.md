# API stability and deprecation

## Public surface

Only symbols documented in [public-api.md](public-api.md) are guaranteed stable across **minor** and **patch** releases. Internal modules may change without a major release if they are not part of that list.

## Deprecation process

When a **stable** symbol must change incompatibly:

1. **Announce** in `CHANGELOG.md` and, where possible, emit a `DeprecationWarning` for at least one **minor** release.
2. **Maintain** a compatibility shim or alias when feasible.
3. **Remove** the old behavior in the next **major** version.

For urgent security fixes, the maintainers may shorten this window; such exceptions will be noted in the changelog.

## Imports

Prefer `from uhg import ...` for the stable API. Deep imports (`from uhg.nn...`) are fine for advanced use but may see more churn.
