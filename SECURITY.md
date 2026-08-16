# Security policy

Do not open a public issue containing credentials, private memory data, or a working exploit. Use GitHub private vulnerability reporting when it is available for this repository.

## Local-runtime guarantees

- The stable API binds only to `127.0.0.1` and has no silent cloud fallback.
- A generated bearer token protects every data endpoint; health output is secret-free.
- Provider credentials live in ignored local secret files or an explicitly named environment variable. They are never stored in the runtime JSON or usage ledger, and resolved provider-key values are not exported into the service process environment.
- Install, state, integration-outbox, and hook-log directories are owner-only (`0700` on POSIX; inheritance removed with an owner ACL on Windows). Secret and state files are owner-only as well.
- Error responses do not include provider response bodies.
- Released PyTorch checkpoints must be loaded with `weights_only=True`. Graph scorer byte counts and SHA-256 values are verified against the public release manifest before startup.
- Message and project deletion remove directly grounded records, truncate SQLite WAL files, and run `VACUUM`. External backups and provider retention remain outside that boundary.

## Build-tool constraint

PyTorch 2.11 currently requires `setuptools < 82`. That runtime tool version is
reported for CVE-2026-59890, whose affected path is source-distribution file
selection on normalization-preserving macOS filesystems. TMCRA does not build or
publish source distributions from the local runtime environment. Its isolated
package build requires `setuptools >= 83.0.0`, and the release tree is selected
by `PUBLIC_RELEASE_MANIFEST.json` plus the fail-closed audit script. Do not use
`.tmcra/venv` to publish unrelated Python source distributions.

## Before publishing

Run:

```bash
python scripts/audit_public_release.py --history
```

Then review the exact staged diff and confirm that only paths permitted by `PUBLIC_RELEASE_MANIFEST.json` are included. Never publish `.tmcra/`, real `.env` files, key files, databases, raw logs, production endpoints, or deployment artifacts.
