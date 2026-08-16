# Public release boundary

This source distribution is designed to be useful without TMCRA infrastructure. A clone contains the memory algorithm, local API, local storage, model-selection policy, released inference weights, benchmark reproduction, tests, and install/start/uninstall tools.

It intentionally excludes:

- hosted account, subscription, quota, payment, and billing systems;
- staff consoles, tenant administration, production queues, alerts, and deployment automation;
- production domains, host addresses, credentials, key pools, database DSNs, and environment files;
- private user data, production memory databases, request logs, and raw operational logs;
- private training corpora, training-machine paths, and resumable optimizer/run state;
- website source and desktop installer binaries.

The local runtime must not import a production service module or silently fall back to a TMCRA-hosted endpoint. Its public listener is fixed to `127.0.0.1`. External generation is possible only through an endpoint and credential explicitly supplied by the user.

`PUBLIC_RELEASE_MANIFEST.json` defines the allowed top-level tree and required runtime files. `scripts/audit_public_release.py` checks the current tracked/untracked release tree for boundary violations, generated state, high-confidence credentials, production hosts, internal machine paths, and forbidden service imports. `--history` also checks reachable Git history without printing matched values.

The scanner is a release gate, not a proof that arbitrary text contains no secret. Maintainers must still review the staged diff, model manifests, third-party licenses, and generated archives before every public push.
