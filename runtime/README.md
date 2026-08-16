# TMCRA Local Runtime

This package is the owner-local execution boundary for TMCRA. It provides the
memory Writer, Source/Fast/Slow graph, project and owner-global scopes, recall,
evidence packing, Visual Atlas, Personal Knowledge projection, physical local
deletion, local provider-token accounting, and a loopback-only FastAPI service.

It does not contain the hosted TMCRA account, subscription, billing, staff,
tenant, or production operations control plane.

Use the repository-level local deployment guide rather than installing this
directory in isolation; the runtime loads the released graph core and learned
scoring assets from the same repository clone.

The stable entry point is `tmcra-local`. Important commands are:

```text
tmcra-local configure          create a secret-free runtime policy
tmcra-local set-key            write a BYOK key to its local secret file
tmcra-local download-model     download and verify a selected model
tmcra-local doctor             validate assets, credentials, and real probes
tmcra-local start              start the API on 127.0.0.1:2009
tmcra-local token              inspect the generated local API token path
```

See `../docs/LOCAL_DEPLOYMENT.md` and `../docs/LOCAL_API.md` for the supported
installation and integration contracts.
