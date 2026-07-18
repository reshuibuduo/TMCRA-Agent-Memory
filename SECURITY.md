# Security policy

Do not open a public issue containing credentials, private memory data, or a
working exploit. Use GitHub's private vulnerability reporting for this
repository when available.

TMCRA checkpoints must be loaded with `torch.load(..., weights_only=True)` and
verified against the published SHA-256 manifest. The maintainers do not
support disabling these checks for untrusted checkpoints.

Provider API keys belong only in local `*.env` files. The example files contain
placeholders and real environment files are excluded by `.gitignore`.
