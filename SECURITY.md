# Security Policy

## Supported versions

Magma is in alpha. Security fixes are made on `main` and included in the next
pre-release; older pre-releases are not patched.

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them
privately through GitHub's
[private vulnerability reporting](https://github.com/pedronahum/Magma/security/advisories/new)
for this repository.

Include what you can of:

- the affected version or commit,
- your platform (OS, architecture, Swift toolchain, PJRT plugin and its XLA commit),
- steps or a minimal program that reproduces the problem, and
- the impact you expect.

You should get an acknowledgement within a week. Once a fix is available we
will publish an advisory and credit you, unless you prefer to stay anonymous.

## Scope

Magma loads PJRT plugins (shared libraries) from the paths described in the
README and executes them in-process. Only point `MAGMA_XLA_PATH` /
`MAGMA_PJRT_PLUGIN` at plugins you trust: a plugin runs with the full
privileges of your program, and that is not a vulnerability in Magma.
Checkpoint and dataset files, by contrast, are treated as untrusted input:
a crafted file that crashes or corrupts memory when loaded is in scope.
