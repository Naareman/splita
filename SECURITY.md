# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.x     | Yes       |

## Reporting a vulnerability

If you discover a security vulnerability in splita, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, email **security@splita.dev** (or open a private security advisory on GitHub) with:

1. A description of the vulnerability.
2. Steps to reproduce.
3. The potential impact.

You should receive an acknowledgement within 48 hours. We will work with you to understand the issue and coordinate a fix before any public disclosure.

## Scope

splita is a statistical computation library. Security concerns most relevant to this project include:

- **Input validation**: ensuring malicious or malformed inputs cannot cause crashes, excessive memory use, or infinite loops.
- **Denial of service**: resource exhaustion through crafted inputs (e.g. extremely large arrays).
- **Numerical stability**: ensuring computations do not produce silently incorrect results.

splita does not handle authentication, networking, file I/O, or user credentials.
