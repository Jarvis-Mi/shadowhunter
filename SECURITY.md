# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | ✅ |
| < 1.0 | ❌ |

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately to **gw2.fighter@gmail.com**, or through GitHub's
[private vulnerability reporting](https://github.com/Jarvis-Mi/shadowhunter/security/advisories/new).

Please include:

- what the issue is and why it is a security problem;
- the steps to reproduce it, ideally with a minimal case;
- the affected version and your environment (`python run.py doctor` output helps);
- any suggested fix, if you have one.

**Expected response:** an acknowledgement within 72 hours, and an assessment with a fix
timeline within 7 days. You will be credited in the advisory unless you prefer otherwise.

## Deployment notes

Shadow Hunter ships a FastAPI backend intended for **local and trusted-network use**. It has
no authentication layer by design. Before exposing it more widely, be aware that:

- `POST /api/scenes/upload` accepts GeoTIFF and image uploads and writes them to
  `data/workspace/`. It is a file-write surface — bound it with an upload size limit and a
  reverse proxy you control.
- `POST /api/train/rl` and `/api/train/cnn` start background compute jobs. Unauthenticated
  access to them is a denial-of-service surface.
- `/ws/telemetry` broadcasts training events to every connected client without filtering.
- The default bind is `127.0.0.1:8077`. Changing it to `0.0.0.0` puts all of the above on the
  network.

The recommended posture is to keep the backend on loopback and reach it through an
authenticating reverse proxy or an SSH tunnel.

## Scope

In scope: the Python package, the FastAPI service, the desktop and web front-ends, and the
supplied CI workflows.

Out of scope: vulnerabilities in third-party dependencies (report those upstream, though a
heads-up here is welcome), and the content of the external datasets referenced in the README.
