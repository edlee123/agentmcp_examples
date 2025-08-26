# agent_backends

This directory contains multiple agent backend examples. Each subfolder represents a different backend (e.g., `react_simple`).

- Place your backend code in a subfolder (e.g., `react_simple/`).
- Each backend should have its own Dockerfile, config, and code files.

## Docker Compose Usage

For the `react_simple` backend, use the Compose file named `compose.react_simple.yaml`:

```bash
docker compose -f compose.react_simple.yaml up -d
```

This allows you to maintain separate Compose files for different backends or deployment scenarios.
