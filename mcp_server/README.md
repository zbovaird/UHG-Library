# UHG MCP Server

Model Context Protocol (MCP) server for the UHG Library. Runs over stdio and exposes tools that offload computation, improve accuracy, and save time.

## When to Use These Tools

Enable the MCP server in Cursor when you are:

- **Running or debugging UHG tests** — Use `run_tests` instead of manually invoking pytest.
- **Computing UHG operations** — Asking the AI to compute quadrance, cross-ratio, or other projective invariants; the tools return exact results.
- **Benchmarking** — Running ProjectiveGraphSAGE or similar benchmarks via `run_benchmark`.
- **Verifying workspace state** — Checking UHG version and test file count with `workspace_status`.

## Tools

| Tool | Description |
|------|-------------|
| `run_tests` | Run pytest on the UHG test suite. Returns pass/fail counts and failure details. |
| `uhg_quadrance` | Compute UHG quadrance between two points (projective coordinates). |
| `uhg_cross_ratio` | Compute cross-ratio of four points (projective invariant). |
| `workspace_status` | Get UHG version, workspace path, and test file count. |
| `run_benchmark` | Run a quick ProjectiveGraphSAGE benchmark with synthetic data. |
| `run_anomaly_smoke` | Run UHGUnsupervisedAnomalyDetector end-to-end on synthetic data (fit→cluster→summarize). |

## Installation

```bash
pip install "mcp[cli]"
# Or with extras:
pip install uhg[mcp]
```

## Usage

From the UHG-Library root:

```bash
python -m mcp_server.uhg_server
```

## Cursor Integration

Add to `~/.cursor/mcp.json` or project `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "uhg": {
      "command": "python",
      "args": ["-m", "mcp_server.uhg_server"],
      "cwd": "/absolute/path/to/UHG-Library"
    }
  }
}
```

Replace `cwd` with your actual workspace path.
