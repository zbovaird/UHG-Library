# MCP server (development tooling)

The [`mcp_server/`](https://github.com/zbovaird/UHG-Library/tree/main/mcp_server) package is **optional** development tooling. It is **not** part of the default `uhg` install on PyPI.

Install extra:

```bash
pip install "uhg[mcp]"
```

Run (from a clone of this repository):

```bash
python -m mcp_server.uhg_server
```

See [`mcp_server/README.md`](https://github.com/zbovaird/UHG-Library/blob/main/mcp_server/README.md) for Cursor and editor integration.

**Future:** The MCP server may move to a separate repository to keep the core library README focused on PyPI consumers. Until then, it remains in-tree under `mcp_server/`.
