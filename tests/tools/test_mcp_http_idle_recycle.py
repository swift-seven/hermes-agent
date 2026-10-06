"""A remote MCP server can stop pinging and close its session once nobody uses it.

``keepalive_interval: 0`` turns the keepalive off, and ``idle_timeout_seconds`` recycles an HTTP
session the way it recycles a stdio one: the transport closes, the tools stay offered, and the
next call reconnects. A server that holds real resources for every open session (a data
collection that keeps its models loaded while a client is connected) is released only when the
client lets go, so neither a ping nor an unrelated discovery pass may reopen it.
"""

import asyncio

import pytest


@pytest.mark.asyncio
async def test_idle_http_session_recycles_without_pinging(monkeypatch):
    from tools import mcp_tool
    from tools.mcp_tool import MCPServerTask

    monkeypatch.setattr(mcp_tool, "_MIN_KEEPALIVE_INTERVAL", 0.01)
    probes: list[int] = []

    async def _probe(self):
        probes.append(1)

    monkeypatch.setattr(MCPServerTask, "_keepalive_probe", _probe)
    server = MCPServerTask("collection")
    server._config = {"url": "http://127.0.0.1:9/sse", "keepalive_interval": 0}
    server._idle_timeout_seconds = 0.3
    server.session = object()

    reason = await asyncio.wait_for(server._wait_for_lifecycle_event(), timeout=3.0)

    assert reason == "recycle"
    assert probes == []
    assert server.session is None and server._is_recycled_stdio()


@pytest.mark.no_isolate
def test_discovery_leaves_a_recycled_server_dormant(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools import mcp_tool_discovery as _mcp_discovery

    woken: list[str] = []

    class _Event:
        def __init__(self, name):
            self._name = name

        def set(self):
            woken.append(self._name)

    class _Dormant:
        session = None

        def __init__(self, name, recycled_reason):
            self.name = name
            self._recycled_reason = recycled_reason
            self._reconnect_event = _Event(name)
            self._registered_tool_names = [f"{name}__tool"]

    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setitem(mcp_tool._servers, "recycled-srv", _Dormant("recycled-srv", "idle_timeout_seconds"))
    monkeypatch.setitem(mcp_tool._servers, "parked-srv", _Dormant("parked-srv", None))

    _mcp_discovery.register_mcp_servers({
        "recycled-srv": {"url": "http://127.0.0.1:9/sse"},
        "parked-srv": {"url": "http://127.0.0.1:9/sse"},
    })

    assert woken == ["parked-srv"]
