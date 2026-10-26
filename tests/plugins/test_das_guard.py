import importlib.util
import json
import os
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from tools import approval_context, mcp_tool
from tools.mcp_tool_schema import mcp_prefixed_tool_name

SERVER = "user-5"
API_KEY = "das_agent_secret_key"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_plugin():
    plugin_dir = _repo_root() / "plugins" / "das-guard"
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.das_guard", plugin_dir / "__init__.py", submodule_search_locations=[str(plugin_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hermes_plugins.das_guard"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def plugin():
    return _load_plugin()


@pytest.fixture
def tools(monkeypatch):
    names, hints = {}, {}
    monkeypatch.setattr(mcp_tool, "_mcp_tool_server_names", names)
    monkeypatch.setattr(mcp_tool, "_tool_read_only_hints", hints)
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: None)

    def register(read_only_by_raw, server=SERVER):
        hints.setdefault(server, {}).update(read_only_by_raw)
        for raw in read_only_by_raw:
            names[mcp_prefixed_tool_name(server, raw)] = server
        return {raw: mcp_prefixed_tool_name(server, raw) for raw in read_only_by_raw}

    return register


@pytest.fixture
def cron(monkeypatch):
    state = {"on": False}
    monkeypatch.setattr(approval_context, "_is_cron_approval_context", lambda: state["on"])
    return state


@pytest.fixture
def pm(plugin, monkeypatch):
    state = {"allowed": False, "asked": []}

    def ask(server, tool):
        state["asked"].append((server, tool))
        return state["allowed"]

    monkeypatch.setattr(plugin, "_ask_pm", ask)
    return state


class _PmHandler(BaseHTTPRequestHandler):
    status = 200
    body = b'{"allowed": true}'
    location = None
    requests = []

    def do_GET(self):
        type(self).requests.append({"path": self.path, "authorization": self.headers.get("Authorization")})
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        if type(self).location:
            self.send_header("Location", type(self).location)
        self.end_headers()
        self.wfile.write(type(self).body)

    def log_message(self, *_):
        pass


@pytest.fixture
def pm_server(monkeypatch):
    handler = type("Handler", (_PmHandler,), {"requests": []})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    monkeypatch.setenv("DAS_PM_URL", f"http://127.0.0.1:{server.server_address[1]}/")
    monkeypatch.setenv("DAS_API_KEY", API_KEY)
    yield handler
    server.shutdown()
    server.server_close()


def test_non_user_tools_pass(plugin, tools, pm, cron):
    tools({"search": False}, server="collection-7")
    assert plugin._on_pre_tool_call(tool_name="terminal", args={"command": "ls"}) is None
    assert plugin._on_pre_tool_call(tool_name="mcp__collection_7__search", args={}) is None
    assert pm["asked"] == []


def test_read_only_tool_passes(plugin, tools, pm, cron):
    names = tools({"list_issues": True})
    assert plugin._on_pre_tool_call(tool_name=names["list_issues"], args={}) is None
    assert pm["asked"] == []


def test_tool_without_annotation_asks(plugin, tools, pm, cron):
    names = tools({"create_issue": False})
    out = plugin._on_pre_tool_call(tool_name=names["create_issue"], args={"title": "x"})
    assert out["action"] == "approve"
    assert pm["asked"] == [(SERVER, "create_issue")]


def test_allowed_tool_passes(plugin, tools, pm, cron):
    names = tools({"create_issue": False})
    pm["allowed"] = True
    cron["on"] = True
    assert plugin._on_pre_tool_call(tool_name=names["create_issue"], args={}) is None
    assert pm["asked"] == [(SERVER, "create_issue")]


def test_not_allowed_in_cron_is_blocked(plugin, tools, pm, cron):
    names = tools({"create_issue": False})
    cron["on"] = True
    out = plugin._on_pre_tool_call(tool_name=names["create_issue"], args={})
    assert out == {
        "action": "block",
        "message": "The analyst has not allowed user-5/create_issue for this chat, so a scheduled run cannot call it. "
                   "Choose \"Allow for this chat\" on that tool's card once in the chat.",
    }


@pytest.mark.parametrize("status, body", [
    (500, b'{"error": "boom"}'),
    (401, b'{"error": "unauthorized"}'),
    (200, b'{"allowed": "yes"}'),
    (200, b'not json'),
])
def test_pm_error_blocks(plugin, tools, cron, pm_server, status, body):
    names = tools({"create_issue": False})
    pm_server.status, pm_server.body = status, body
    out = plugin._on_pre_tool_call(tool_name=names["create_issue"], args={})
    assert out["action"] == "block"
    assert "user-5/create_issue" in out["message"]
    assert API_KEY not in out["message"]


def test_unreachable_pm_blocks(plugin, tools, cron, monkeypatch):
    names = tools({"create_issue": False})
    monkeypatch.setenv("DAS_PM_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("DAS_API_KEY", API_KEY)
    out = plugin._on_pre_tool_call(tool_name=names["create_issue"], args={})
    assert out["action"] == "block"
    assert API_KEY not in out["message"]


@pytest.mark.parametrize("url, key", [
    (None, API_KEY),
    ("http://127.0.0.1:1", None),
    ("http://127.0.0.1:1", "secret-part\nother-part"),
])
def test_missing_or_malformed_settings_block(plugin, tools, cron, monkeypatch, url, key):
    names = tools({"create_issue": False})
    for name, value in (("DAS_PM_URL", url), ("DAS_API_KEY", key)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    out = plugin._on_pre_tool_call(tool_name=names["create_issue"], args={})
    assert out["action"] == "block"
    assert "secret-part" not in out["message"]


def test_ask_pm_sends_raw_names_with_the_key(plugin, tools, cron, pm_server, monkeypatch):
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    names = tools({"create issue/v2": False})
    assert plugin._on_pre_tool_call(tool_name=names["create issue/v2"], args={}) is None
    [request] = pm_server.requests
    assert request["path"] == "/mcp/das/allowance?server=user-5&tool=create%20issue%2Fv2"
    assert request["authorization"] == f"Bearer {API_KEY}"


def test_clamped_long_name_is_resolved(plugin, tools, pm, cron):
    raw = "create_" + "very_long_" * 8 + "issue"
    names = tools({raw: False, "list_issues": True})
    assert len(names[raw]) == 64 and raw not in names[raw]
    out = plugin._on_pre_tool_call(tool_name=names[raw], args={})
    assert out["action"] == "approve"
    assert pm["asked"] == [(SERVER, raw)]


def test_rule_key_carries_raw_names(plugin, tools, pm, cron):
    names = tools({"create-issue": False})
    args = {"title": "Звіт", "labels": ["a"]}
    out = plugin._on_pre_tool_call(tool_name=names["create-issue"], args=args)
    assert out == {
        "action": "approve",
        "rule_key": "das-guard:user-5:create-issue",
        "message": "user-5 → create-issue\n\"title\": \"Звіт\"\n\"labels\": [\"a\"]",
    }


def test_unknown_or_ambiguous_tool_is_blocked(plugin, tools, pm, cron):
    names = tools({"create-issue": True, "create_issue": False})
    assert names["create-issue"] == names["create_issue"]
    out = plugin._on_pre_tool_call(tool_name=names["create_issue"], args={})
    assert out["action"] == "block"
    assert plugin._on_pre_tool_call(tool_name="mcp__user_5__gone", args={})["action"] == "block"
    assert pm["asked"] == []


def test_manifest_declares_registered_hooks(plugin):
    import hermes_yaml as yaml

    manifest = yaml.safe_load((_repo_root() / "plugins" / "das-guard" / "plugin.yaml").read_text(encoding="utf-8"))
    registered = []

    class HookContext:
        def register_hook(self, name, _callback):
            registered.append(name)

    plugin.register(HookContext())
    assert manifest["name"] == "das-guard"
    assert set(manifest["provides_hooks"]) == set(registered) == {"pre_tool_call"}


@pytest.mark.parametrize("utility", ["list_resources", "read_resource", "list_prompts", "get_prompt"])
def test_utility_tools_pass(plugin, tools, pm, cron, utility):
    tools({"create_issue": False})
    name = mcp_prefixed_tool_name(SERVER, utility)
    mcp_tool._mcp_tool_server_names[name] = SERVER
    assert plugin._on_pre_tool_call(tool_name=name, args={}) is None
    assert pm["asked"] == []


def test_own_tool_named_like_a_utility_still_asks(plugin, tools, pm, cron):
    names = tools({"read_resource": False})
    out = plugin._on_pre_tool_call(tool_name=names["read_resource"], args={"uri": "x"})
    assert out["action"] == "approve"
    assert pm["asked"] == [(SERVER, "read_resource")]


def test_utility_tools_pass_only_on_user_servers(plugin, tools, pm, cron):
    tools({}, server="user_5")
    name = mcp_prefixed_tool_name("user_5", "read_resource")
    mcp_tool._mcp_tool_server_names[name] = "user_5"
    assert plugin._on_pre_tool_call(tool_name=name, args={})["action"] == "block"


@pytest.mark.parametrize("module, attribute", [
    ("tools.mcp_tool", "_tool_read_only_hints"),
    ("tools.mcp_tool_schema", "_build_utility_schemas"),
    ("tools.mcp_tool_scope", "_resolve_server_key"),
])
def test_moved_hermes_internal_blocks(tools, cron, monkeypatch, module, attribute):
    names = tools({"create_issue": False})
    monkeypatch.delattr(importlib.import_module(module), attribute)
    plugin = _load_plugin()
    out = plugin._on_pre_tool_call(tool_name=names["create_issue"], args={})
    assert out["action"] == "block"
    assert "cannot read Hermes' MCP tool registry" in out["message"]
    assert plugin._on_pre_tool_call(tool_name="terminal", args={}) is None


def test_redirect_blocks_without_following(plugin, tools, cron, pm_server):
    names = tools({"create_issue": False})
    pm_server.status, pm_server.body = 302, b""
    pm_server.location = os.environ["DAS_PM_URL"] + "redirected"
    out = plugin._on_pre_tool_call(tool_name=names["create_issue"], args={})
    assert out["action"] == "block"
    assert [r["path"] for r in pm_server.requests] == ["/mcp/das/allowance?server=user-5&tool=create_issue"]


def test_a_long_value_is_marked(plugin, tools, pm, cron):
    names = tools({"create_issue": False})
    fields = {"note": "y" * 1000}
    out = plugin._on_pre_tool_call(tool_name=names["create_issue"],
                                   args={"body": "x" * 1000, "fields": fields, "title": "t"})
    compact = json.dumps(fields, separators=(",", ":"))
    assert out["message"].split("\n") == [
        "user-5 → create_issue",
        "\"body\": \"" + "x" * 300 + "\"… (700 more characters)",
        "\"fields\": " + compact[:300] + f"… ({len(compact) - 300} more characters)",
        "\"title\": \"t\"",
    ]


def test_every_key_appears_when_values_are_long(plugin, tools, pm, cron):
    names = tools({"create_issue": False})
    message = plugin._on_pre_tool_call(tool_name=names["create_issue"],
                                       args={f"field_{i}": "z" * 2000 for i in range(8)})["message"]
    assert all(f"\"field_{i}\": \"" in message for i in range(8))
    assert "more argument" not in message


def test_the_message_is_bounded(plugin, tools, pm, cron):
    names = tools({"create_issue": False})
    lines = plugin._on_pre_tool_call(tool_name=names["create_issue"],
                                     args={f"field_{i:03}": "z" * 2000 for i in range(100)})["message"].split("\n")
    shown = len(lines) - 2
    assert 0 < shown < 100
    assert lines[-1] == f"… ({100 - shown} more arguments)"
    assert all(line.startswith(f"\"field_{i:03}\": ") for i, line in enumerate(lines[1:-1]))
    assert len("\n".join(lines[:-1])) <= 3000


def test_the_card_carries_the_rule_through_hermes(tools, cron, pm_server, monkeypatch):
    from hermes_cli import plugins
    from tools import approval

    Path(os.environ["HERMES_HOME"], "config.yaml").write_text("plugins:\n  enabled: [das-guard]\n", encoding="utf-8")
    plugins.discover_plugins(force=True)
    names = tools({"create_issue": False})
    pm_server.body = b'{"allowed": false}'
    for module in (approval, approval_context):
        monkeypatch.setattr(module, "get_current_session_key", lambda default="default": "das-session")
        monkeypatch.setattr(module, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval, "is_approved", lambda sk, pk: False)
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False, raising=False)
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
    monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "api_server")
    notified = []

    def deny(data):
        notified.append(data)
        approval.resolve_gateway_approval("das-session", "deny", request_id=data["request_id"])

    approval.register_gateway_notify("das-session", deny)
    try:
        blocked = plugins.resolve_pre_tool_block(names["create_issue"], {"title": "x"})
    finally:
        approval.unregister_gateway_notify("das-session")
    assert [data["pattern_key"] for data in notified] == ["plugin_rule:das-guard:user-5:create_issue"]
    assert notified[0]["description"] == "user-5 → create_issue\n\"title\": \"x\""
    assert blocked
    assert [r["path"] for r in pm_server.requests] == ["/mcp/das/allowance?server=user-5&tool=create_issue"]
