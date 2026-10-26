from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote, urlencode

_USER_TOOL_PREFIX = "mcp__user_"
_USER_SERVER_PREFIX = "user-"
_PM_TIMEOUT_SECONDS = 5
_VALUE_CHARS = 300
_MESSAGE_CHARS = 3000


def _resolve(tool_name: str) -> Optional[Tuple[str, str, bool]]:
    # Imported per call: at load time an ImportError makes Hermes skip the plugin with a warning,
    # and the gate silently disappears.
    from tools import mcp_tool
    from tools.mcp_tool_schema import _build_utility_schemas, mcp_prefixed_tool_name
    from tools.mcp_tool_scope import _resolve_server_key

    server = mcp_tool._mcp_tool_server_names.get(tool_name)
    if server is None:
        return None
    hints = dict(mcp_tool._tool_read_only_hints.get(_resolve_server_key(server), {}))
    matches = [raw for raw in hints if mcp_prefixed_tool_name(server, raw) == tool_name]
    # Only after the server's own tools: Hermes drops its generated utility when an own tool takes that name.
    if not matches and server.startswith(_USER_SERVER_PREFIX):
        utilities = {entry["schema"]["name"]: entry["handler_key"] for entry in _build_utility_schemas(server)}
        if tool_name in utilities:
            return server, utilities[tool_name], True
    if len(matches) != 1:
        return None
    return server, matches[0], hints[matches[0]] is True


def _ask_pm(server: str, tool: str) -> bool:
    from agent.secret_scope import get_secret

    base, key = os.environ.get("DAS_PM_URL"), get_secret("DAS_API_KEY")
    if not base or not key:
        raise RuntimeError("DAS_PM_URL or DAS_API_KEY is not set")
    # http.client quotes a header value it rejects, the whole key included, in its error text.
    if not (key.isascii() and key.isprintable()):
        raise RuntimeError("DAS_API_KEY is malformed")
    query = urlencode({"server": server, "tool": tool}, quote_via=quote)
    request = urllib.request.Request(f"{base.rstrip('/')}/mcp/das/allowance?{query}",
                                     headers={"Authorization": f"Bearer {key}"})
    # The default opener follows a redirect with the key to the new host and goes through *_proxy from env.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _RefuseRedirects())
    with opener.open(request, timeout=_PM_TIMEOUT_SECONDS) as response:
        allowed = json.load(response)["allowed"]
    if not isinstance(allowed, bool):
        raise ValueError(f"PM answered allowed={allowed!r}")
    return allowed


def _on_pre_tool_call(tool_name: str = "", args: Any = None, **_: Any) -> Optional[Dict[str, str]]:
    if not tool_name.startswith(_USER_TOOL_PREFIX):
        return None
    try:
        resolved = _resolve(tool_name)
    except Exception as exc:
        return {"action": "block",
                "message": f"das-guard cannot read Hermes' MCP tool registry, so {tool_name} is blocked: "
                           f"{type(exc).__name__}: {exc}"}
    if resolved is None:
        return {"action": "block", "message": f"das-guard cannot identify the tool {tool_name}, so it is blocked."}
    server, raw, read_only = resolved
    if read_only:
        return None
    # Fail closed: a call PM did not clearly allow must never run unasked.
    try:
        from tools.approval_context import _is_cron_approval_context

        if _ask_pm(server, raw):
            return None
        if _is_cron_approval_context():
            return {"action": "block",
                    "message": f"The analyst has not allowed {server}/{raw} for this chat, so a scheduled run cannot "
                               f"call it. Choose \"Allow for this chat\" on that tool's card once in the chat."}
        return {"action": "approve", "rule_key": f"das-guard:{server}:{raw}",
                "message": _description(server, raw, args if isinstance(args, dict) else {})}
    except Exception as exc:
        return {"action": "block",
                "message": f"das-guard could not check with DAS whether {server}/{raw} is allowed for this chat, "
                           f"so the call is blocked: {type(exc).__name__}: {exc}"}


def _more(count: int, noun: str) -> str:
    return f"… ({count} more {noun}{'' if count == 1 else 's'})"


def _shown(value: Any) -> str:
    if isinstance(value, str):
        cut = len(value) - _VALUE_CHARS
        return json.dumps(value[:_VALUE_CHARS], ensure_ascii=False) + (_more(cut, "character") if cut > 0 else "")
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    cut = len(text) - _VALUE_CHARS
    return text[:_VALUE_CHARS] + (_more(cut, "character") if cut > 0 else "")


def _description(server: str, raw: str, args: Dict[str, Any]) -> str:
    lines = [f"{server} → {raw}"]
    size = len(lines[0])
    items = list(args.items())
    for index, (key, value) in enumerate(items):
        line = f"{_shown(key)}: {_shown(value)}"
        if size + 1 + len(line) > _MESSAGE_CHARS:
            lines.append(_more(len(items) - index, "argument"))
            break
        lines.append(line)
        size += 1 + len(line)
    return "\n".join(lines)


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None
