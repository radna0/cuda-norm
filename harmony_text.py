from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Iterable, Iterator


START_TAG = "<|start|>"
END_TAG = "<|end|>"
CALL_TAG = "<|call|>"
CHANNEL_TAG = "<|channel|>"
MESSAGE_TAG = "<|message|>"
RETURN_TAG = "<|return|>"


@dataclass(frozen=True)
class HarmonyMessage:
    role: str
    content: str
    channel: str | None = None
    start: int | None = None
    end: int | None = None
    end_tag: str | None = None
    content_start: int | None = None
    content_end: int | None = None


def render_harmony(messages: Iterable[dict[str, Any]], *, add_return_tag: bool = True) -> str:
    msg_list = list(messages)
    parts: list[str] = []
    for idx, m in enumerate(msg_list):
        role = str(m.get("role") or "")
        if not role:
            raise ValueError("message missing role")
        content = m.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)

        if add_return_tag and idx == len(msg_list) - 1:
            end_tag = m.get("end_tag") or RETURN_TAG
        else:
            end_tag = m.get("end_tag") or END_TAG
        if end_tag not in {END_TAG, CALL_TAG, RETURN_TAG}:
            raise ValueError(f"unsupported end_tag: {end_tag!r}")

        channel = m.get("channel")
        if channel is not None and channel != "":
            parts.append(
                f"{START_TAG}{role}{CHANNEL_TAG}{channel}{MESSAGE_TAG}{content}{end_tag}"
            )
        else:
            parts.append(f"{START_TAG}{role}{MESSAGE_TAG}{content}{end_tag}")

    return "".join(parts)


def parse_harmony(text: str) -> list[HarmonyMessage]:
    if not isinstance(text, str):
        raise TypeError("text must be str")

    messages: list[HarmonyMessage] = []
    i = 0
    n = len(text)
    while True:
        start = text.find(START_TAG, i)
        if start < 0:
            break
        role_start = start + len(START_TAG)
        msg_tag = text.find(MESSAGE_TAG, role_start)
        if msg_tag < 0:
            raise ValueError("malformed harmony text: missing <|message|>")

        header = text[role_start:msg_tag]
        channel = None
        if CHANNEL_TAG in header:
            ch_pos = header.find(CHANNEL_TAG)
            role = header[:ch_pos]
            channel = header[ch_pos + len(CHANNEL_TAG) :]
        else:
            role = header

        content_start = msg_tag + len(MESSAGE_TAG)
        end_pos = text.find(END_TAG, content_start)
        call_pos = text.find(CALL_TAG, content_start)
        return_pos = text.find(RETURN_TAG, content_start)

        candidates: list[tuple[int, str]] = []
        if end_pos >= 0:
            candidates.append((end_pos, END_TAG))
        if call_pos >= 0:
            candidates.append((call_pos, CALL_TAG))
        if return_pos >= 0:
            candidates.append((return_pos, RETURN_TAG))

        if not candidates:
            raise ValueError(
                "malformed harmony text: missing <|end|>, <|call|>, or <|return|>"
            )

        delim_pos, delim_tag = min(candidates, key=lambda t: t[0])

        content = text[content_start:delim_pos]
        messages.append(
            HarmonyMessage(
                role=role,
                channel=channel,
                content=content,
                start=start,
                end=delim_pos + len(delim_tag),
                end_tag=delim_tag,
                content_start=content_start,
                content_end=delim_pos,
            )
        )
        i = delim_pos + len(delim_tag)
        if i >= n:
            break
    return messages


def assistant_content_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in parse_harmony(text):
        if not (m.role == "assistant" or m.role.startswith("assistant ")):
            continue
        if m.content_start is None or m.content_end is None:
            continue
        spans.append((m.content_start, m.content_end))
    return spans


def sha1_text_id(text: str) -> str:
    return sha1(text.encode("utf-8")).hexdigest()


def infer_loss_mode_from_text(text: str) -> str:
    # Our default: loss only on assistant message bodies (analysis+final),
    # excluding user/system/developer/tool.
    _ = text  # placeholder for future heuristics
    return "assistant_all"


def has_return_tag(text: str) -> bool:
    return text.rstrip().endswith(RETURN_TAG)


def extract_assistant_messages(messages: list[HarmonyMessage]) -> list[HarmonyMessage]:
    return [m for m in messages if m.role == "assistant" or m.role.startswith("assistant ")]


def last_assistant_final(messages: list[HarmonyMessage]) -> HarmonyMessage | None:
    for m in reversed(messages):
        if (m.role == "assistant" or m.role.startswith("assistant ")) and (
            m.channel == "final" or m.channel is None
        ):
            return m
    return None


def validate_tool_payload(payload: str) -> bool:
    s = payload.strip()
    if not s:
        return True
    if not (s.startswith("{") or s.startswith("[")):
        return True
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def basic_quality_flags_from_text(
    text: str, *, min_completion_chars: int = 32
) -> dict[str, Any]:
    try:
        msgs = parse_harmony(text)
        valid_harmony = True
    except Exception:
        msgs = []
        valid_harmony = False

    if not valid_harmony:
        return {
            "valid_harmony": False,
            "has_assistant": False,
            "completion_nonempty": False,
            "has_tool": False,
            "valid_tool_schema": False,
            "has_return_tag": has_return_tag(text),
            "num_messages": 0,
            "num_assistant_messages": 0,
        }

    assistant_msgs = extract_assistant_messages(msgs)
    last_final = last_assistant_final(msgs)
    completion_nonempty = bool(last_final and len(last_final.content.strip()) >= min_completion_chars)
    has_tool = CALL_TAG in text or any(
        (m.role == "tool")
        or (m.role not in {"system", "developer", "user"} and not m.role.startswith("assistant"))
        for m in msgs
    )
    valid_tool_schema = True
    if has_tool:
        for m in msgs:
            if m.role == "tool" or (
                m.role not in {"system", "developer", "user"} and not m.role.startswith("assistant")
            ):
                if not validate_tool_payload(m.content):
                    valid_tool_schema = False
                    break

    return {
        "valid_harmony": True,
        "has_assistant": bool(assistant_msgs),
        "completion_nonempty": completion_nonempty,
        "has_tool": has_tool,
        "valid_tool_schema": valid_tool_schema,
        "has_return_tag": has_return_tag(text),
        "num_messages": len(msgs),
        "num_assistant_messages": len(assistant_msgs),
    }


def iter_messages_by_role(
    messages: list[HarmonyMessage], *, role: str
) -> Iterator[HarmonyMessage]:
    role = role.strip()
    for m in messages:
        if m.role == role:
            yield m


def extract_user_prompt_text(text: str) -> str:
    """Extract a prompt-only text view for embedding/clustering.

    Default policy: include only user messages (no system/developer/tool/assistant) to avoid
    generic system prompts dominating embeddings and causing truncation.
    """
    parts: list[str] = []
    try:
        msgs = parse_harmony(text)
    except Exception:
        return ""
    for m in msgs:
        if m.role == "user":
            s = m.content.strip()
            if s:
                parts.append(s)
    return "\n\n".join(parts).strip()


def extract_assistant_reasoning_excerpt_text(
    text: str, *, max_chars: int = 50_000, prefer_analysis: bool = True
) -> str:
    """Extract an assistant-only reasoning excerpt for embedding/clustering.

    Goals:
    - Capture "reasoning style" diversity without embedding full Harmony text.
    - Exclude tool-call payloads (`<|call|>`) and tool outputs (role != assistant).
    - Prefer assistant channel="analysis" when present, else fall back to assistant final/None.
    """
    try:
        msgs = parse_harmony(text)
    except Exception:
        return ""

    assistant_noncall: list[HarmonyMessage] = []
    assistant_analysis: list[HarmonyMessage] = []
    assistant_finalish: list[HarmonyMessage] = []
    for m in msgs:
        if not (m.role == "assistant" or m.role.startswith("assistant ")):
            continue
        if m.end_tag == CALL_TAG:
            continue
        content = (m.content or "").strip()
        if not content:
            continue
        assistant_noncall.append(m)
        if (m.channel or "").strip().lower() == "analysis":
            assistant_analysis.append(m)
        elif (m.channel or "").strip().lower() in {"final", ""} or m.channel is None:
            assistant_finalish.append(m)

    chosen = assistant_analysis if (prefer_analysis and assistant_analysis) else assistant_finalish
    if not chosen:
        chosen = assistant_noncall
    if not chosen:
        return ""

    out = "\n\n".join((m.content or "").strip() for m in chosen if (m.content or "").strip()).strip()
    if max_chars > 0 and len(out) > max_chars:
        out = out[:max_chars].rstrip()
    return out


_ANSWER_TYPE_NUM_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*$")


def infer_answer_type(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return "empty"
    if "\\boxed" in s:
        return "boxed"
    if "```" in s:
        return "code"
    if _ANSWER_TYPE_NUM_RE.match(s):
        return "numeric"
    if s.startswith("{") or s.startswith("["):
        try:
            json.loads(s)
            return "json"
        except Exception:
            pass
    return "text"


def _maybe_json_load(s: str, *, max_chars: int = 50_000) -> Any | None:
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not t:
        return None
    if len(t) > max_chars:
        return None
    if not (t.startswith("{") or t.startswith("[")):
        return None
    try:
        return json.loads(t)
    except Exception:
        return None


def _extract_tool_names_from_obj(obj: Any) -> list[str]:
    names: list[str] = []
    if isinstance(obj, dict):
        # Common patterns.
        for k in ("tool_name", "name", "tool", "function_name"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                names.append(v.strip())
                break
        fn = obj.get("function")
        if isinstance(fn, dict):
            v = fn.get("name")
            if isinstance(v, str) and v.strip():
                names.append(v.strip())
        # Search nested call containers.
        for k in ("tool_calls", "calls", "call", "function_call", "tool_call"):
            v = obj.get(k)
            if isinstance(v, (list, dict)):
                names.extend(_extract_tool_names_from_obj(v))
    elif isinstance(obj, list):
        for item in obj:
            names.extend(_extract_tool_names_from_obj(item))
    return names


def _extract_tool_arg_keys_from_obj(obj: Any) -> list[str]:
    keys: set[str] = set()

    def add_from_args(args_obj: Any) -> None:
        if isinstance(args_obj, dict):
            for k in args_obj.keys():
                if isinstance(k, str) and k.strip():
                    keys.add(k.strip())
        elif isinstance(args_obj, str):
            parsed = _maybe_json_load(args_obj, max_chars=10_000)
            if isinstance(parsed, dict):
                add_from_args(parsed)

    if isinstance(obj, dict):
        if "arguments" in obj:
            add_from_args(obj.get("arguments"))
        fn = obj.get("function")
        if isinstance(fn, dict) and "arguments" in fn:
            add_from_args(fn.get("arguments"))
        for k in ("tool_calls", "calls", "call", "function_call", "tool_call"):
            v = obj.get(k)
            if isinstance(v, (list, dict)):
                keys.update(_extract_tool_arg_keys_from_obj(v))
        # Some Harmony tool-call payloads are *just* a JSON args object (no nested "arguments").
        # In that case, treat top-level keys as argument keys (we never record values).
        if not keys and not any(
            k in obj for k in ("tool_calls", "calls", "call", "function_call", "tool_call", "function")
        ):
            for k in obj.keys():
                if isinstance(k, str) and k.strip():
                    keys.add(k.strip())
    elif isinstance(obj, list):
        for item in obj:
            keys.update(_extract_tool_arg_keys_from_obj(item))
    return sorted(keys)


def _iter_tool_call_dicts(obj: Any) -> list[dict[str, Any]]:
    """Best-effort extraction of tool-call dicts in order."""
    out: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        # Most common: {"tool_calls": [ ... ]}.
        v = obj.get("tool_calls")
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    out.append(item)
        # Some variants: {"calls": [...]} or {"call": {...}}.
        v = obj.get("calls")
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    out.append(item)
        v = obj.get("call")
        if isinstance(v, dict):
            out.append(v)
        # OpenAI-like: {"function_call": {...}}
        v = obj.get("function_call")
        if isinstance(v, dict):
            out.append(v)
        # Some single-call shapes are the dict itself.
        if any(k in obj for k in ("tool_name", "tool", "name", "function", "function_name", "arguments")):
            out.append(obj)
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_iter_tool_call_dicts(item))
    return out


def _tool_name_from_call_dict(d: dict[str, Any]) -> str:
    for k in ("tool_name", "name", "tool", "function_name"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    fn = d.get("function")
    if isinstance(fn, dict):
        v = fn.get("name")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _tool_name_from_assistant_role(role: str) -> str:
    # Common Harmony pattern: role="assistant to=<tool_name>".
    s = (role or "").strip()
    m = re.match(r"^assistant\s+to=([^\s]+)", s)
    if not m:
        return ""
    return m.group(1).strip()


def _tool_arg_keys_from_call_dict(d: dict[str, Any], *, max_keys: int) -> list[str]:
    keys = _extract_tool_arg_keys_from_obj(d)
    if max_keys > 0:
        keys = keys[:max_keys]
    return keys


def _classify_tool_return(content: str) -> str:
    s = (content or "").strip()
    if not s:
        return "empty"
    if "traceback" in s.lower() or "error" in s.lower() or "exception" in s.lower():
        # Keep as 'error' even when it is JSON (common in tool errors).
        return "error"
    obj = _maybe_json_load(s, max_chars=50_000)
    if obj is None:
        return "ok"
    if isinstance(obj, dict):
        if not obj:
            return "empty"
        for k in ("error", "exception", "traceback", "stderr"):
            if k in obj:
                return "error"
        return "ok"
    if isinstance(obj, list):
        return "empty" if len(obj) == 0 else "ok"
    return "ok"


def _tool_return_keyset(content: str, *, max_keys: int = 16) -> str:
    obj = _maybe_json_load(content or "", max_chars=50_000)
    if isinstance(obj, dict):
        keys = [k for k in obj.keys() if isinstance(k, str) and k.strip()]
        keys = sorted(keys)[:max_keys]
        return ",".join(keys) if keys else "dict_empty"
    if isinstance(obj, list):
        return "list"
    if obj is None:
        return "nonjson"
    return type(obj).__name__


def extract_tool_call_sequence(text: str) -> list[str]:
    """Return a best-effort tool call name sequence in invocation order."""
    try:
        msgs = parse_harmony(text)
    except Exception:
        return []
    return tool_call_sequence_from_messages(msgs)


def extract_tool_output_role_sequence(text: str) -> list[str]:
    """Return tool output roles (role field) in order, excluding generic roles."""
    try:
        msgs = parse_harmony(text)
    except Exception:
        return []
    return tool_output_role_sequence_from_messages(msgs)


def tool_call_sequence_from_messages(messages: list[HarmonyMessage]) -> list[str]:
    seq: list[str] = []
    for m in messages:
        if m.end_tag != CALL_TAG:
            continue
        obj = _maybe_json_load(m.content, max_chars=20_000)
        if obj is None:
            continue
        calls = _iter_tool_call_dicts(obj)
        if calls:
            for d in calls:
                name = _tool_name_from_call_dict(d)
                if name:
                    seq.append(name)
        else:
            for name in _extract_tool_names_from_obj(obj):
                if name:
                    seq.append(name)
    return seq


def tool_output_role_sequence_from_messages(messages: list[HarmonyMessage]) -> list[str]:
    seq: list[str] = []
    for m in messages:
        role = m.role.strip()
        if not role:
            continue
        if role in {"system", "developer", "user"} or role.startswith("assistant"):
            continue
        # Some datasets use role="tool"; some use role="<tool_name>".
        if role == "tool":
            obj = _maybe_json_load(m.content, max_chars=20_000)
            if obj is not None:
                names = _extract_tool_names_from_obj(obj)
                if names:
                    seq.append(names[0])
                    continue
        seq.append(role)
    return seq


def tool_output_keysets_from_messages(
    messages: list[HarmonyMessage], *, max_keys_per_msg: int = 16
) -> list[str]:
    """Return compact tool-output schema keysets for messages that look like tool outputs.

    This is a fallback when tool identity is not present (e.g., role="tool" with JSON payload
    that doesn't include a name). We only record *keys*, never values.
    """
    out: list[str] = []
    for m in messages:
        role = m.role.strip()
        if not role:
            continue
        if role in {"system", "developer", "user"} or role.startswith("assistant"):
            continue
        obj = _maybe_json_load(m.content, max_chars=50_000)
        if isinstance(obj, dict):
            keys = [k for k in obj.keys() if isinstance(k, str) and k.strip()]
            keys = sorted(keys)[:max_keys_per_msg]
            out.append(",".join(keys) if keys else "dict_empty")
        elif isinstance(obj, list):
            out.append("list")
        elif obj is None:
            out.append("nonjson")
        else:
            out.append(type(obj).__name__)
    return out


def build_behavior_signature(text: str, *, max_tools: int = 16, max_arg_keys: int = 16) -> str:
    """Build a compact, tool-aware behavior trace sketch string for embedding.

    This is intentionally *not* the full Harmony text. It aims to encode tool-policy shape
    (sequence + schema keys + return status) without including raw tool payloads/outputs.
    """
    try:
        msgs = parse_harmony(text)
    except Exception:
        return ""

    # Extract ordered tool calls with per-call arg keysets.
    calls: list[tuple[str, list[str]]] = []
    for m in msgs:
        if m.end_tag != CALL_TAG:
            continue
        role_tool = _tool_name_from_assistant_role(m.role)
        obj = _maybe_json_load(m.content, max_chars=20_000)
        if obj is None:
            if role_tool:
                calls.append((role_tool, []))
            continue
        call_dicts = _iter_tool_call_dicts(obj)
        if not call_dicts:
            name = (_extract_tool_names_from_obj(obj) or [""])[0] or role_tool
            if name:
                calls.append((name, _extract_tool_arg_keys_from_obj(obj)[:max_arg_keys]))
            continue
        for d in call_dicts:
            name = _tool_name_from_call_dict(d) or role_tool
            if not name:
                continue
            calls.append((name, _tool_arg_keys_from_call_dict(d, max_keys=max_arg_keys)))
            if len(calls) >= max_tools:
                break
        if len(calls) >= max_tools:
            break

    # Fallback: some datasets (e.g., Nemotron-Agentic tool_calling) emit tool *outputs* without
    # explicit <|call|> boundaries. In that case, treat tool output messages as implicit calls.
    tool_out_msgs: list[HarmonyMessage] = []
    for m in msgs:
        role = (m.role or "").strip()
        if not role:
            continue
        if role in {"system", "developer", "user"} or role.startswith("assistant"):
            continue
        tool_out_msgs.append(m)

    calls_from_tool_outputs = False
    if not calls and tool_out_msgs:
        calls_from_tool_outputs = True
        for m in tool_out_msgs:
            role = (m.role or "").strip()
            keyset = _tool_return_keyset(m.content, max_keys=16)
            # Preserve explicit tool name when present; otherwise synthesize from output schema.
            name = role if role and role != "tool" else f"tool[{keyset}]"
            calls.append((name, []))
            if len(calls) >= max_tools:
                break

    call_names = [c[0] for c in calls]
    call_count = len(call_names)

    # Walk messages again to pair each call with the next tool-like output message.
    returns_status: list[str] = []
    returns_keys: list[str] = []
    returns_roles: list[str] = []
    if call_count:
        if calls_from_tool_outputs:
            for m, (name, _) in zip(tool_out_msgs, calls):
                returns_roles.append(name)
                returns_status.append(_classify_tool_return(m.content))
                returns_keys.append(_tool_return_keyset(m.content, max_keys=16))
        else:
            call_seen = 0
            expecting_return = False
            for m in msgs:
                if m.end_tag == CALL_TAG and call_seen < call_count:
                    expecting_return = True
                    call_seen += 1
                    continue
                if not expecting_return:
                    continue
                role = (m.role or "").strip()
                if not role:
                    continue
                if role in {"system", "developer", "user"} or role.startswith("assistant"):
                    continue
                # This looks like a tool output / tool message.
                if role == "tool" and len(returns_roles) < len(call_names) and call_names:
                    returns_roles.append(call_names[len(returns_roles)])
                else:
                    returns_roles.append(role)
                returns_status.append(_classify_tool_return(m.content))
                returns_keys.append(_tool_return_keyset(m.content, max_keys=16))
                expecting_return = False
                if len(returns_status) >= call_count:
                    break

    while len(returns_status) < call_count:
        returns_status.append("missing")
        returns_keys.append("missing")
        returns_roles.append("missing")

    final = last_assistant_final(msgs)
    final_text = final.content if final else ""
    final_type = infer_answer_type(final_text)
    used_tool_output = (
        "yes"
        if re.search(
            r"\b(tool output|based on the tool|according to the tool)\b",
            final_text,
            flags=re.IGNORECASE,
        )
        else "no"
    )

    plan_excerpt = ""
    for m in msgs:
        if m.role == "assistant" and m.channel == "analysis" and (m.content or "").strip():
            plan_excerpt = m.content.strip()
            break
    if not plan_excerpt:
        for m in msgs:
            if m.role == "assistant" and (m.content or "").strip():
                plan_excerpt = m.content.strip()
                break
    if plan_excerpt:
        # Keep this small; never embed the full completion.
        words = [w for w in plan_excerpt.split() if w]
        plan_excerpt = " ".join(words[:128]).strip()
        if len(plan_excerpt) > 800:
            plan_excerpt = plan_excerpt[:800].rstrip()

    args_per_call: list[str] = []
    for name, keys in calls:
        ks = ",".join(keys) if keys else ""
        args_per_call.append(f"{name}[{ks}]")

    parts: list[str] = []
    parts.append(f"TOOL_SEQ: {' -> '.join(call_names) if call_names else 'none'}")
    parts.append(f"CALLS: {call_count}")
    parts.append(f"ARGS: {' | '.join(args_per_call) if args_per_call else 'none'}")
    parts.append(f"RETURNS: {' | '.join(returns_status) if returns_status else 'none'}")
    parts.append(f"RET_KEYS: {' | '.join(returns_keys) if returns_keys else 'none'}")
    parts.append(f"RET_ROLES: {' | '.join(returns_roles) if returns_roles else 'none'}")
    parts.append(f"USED_TOOL_OUTPUT: {used_tool_output}")
    parts.append(f"ANSWER_TYPE: {final_type}")
    if plan_excerpt:
        parts.append(f"PLAN_EXCERPT: {plan_excerpt}")
    return "\n".join(parts).strip()
