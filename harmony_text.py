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
    elif isinstance(obj, list):
        for item in obj:
            keys.update(_extract_tool_arg_keys_from_obj(item))
    return sorted(keys)


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
    """Build a compact, tool-aware behavior signature string for embedding.

    This is intentionally *not* the full Harmony text. It aims to capture:
    - tool identity + sequence (agentic behavior)
    - argument schema (keys only; no payloads/tool outputs)
    - conversation shape (assistant turns)
    - final answer type
    """
    try:
        msgs = parse_harmony(text)
    except Exception:
        return ""

    tool_calls = tool_call_sequence_from_messages(msgs)[:max_tools]
    tool_roles = tool_output_role_sequence_from_messages(msgs)[:max_tools]
    tool_keysets = tool_output_keysets_from_messages(msgs)[:max_tools]

    # Arg keys are extracted from CALL payloads only (cheap and avoids tool output pollution).
    arg_keys: list[str] = []
    for m in msgs:
        if m.end_tag != CALL_TAG:
            continue
        obj = _maybe_json_load(m.content, max_chars=20_000)
        if obj is None:
            continue
        arg_keys.extend(_extract_tool_arg_keys_from_obj(obj))
        if len(arg_keys) >= max_arg_keys:
            arg_keys = arg_keys[:max_arg_keys]
            break

    assistant_msgs = extract_assistant_messages(msgs)
    assistant_turns = len(assistant_msgs)
    assistant_chars = sum(len(m.content or "") for m in assistant_msgs)

    final = last_assistant_final(msgs)
    final_type = infer_answer_type(final.content if final else "")

    parts: list[str] = []
    parts.append(f"assistant_turns={assistant_turns}")
    parts.append(f"assistant_chars={assistant_chars}")
    parts.append(f"tool_call_count={len(tool_calls)}")
    parts.append(f"tool_call_seq={'->'.join(tool_calls) if tool_calls else 'none'}")
    parts.append(f"tool_output_roles={'->'.join(tool_roles) if tool_roles else 'none'}")
    parts.append(f"tool_output_keysets={'|'.join(tool_keysets) if tool_keysets else 'none'}")
    parts.append(f"tool_arg_keys={','.join(arg_keys) if arg_keys else 'none'}")
    parts.append(f"final_type={final_type}")
    return "\n".join(parts).strip()
