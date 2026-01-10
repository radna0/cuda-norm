from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Iterable


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
