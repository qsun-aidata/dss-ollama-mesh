"""Convert Dataiku LLM messages and tool settings to OpenAI Chat format.

Dataiku's internal message schema differs from OpenAI's chat.completions API.
This module is the single place that normalizes roles, multimodal parts, tools,
and tool results before they reach the Ollama client.
"""

import json
import logging

from dssollamamesh.util import get_any, get_first_set, guess_image_mime, validate_image_uri

logger = logging.getLogger(__name__)


def build_conversation(messages):
    """Map Dataiku messages to OpenAI Chat messages, including tool roles."""
    logger.debug("MESSAGES_IN count=%d roles=%s", len(messages), [m.get("role") for m in messages])

    conversation = []
    for message in messages:
        role = message.get("role")

        # DSS tool results: {'role':'tool', 'toolOutputs':[{'callId','output'},...]}
        # One DSS message may hold several parallel results — split into OpenAI tool messages.
        if role == "tool":
            tool_outputs = message.get("toolOutputs")
            if tool_outputs:
                for out in tool_outputs:
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": get_any(out, "callId", "toolCallId", "id"),
                        "content": out.get("output") or "",
                    })
            else:
                conversation.append({
                    "role": "tool",
                    "tool_call_id": get_any(message, "toolCallId", "tool_call_id", "id"),
                    "content": message.get("content") or "",
                })
            continue

        # Assistant tool_calls must be preserved in history or the model repeats calls.
        tool_calls = get_any(message, "toolCalls", "tool_calls")
        if role == "assistant" and tool_calls:
            normalized = []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                normalized.append({
                    "id": get_any(tc, "id", "callId", "toolCallId") or "call_0",
                    "type": "function",
                    "function": {
                        "name": get_any(fn, "name") or tc.get("name"),
                        "arguments": get_any(fn, "arguments") or tc.get("arguments") or "{}",
                    },
                })
            conversation.append({
                "role": "assistant",
                "content": message.get("content") or None,
                "tool_calls": normalized,
            })
            continue

        # Plain string content (most user/assistant/system messages). Tested for a
        # real value, not mere presence: a multimodal message can carry both a null
        # `content` and a populated `parts`, and taking this branch would drop the
        # images silently.
        if message.get("content") is not None:
            conversation.append({"role": role, "content": message["content"]})
            continue

        # Multimodal: DSS uses a parts array; OpenAI expects a content array.
        parts = message.get("parts")
        if parts:
            content = []
            for part in parts:
                part_type = part.get("type")
                if part_type == "TEXT":
                    content.append({"type": "text", "text": part["text"]})
                elif part_type == "IMAGE_INLINE":
                    # DSS does not pass a MIME type, so sniff it from the payload header
                    # (guess_image_mime falls back to the jpeg template default).
                    inline = part["inlineImage"]
                    mime = get_any(part, "mimeType", "mime_type") or guess_image_mime(inline)
                    image_url = "data:%s;base64,%s" % (mime, inline)
                    content.append({"type": "image_url", "image_url": {"url": image_url}})
                elif part_type == "IMAGE_URI":
                    image_url = validate_image_uri(part["imageUrl"])
                    content.append({"type": "image_url", "image_url": {"url": image_url}})
                else:
                    # Say so rather than dropping it silently — a part type we do
                    # not handle yet looks exactly like a model ignoring the input.
                    logger.warning("Skipping unsupported message part type %r", part_type)
            conversation.append({"role": role, "content": content})

    logger.debug("CONVERSATION_OUT count=%d", len(conversation))
    return conversation


def build_tools(settings):
    """Map DSS tool definitions to OpenAI tools parameters."""
    tools = (settings or {}).get("tools")
    if not tools:
        return None
    out = []
    for tool in tools:
        fn = tool.get("function", tool)
        out.append({
            "type": "function",
            "function": {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    return out


def validate_tool_calls(tool_calls, settings):
    """Reject tool calls that the request did not declare.

    The Ollama endpoint is a trust boundary. A compromised server must not be
    able to invent a tool name and have DSS execute it merely because the model
    response was shaped like an OpenAI tool call.
    """
    declared = build_tools(settings) or []
    allowed_names = {tool["function"]["name"] for tool in declared}
    for index, tool_call in enumerate(tool_calls or []):
        function = tool_call.get("function") or {}
        name = function.get("name") or tool_call.get("name")
        if name not in allowed_names:
            raise ValueError("Ollama returned undeclared tool call at index %d" % index)
        arguments = function.get("arguments")
        if arguments is None:
            arguments = tool_call.get("arguments") or "{}"
        if not isinstance(arguments, str):
            raise ValueError("Ollama tool call arguments must be a JSON object string")
        try:
            parsed_arguments = json.loads(arguments or "{}")
        except (TypeError, ValueError) as err:
            raise ValueError("Ollama returned invalid JSON tool call arguments") from err
        if not isinstance(parsed_arguments, dict):
            raise ValueError("Ollama tool call arguments must encode a JSON object")
    return tool_calls


def build_tool_choice(settings):
    """Map DSS toolChoice to OpenAI tool_choice (auto / none / required / function)."""
    choice = (settings or {}).get("toolChoice")
    if not choice:
        return None
    choice_type = choice.get("type", "AUTO").upper()
    if choice_type == "AUTO":
        return "auto"
    if choice_type == "NONE":
        return "none"
    if choice_type in ("REQUIRED", "ANY"):
        return "required"
    if choice_type == "FUNCTION" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return "auto"


# OpenAI kwarg -> the DSS spellings it can arrive under. DSS uses camelCase, but
# settings dicts are also built by hand in recipes, so accept snake_case too.
_GENERATION_PARAMS = (
    ("temperature", ("temperature",)),
    ("max_tokens", ("maxOutputTokens", "maxTokens", "max_tokens")),
    ("top_p", ("topP", "top_p")),
    ("stop", ("stopSequences", "stop_sequences", "stop")),
    ("seed", ("seed",)),
)

# Ollama honours top_k; the OpenAI chat schema has no such field, so it has to
# ride along in extra_body rather than as a first-class kwarg.
_EXTRA_BODY_PARAMS = (
    ("top_k", ("topK", "top_k")),
)


def build_response_format(settings):
    """Map a DSS responseFormat to OpenAI's response_format. JSON mode only."""
    fmt = (settings or {}).get("responseFormat")
    if not fmt:
        return None
    fmt_type = fmt.get("type") if isinstance(fmt, dict) else fmt
    if str(fmt_type or "").upper() in ("JSON", "JSON_OBJECT"):
        return {"type": "json_object"}
    logger.debug("Ignoring unsupported responseFormat %r", fmt_type)
    return None


def build_generation_kwargs(settings):
    """Map DSS generation settings (temperature, max tokens, ...) to OpenAI kwargs.

    Only settings that were actually provided are emitted — passing an explicit
    None would override the model's own default rather than leave it alone. Uses
    get_first_set, not get_any, so that temperature=0 survives.
    """
    settings = settings or {}
    kwargs = {}

    for name, aliases in _GENERATION_PARAMS:
        value = get_first_set(settings, *aliases)
        if value is None:
            continue
        # DSS sends 0 for "no limit"; OpenAI reads it as "emit nothing".
        if name == "max_tokens" and value <= 0:
            continue
        kwargs[name] = value

    extra_body = {}
    for name, aliases in _EXTRA_BODY_PARAMS:
        value = get_first_set(settings, *aliases)
        if value is not None:
            extra_body[name] = value
    if extra_body:
        kwargs["extra_body"] = extra_body

    response_format = build_response_format(settings)
    if response_format:
        kwargs["response_format"] = response_format

    return kwargs


def build_chat_kwargs(model, query, settings):
    """Assemble the keyword arguments for client.chat.completions.create()."""
    kwargs = {
        "model": model,
        "messages": build_conversation(query["messages"]),
    }
    kwargs.update(build_generation_kwargs(settings))
    tools = build_tools(settings)
    if tools:
        kwargs["tools"] = tools
        tool_choice = build_tool_choice(settings)
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
    return kwargs
