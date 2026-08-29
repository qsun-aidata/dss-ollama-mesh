import base64

import pytest

from dssollamamesh.messages import (
    build_chat_kwargs,
    build_conversation,
    build_generation_kwargs,
    build_response_format,
    build_tool_choice,
    build_tools,
    validate_tool_calls,
)


def test_build_conversation_user_message():
    out = build_conversation([{"role": "user", "content": "hello"}])
    assert out == [{"role": "user", "content": "hello"}]


def test_build_conversation_tool_outputs():
    out = build_conversation([{
        "role": "tool",
        "toolOutputs": [{"callId": "c1", "output": "ok"}],
    }])
    assert out == [{"role": "tool", "tool_call_id": "c1", "content": "ok"}]


def test_build_conversation_splits_parallel_tool_outputs():
    # Pitfall #4: DSS bundles parallel results in one message.
    out = build_conversation([{
        "role": "tool",
        "toolOutputs": [
            {"callId": "c1", "output": "first"},
            {"callId": "c2", "output": "second"},
        ],
    }])
    assert out == [
        {"role": "tool", "tool_call_id": "c1", "content": "first"},
        {"role": "tool", "tool_call_id": "c2", "content": "second"},
    ]


def test_build_conversation_preserves_assistant_tool_calls():
    # Pitfall #3: dropping these makes the model repeat the same calls.
    out = build_conversation([{
        "role": "assistant",
        "toolCalls": [{"id": "c1", "function": {"name": "lookup", "arguments": '{"q":1}'}}],
    }])
    assert out == [{
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "c1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":1}'},
        }],
    }]


def test_build_conversation_multimodal_sniffs_image_mime():
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32).decode()
    out = build_conversation([{
        "role": "user",
        "parts": [
            {"type": "TEXT", "text": "what is this?"},
            {"type": "IMAGE_INLINE", "inlineImage": png},
            {"type": "IMAGE_URI", "imageUrl": "https://example.com/a.png"},
        ],
    }])
    content = out[0]["content"]
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[2]["image_url"]["url"] == "https://example.com/a.png"


def test_build_conversation_rejects_non_http_image_uri():
    with pytest.raises(ValueError):
        build_conversation([{
            "role": "user",
            "parts": [{"type": "IMAGE_URI", "imageUrl": "file:///etc/passwd"}],
        }])


def test_build_conversation_honours_explicit_mime_type():
    out = build_conversation([{
        "role": "user",
        "parts": [{"type": "IMAGE_INLINE", "inlineImage": "AAAA", "mimeType": "image/webp"}],
    }])
    assert out[0]["content"][0]["image_url"]["url"].startswith("data:image/webp;base64,")


def test_build_conversation_prefers_parts_over_null_content():
    # A multimodal message can carry both; taking the content branch loses the image.
    out = build_conversation([{
        "role": "user",
        "content": None,
        "parts": [{"type": "TEXT", "text": "hi"}],
    }])
    assert out == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_build_conversation_skips_unknown_part_types():
    out = build_conversation([{
        "role": "user",
        "parts": [{"type": "TEXT", "text": "hi"}, {"type": "AUDIO_INLINE"}, {}],
    }])
    assert out[0]["content"] == [{"type": "text", "text": "hi"}]


def test_build_generation_kwargs_maps_dss_settings():
    out = build_generation_kwargs({
        "temperature": 0.7,
        "maxOutputTokens": 512,
        "topP": 0.9,
        "stopSequences": ["\n\n"],
        "seed": 42,
    })
    assert out == {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
        "stop": ["\n\n"],
        "seed": 42,
    }


def test_build_generation_kwargs_accepts_snake_case():
    out = build_generation_kwargs({"max_tokens": 128, "top_p": 0.5, "stop": ["END"]})
    assert out == {"max_tokens": 128, "top_p": 0.5, "stop": ["END"]}


def test_build_generation_kwargs_keeps_zero_temperature():
    # The whole reason build_generation_kwargs uses get_first_set, not get_any.
    assert build_generation_kwargs({"temperature": 0})["temperature"] == 0


def test_build_generation_kwargs_omits_unset_settings():
    assert build_generation_kwargs(None) == {}
    assert build_generation_kwargs({}) == {}
    assert build_generation_kwargs({"temperature": None}) == {}


def test_build_generation_kwargs_drops_nonpositive_max_tokens():
    # DSS sends 0 for "no limit"; OpenAI reads it as "emit nothing".
    assert build_generation_kwargs({"maxOutputTokens": 0}) == {}


def test_build_generation_kwargs_routes_top_k_through_extra_body():
    assert build_generation_kwargs({"topK": 40}) == {"extra_body": {"top_k": 40}}
    assert build_generation_kwargs({"top_k": 40}) == {"extra_body": {"top_k": 40}}


def test_build_response_format_json_only():
    assert build_response_format({"responseFormat": {"type": "JSON"}}) == {"type": "json_object"}
    assert build_response_format({"responseFormat": "json_object"}) == {"type": "json_object"}
    assert build_response_format({"responseFormat": {"type": "TEXT"}}) is None
    assert build_response_format({}) is None
    assert build_response_format(None) is None


def test_build_chat_kwargs_carries_generation_settings():
    kwargs = build_chat_kwargs(
        "llama3.1",
        {"messages": [{"role": "user", "content": "hi"}]},
        {"temperature": 0, "maxOutputTokens": 256},
    )
    assert kwargs["temperature"] == 0
    assert kwargs["max_tokens"] == 256


def test_build_tools_from_settings():
    settings = {
        "tools": [{
            "function": {
                "name": "lookup",
                "description": "Find rows",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
    }
    out = build_tools(settings)
    assert out[0]["function"]["name"] == "lookup"


def test_build_tools_returns_none_without_tools():
    assert build_tools(None) is None
    assert build_tools({}) is None
    assert build_tools({"tools": []}) is None


def test_build_tool_choice_auto():
    assert build_tool_choice({"toolChoice": {"type": "AUTO"}}) == "auto"


def test_build_tool_choice_variants():
    assert build_tool_choice({"toolChoice": {"type": "NONE"}}) == "none"
    assert build_tool_choice({"toolChoice": {"type": "REQUIRED"}}) == "required"
    assert build_tool_choice({"toolChoice": {"type": "ANY"}}) == "required"
    assert build_tool_choice({"toolChoice": {"type": "FUNCTION", "name": "lookup"}}) == {
        "type": "function",
        "function": {"name": "lookup"},
    }
    assert build_tool_choice({"toolChoice": {"type": "FUNCTION"}}) == "auto"
    assert build_tool_choice({}) is None


def test_build_chat_kwargs_omits_tools_when_absent():
    kwargs = build_chat_kwargs("llama3.1", {"messages": [{"role": "user", "content": "hi"}]}, None)
    assert kwargs == {
        "model": "llama3.1",
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_build_chat_kwargs_includes_tools_and_choice():
    settings = {
        "tools": [{"function": {"name": "lookup"}}],
        "toolChoice": {"type": "REQUIRED"},
    }
    kwargs = build_chat_kwargs("llama3.1", {"messages": []}, settings)
    assert kwargs["tools"][0]["function"]["name"] == "lookup"
    assert kwargs["tool_choice"] == "required"


def test_validate_tool_calls_accepts_declared_json_object():
    calls = [{
        "id": "c1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"q":"rows"}'},
    }]
    settings = {"tools": [{"function": {"name": "lookup"}}]}
    assert validate_tool_calls(calls, settings) == calls


@pytest.mark.parametrize(
    "call",
    [
        {"function": {"name": "delete_all", "arguments": "{}"}},
        {"function": {"name": "lookup", "arguments": "not-json"}},
        {"function": {"name": "lookup", "arguments": "[]"}},
    ],
)
def test_validate_tool_calls_rejects_undeclared_or_invalid_calls(call):
    settings = {"tools": [{"function": {"name": "lookup"}}]}
    with pytest.raises(ValueError):
        validate_tool_calls([call], settings)
