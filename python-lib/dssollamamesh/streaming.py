"""Stitch OpenAI streaming tool-call deltas back into whole tool calls.

Names and JSON arguments arrive in fragments keyed by delta index, so they have to
be accumulated across events before DSS can be handed a toolCalls chunk. Kept out
of llm.py so it can be tested without a DSS runtime.
"""


def accumulate_tool_calls(acc, deltas):
    """Merge one delta's tool_calls fragments into the accumulator, keyed by index."""
    for tc in deltas or []:
        idx = getattr(tc, "index", None)
        if idx is None:
            idx = 0
        slot = acc.setdefault(idx, {"id": None, "name": "", "arguments": ""})
        if getattr(tc, "id", None):
            slot["id"] = tc.id
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        if getattr(fn, "name", None):
            slot["name"] += fn.name
        if getattr(fn, "arguments", None):
            slot["arguments"] += fn.arguments
    return acc


def finalize_tool_calls(acc):
    """Turn the accumulator into OpenAI-shaped tool calls, ordered by delta index."""
    return [
        {
            "id": slot["id"] or ("call_%d" % idx),
            "type": "function",
            "function": {
                "name": slot["name"],
                "arguments": slot["arguments"] or "{}",
            },
        }
        for idx, slot in sorted(acc.items())
    ]
