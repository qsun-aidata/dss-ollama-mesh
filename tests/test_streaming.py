from dssollamamesh.streaming import accumulate_tool_calls, finalize_tool_calls


class Fn:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class Delta:
    """Stands in for an openai ChoiceDeltaToolCall."""

    def __init__(self, index=0, id=None, function=None):
        self.index = index
        self.id = id
        self.function = function


def test_fragments_are_stitched_into_one_call():
    acc = {}
    accumulate_tool_calls(acc, [Delta(0, "call_abc", Fn(name="lookup"))])
    accumulate_tool_calls(acc, [Delta(0, None, Fn(arguments='{"q":'))])
    accumulate_tool_calls(acc, [Delta(0, None, Fn(arguments='"rows"}'))])

    assert finalize_tool_calls(acc) == [{
        "id": "call_abc",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"q":"rows"}'},
    }]


def test_split_names_are_concatenated():
    acc = {}
    accumulate_tool_calls(acc, [Delta(0, "c1", Fn(name="look"))])
    accumulate_tool_calls(acc, [Delta(0, None, Fn(name="up"))])
    assert finalize_tool_calls(acc)[0]["function"]["name"] == "lookup"


def test_parallel_calls_stay_separate_and_ordered():
    acc = {}
    accumulate_tool_calls(acc, [
        Delta(1, "c2", Fn(name="second", arguments="{}")),
        Delta(0, "c1", Fn(name="first", arguments="{}")),
    ])
    names = [tc["function"]["name"] for tc in finalize_tool_calls(acc)]
    assert names == ["first", "second"]


def test_missing_index_and_id_get_defaults():
    acc = {}
    accumulate_tool_calls(acc, [Delta(None, None, Fn(name="anon"))])
    out = finalize_tool_calls(acc)
    assert out == [{
        "id": "call_0",
        "type": "function",
        "function": {"name": "anon", "arguments": "{}"},
    }]


def test_deltas_without_tool_calls_are_ignored():
    acc = {}
    accumulate_tool_calls(acc, None)
    accumulate_tool_calls(acc, [])
    accumulate_tool_calls(acc, [Delta(0, "c1", None)])
    assert finalize_tool_calls(acc) == [{
        "id": "c1",
        "type": "function",
        "function": {"name": "", "arguments": "{}"},
    }]
