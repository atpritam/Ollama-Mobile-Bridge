import json
import pytest

from services.stream_service import StreamService


class MockStreamClient:
    def __init__(self, chunks):
        self._chunks = chunks

    async def chat(self, model, messages, stream=False, **kwargs):
        async def token_stream():
            for c in self._chunks:
                yield {"message": {"content": c}}
        return token_stream()


@pytest.mark.anyio
async def test_streamed_tokens_match_done_full_response_all_scenarios(chat_context):
    """The test collects the SSE events emitted by the
    `StreamService` and asserts the concatenation of `token` events equals
    the `full_response` field from the final `done` metadata.
    """
    scenarios = [
        {"name": "default", "client": chat_context.client},
        {"name": "regression", "client": MockStreamClient(["Hello\n", "dup"])},
    ]

    request = chat_context.request
    messages = chat_context.messages
    context = chat_context

    for scenario in scenarios:
        client = scenario["client"]

        events = []
        async for sse in StreamService.stream_with_realtime_sanitization_and_cutoff_detection(
            client=client,
            request=request,
            messages=messages,
            context=context,
            call_number=1,
        ):
            events.append(sse)

        tokens = []
        done_metadata = None

        for ev in events:
            lines = [l for l in ev.splitlines() if l.strip()]
            if not lines:
                continue

            ev_type = lines[0].split(":", 1)[1].strip()

            data_line = None
            for ln in lines:
                if ln.startswith("data:"):
                    data_line = ln[len("data:"):].strip()
                    break
            if data_line is None:
                continue

            payload = json.loads(data_line)
            if ev_type == "token":
                tokens.append(payload.get("content", ""))
            elif ev_type == "done":
                done_metadata = payload

        assembled = "".join(tokens)

        assert done_metadata is not None, f"Scenario {scenario['name']} missing final done metadata"
        assert assembled == done_metadata["full_response"], f"Mismatch in scenario {scenario['name']}"
