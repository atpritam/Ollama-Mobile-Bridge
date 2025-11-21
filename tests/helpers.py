import re
import json

def assert_sse_event(body, event_type, **expected_data):
    """
    Assert that an SSE event with the given type and expected data exists in the body.
    Checks all occurrences of the event type.
    """
    pattern = re.compile(rf'event: {event_type}\ndata: ({{.*?}})\n\n')
    
    found_match = False
    for match in pattern.finditer(body):
        event_data_str = match.group(1)
        try:
            data = json.loads(event_data_str)
            all_keys_match = True
            for key, value in expected_data.items():
                if key not in data or data[key] != value:
                    all_keys_match = False
                    break
            if all_keys_match:
                found_match = True
                break
        except json.JSONDecodeError:
            pass
            
    assert found_match, f"No '{event_type}' event found with all expected data: {expected_data} in SSE body:\n{body}"

def assert_search_performed(response_data):
    """Assert search metadata in response."""
    assert "search_performed" in response_data and response_data["search_performed"], "Search was not marked as performed."
    assert "search_id" in response_data, "'search_id' is missing from the response."
    assert "search_type" in response_data, "'search_type' is missing from the response."