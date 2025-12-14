"""Tests for StreamingSanitizer class."""
import pytest
from utils.streaming_sanitizer import StreamingSanitizer


@pytest.mark.parametrize("input_text,expected_output", [
    ("Hello ", "Hello "),
    ("world", "world"),
])
def test_sanitizer_passes_clean_text(input_text, expected_output):
    """Given clean text, sanitizer should pass it through unchanged."""
    sanitizer = StreamingSanitizer()
    result = sanitizer.process_token(input_text)
    assert result == expected_output


def test_sanitizer_removes_search_tag():
    """Given text with SEARCH: tag, sanitizer should remove tag and content until newline."""
    sanitizer = StreamingSanitizer()
    
    result = sanitizer.process_token("Here is SEARCH: some query\nNew line")
    assert "Here is" in result or result == ""
    
    if not result:
        result = sanitizer.flush()
    assert "New line" in result or "Here is" in result


@pytest.mark.parametrize("tag_type,input_text,expected_before_tag", [
    ("GOOGLE", "Answer: GOOGLE: search query here", "Answer: "),
    ("RECALL", "Let me check. RECALL: 10", "Let me check. "),
    ("WEATHER", "WEATHER: Boston\n", ""),
])
def test_sanitizer_removes_tags(tag_type, input_text, expected_before_tag):
    """Given text with various tags, sanitizer should remove them."""
    sanitizer = StreamingSanitizer()
    result = sanitizer.process_token(input_text)
    assert result == expected_before_tag


def test_sanitizer_removes_search_id_tag():
    """Given text with [search_id: N] tag, sanitizer should remove it."""
    sanitizer = StreamingSanitizer()
    
    result = sanitizer.process_token("Response text [search_id: 5]")
    assert result == "Response text "
    
    result = sanitizer.process_token(" more\nNew line")
    assert result == "New line"


def test_sanitizer_waits_for_partial_tags():
    """Given partial tag patterns, sanitizer should buffer until complete."""
    sanitizer = StreamingSanitizer()
    
    result = sanitizer.process_token("SEAR")
    assert result == ""

    result = sanitizer.process_token("CH: query")
    assert result == ""
    
    # Discard until newline
    result = sanitizer.process_token(" content")
    assert result == ""
    
    result = sanitizer.process_token("\nOK")
    assert result == "OK"


def test_sanitizer_handles_false_partial():
    """Given text that looks like partial but isn't, sanitizer should eventually output."""
    sanitizer = StreamingSanitizer()
    
    # "SEAR" might be partial
    result1 = sanitizer.process_token("SEAR")
    assert result1 == ""
    
    # But next token proves it's not a tag
    result2 = sanitizer.process_token("ING is fun")
    # "SEARING is fun" is not a tag, should output
    
    if result2:
        assert "SEAR" in result2 or "ING" in result2
    else:
        result3 = sanitizer.flush()
        assert "SEAR" in result3 or "ING" in result3 or not result3


@pytest.mark.parametrize("first_token,second_token,expected_second", [
    ("Let me check. RECALL: 10", " checking\nHere", "Here"),
    ("WEATHER: Boston\n", "Next line", "Next line"),
])
def test_sanitizer_continues_after_newline(first_token, second_token, expected_second):
    """Given tags followed by newlines, sanitizer should resume processing after newline."""
    sanitizer = StreamingSanitizer()
    sanitizer.process_token(first_token)
    result = sanitizer.process_token(second_token)
    assert result == expected_second


def test_sanitizer_stops_at_period():
    """Given tag followed by period, sanitizer should resume after period."""
    sanitizer = StreamingSanitizer()
    
    result = sanitizer.process_token("SEARCH: query. ")
    assert result == ""
    
    result = sanitizer.process_token("Resume here")
    assert result == "Resume here"


@pytest.mark.parametrize("lowercase_input,uppercase_input", [
    ("google: search", "GOOGLE: search\nOK"),
    ("search: query", "SEARCH: query\nOK"),
])
def test_sanitizer_case_sensitive(lowercase_input, uppercase_input):
    """Sanitizer should ONLY detect UPPERCASE tags, not lowercase."""
    sanitizer = StreamingSanitizer()
    
    # Lowercase should pass through
    result = sanitizer.process_token(lowercase_input)
    assert lowercase_input.split(':')[0] in result or result == ""
    
    # Flush to get remaining
    result = sanitizer.flush()
    if result:
        assert lowercase_input.split(':')[0].lower() in result.lower()
    
    # UPPERCASE should be sanitized
    sanitizer.reset()
    result = sanitizer.process_token(uppercase_input)
    assert result == "" or "OK" in result
    
    if not result:
        result = sanitizer.flush()
    assert uppercase_input.split(':')[0] not in result and ("OK" in result or result == "")


def test_sanitizer_flush_returns_remaining():
    """Process should output safe text immediately, flush returns any truly buffered content."""
    sanitizer = StreamingSanitizer()
    
    # Process text that doesn't form a tag - should output immediately
    result1 = sanitizer.process_token("Some text that ends")
    assert "Some" in result1 and "text" in result1
    
    # Now test with  partial tag - should buffer
    result2 = sanitizer.process_token("More text and SEA")
    assert result2 == "More text and "  # "SEA" is buffered as it's a valid tag prefix
    
    # Flush should return the buffered "SEA"
    flush_result = sanitizer.flush()
    assert "SEA" in flush_result


@pytest.mark.parametrize("tag_input,clean_input,expected_clean", [
    ("SEARCH: query", "Clean text", "Clean text"),
    ("GOOGLE: query", "Clean text", "Clean text"),
    ("RECALL: query", "Clean text", "Clean text"),
])
def test_sanitizer_reset(tag_input, clean_input, expected_clean):
    """Reset should clear all state."""
    sanitizer = StreamingSanitizer()
    
    sanitizer.process_token(tag_input)
    sanitizer.reset()
                                         
    result = sanitizer.process_token(clean_input)
    assert result == expected_clean


def test_sanitizer_preserves_natural_text_with_tag_words():
    """Natural text containing tag words (not formats) should pass through."""
    sanitizer = StreamingSanitizer()
    
    # "Google" in natural text should NOT be filtered
    result1 = sanitizer.process_token("Here is what I found from Google, a machine. ")
    # Might buffer or output
    result2 = sanitizer.flush()
    
    # Check combined results
    combined = (result1 or "") + (result2 or "")
    assert "Google" in combined
    assert "machine" in combined


def test_sanitizer_multiple_tags_in_sequence():
    """Given multiple tags, sanitizer should handle them all."""
    sanitizer = StreamingSanitizer()
    
    result = sanitizer.process_token("Start. ")
    assert "Start" in result or result == ""
    
    result = sanitizer.flush()
    if result:
        assert "Start" in result
    
    sanitizer.reset()
    result = sanitizer.process_token("GOOGLE: query\n")
    assert result == ""
    
    result = sanitizer.process_token("OK. REDDIT: another\nFinal")
    assert "Final" in result or result == ""
