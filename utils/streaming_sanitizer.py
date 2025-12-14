"""
Streaming sanitizer for real-time token sanitization.
Handles partial tag detection and removal during streaming.
"""

class StreamingSanitizer:
    """Sanitizes tokens in real-time during streaming.
    
    Detects tags like SEARCH:, GOOGLE:, RECALL:, [search_id: N] and removes them
    along with content until the next newline or period.
    """
    
    TAG_PATTERNS = [
        "SEARCH:", "GOOGLE:", "WEATHER:", "REDDIT:", 
        "RECALL:", "WIKI:", "WIKIPEDIA:", "[search_id:"
    ]

    MAX_BUFFER_SIZE = 12
    
    def __init__(self):
        self.buffer = ""
        self.in_discard_mode = False
    
    def _is_potential_tag_prefix(self, text: str) -> bool:
        """Check if text could be the start of a known tag pattern."""
        if not text:
            return False
        
        for tag in self.TAG_PATTERNS:
            if tag.startswith(text):
                return True
        return False
    
    def process_token(self, token: str) -> str:
        """Process a single token and return sanitized output."""
        if self.in_discard_mode:
            self.buffer += token
            
            # Check if we hit a delimiter to stop discarding
            if '\n' in self.buffer or '. ' in self.buffer:
                if '\n' in self.buffer:
                    parts = self.buffer.split('\n', 1)
                    self.buffer = parts[1] if len(parts) > 1 else ""
                elif '. ' in self.buffer:
                    parts = self.buffer.split('. ', 1)
                    self.buffer = parts[1] if len(parts) > 1 else ""
                
                self.in_discard_mode = False
                
                # Process what's after the delimiter
                if self.buffer:
                    result = self.buffer
                    self.buffer = ""
                    return result
            
            return "" 
        
        # Not in discard mode, accumulate and check
        self.buffer += token
        
        # Check for complete tag patterns (case-sensitive)
        for pattern in self.TAG_PATTERNS:
            if pattern in self.buffer:
                idx = self.buffer.find(pattern)
                pre_tag = self.buffer[:idx]
                
                # Enter discard mode
                self.in_discard_mode = True
                self.buffer = self.buffer[idx + len(pattern):]
                
                return pre_tag
        
        # Check for potential tag prefixes at the end of buffer
        # Start from longest possible suffix to shortest
        # We want to find the longest suffix that's a valid tag prefix
        for i in range(len(self.buffer), 0, -1):
            suffix = self.buffer[-i:]
            if self._is_potential_tag_prefix(suffix):
                if i == len(self.buffer):
                    if len(self.buffer) > self.MAX_BUFFER_SIZE:
                        result = self.buffer
                        self.buffer = ""
                        return result
                    return ""
                else:
                    safe_part = self.buffer[:-i]
                    self.buffer = suffix
                    return safe_part
        
        # No tag prefix found - safe to output everything
        result = self.buffer
        self.buffer = ""
        return result
    
    def flush(self) -> str:
        """Flush any remaining buffered content.
        
        Returns:
            Any remaining buffered content
        """
        if self.in_discard_mode:
            self.buffer = ""
            self.in_discard_mode = False
            return ""
        
        result = self.buffer
        self.buffer = ""
        return result
    
    def reset(self):
        """Reset the sanitizer state."""
        self.buffer = ""
        self.in_discard_mode = False
