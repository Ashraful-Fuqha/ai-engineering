import re
from typing import Any

# Sensitive key patterns to redact (excludes LLM token metrics like prompt_tokens)
SENSITIVE_KEY_PATTERNS = re.compile(
    r"(api[-_]?key|authorization|bearer|secret|password|credential|access[-_]?token|auth[-_]?token)",
    re.IGNORECASE,
)

# Regex to redact API keys inside URLs like ?key=AIzaSy...
URL_KEY_PATTERN = re.compile(r"([?&]key=)([^&]+)", re.IGNORECASE)


def redact_text(text: str) -> str:
    """Mask sensitive keys appearing inside strings or URLs."""
    return URL_KEY_PATTERN.sub(r"\1[REDACTED]", text)


def redact_sensitive_data(obj: Any) -> Any:
    """
    Recursively sanitize dictionaries, lists, and primitives,
    masking any credential or authorization tokens.
    """
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for k, v in obj.items():
            if SENSITIVE_KEY_PATTERNS.search(k):
                cleaned[k] = "[REDACTED]"
            elif isinstance(v, (dict, list)):
                cleaned[k] = redact_sensitive_data(v)
            elif isinstance(v, str):
                cleaned[k] = redact_text(v)
            else:
                cleaned[k] = v
        return cleaned
    
    if isinstance(obj, list):
        return [redact_sensitive_data(item) for item in obj]
    
    if isinstance(obj, str):
        return redact_text(obj)
    
    return obj
