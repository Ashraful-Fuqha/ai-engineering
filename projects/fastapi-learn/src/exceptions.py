class ApiClientError(Exception):
    """Base exception for all external API client failures."""
    pass


class ApiTimeoutError(ApiClientError):
    """Raised when an external request exceeds the configured deadline."""
    pass


class ApiResponseError(ApiClientError):
    """Raised when the upstream service returns a non-2xx status code."""
    def __init__(self, status_code: int, detail: str = ""):
        super().__init__(f"Upstream API returned status {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class ApiParseError(ApiClientError):
    """Raised when response is malformed (invalid JSON or failed schema validation)."""
    pass
