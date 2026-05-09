from __future__ import annotations

from typing import Any


class GDRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        endpoint: str,
        payload: dict[str, Any],
        status: int | None = None,
        response_text: str = "",
    ) -> None:
        super().__init__(message)
        self.endpoint = endpoint
        self.payload = payload
        self.status = status
        self.response_text = response_text
