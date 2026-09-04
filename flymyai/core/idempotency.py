from __future__ import annotations


def idempotency_headers(idempotency_key: str) -> dict[str, str]:
    """Validate and forward one caller-owned durable operation key exactly."""

    if not isinstance(idempotency_key, str):
        raise ValueError("idempotency_key must be a string.")
    if not idempotency_key.strip():
        raise ValueError("idempotency_key must not be blank.")
    if idempotency_key != idempotency_key.strip(" "):
        raise ValueError("idempotency_key must not contain leading or trailing spaces.")
    if len(idempotency_key) > 255:
        raise ValueError("idempotency_key must contain at most 255 characters.")
    if not idempotency_key.isprintable():
        raise ValueError(
            "idempotency_key must not contain control or non-printable characters."
        )
    if not all(0x20 <= ord(character) <= 0x7E for character in idempotency_key):
        raise ValueError(
            "idempotency_key must contain only printable ASCII characters."
        )
    return {"Idempotency-Key": idempotency_key}
