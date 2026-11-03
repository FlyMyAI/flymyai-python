"""Adding funds to the account (``flymy.top-up-link.v1``).

``client.billing.create_top_up_link(amount_usd)`` makes a one-time payment link for the
account owner in whole US dollars, as the web app's add funds takes them. Nothing is
charged until the owner pays, and an unpaid link expires, so the call takes no
idempotency key. Show ``checkout_url`` to the owner; a program never pays it. Unknown
answer fields are kept on the models (``extra="allow"``).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

_TOP_UP_LINK_PATH = "/api/v1/agents/billing/top-up-link/"


class TopUpLink(BaseModel):
    """A one-time payment link that adds ``amount_usd`` to the balance."""

    model_config = ConfigDict(
        extra="allow", hide_input_in_errors=True, populate_by_name=True
    )

    schema_version: str = Field(default="flymy.top-up-link.v1", alias="schema")
    checkout_url: str
    session_id: str
    amount_usd: Decimal
    minimum_usd: Optional[Decimal] = None
    maximum_usd: Optional[Decimal] = None


def top_up_amount(amount_usd: int) -> str:
    """Whole US dollars as the decimal string the API parses; cents are refused here."""
    if isinstance(amount_usd, bool) or not isinstance(amount_usd, int):
        raise ValueError(
            "amount_usd must be whole US dollars as an int, for example 10."
        )
    if amount_usd < 1:
        raise ValueError("amount_usd must be at least 1 US dollar.")
    return str(amount_usd)


def _body(amount_usd: int) -> dict:
    return {"amount_usd": top_up_amount(amount_usd)}


class Billing:
    """``client.billing``: add funds to the account."""

    def __init__(self, client: Any) -> None:
        self._c = client

    def create_top_up_link(self, amount_usd: int) -> TopUpLink:
        """A one-time payment link that adds ``amount_usd`` whole US dollars to the
        balance (an ``int``, no cents).

        Raises :class:`~flymyai.agents.InvalidAmountError` (with ``minimum_usd`` and
        ``maximum_usd``) for an amount outside the account's limits. After an
        :class:`~flymyai.agents.InsufficientBalanceError`, ask for at least its
        ``shortfall_usd`` rounded up to whole dollars, or send the owner to its
        ``top_up_url``. Too many links in an hour raise
        :class:`~flymyai.agents.RateLimitedError`; its ``response_body`` names
        ``top_up_url``, the page where the owner adds funds themselves."""
        return TopUpLink.model_validate(
            self._c._request("POST", _TOP_UP_LINK_PATH, json=_body(amount_usd))
        )


class AsyncBilling:
    """``client.billing`` of :class:`AsyncAgentClient`."""

    def __init__(self, client: Any) -> None:
        self._c = client

    async def create_top_up_link(self, amount_usd: int) -> TopUpLink:
        return TopUpLink.model_validate(
            await self._c._request("POST", _TOP_UP_LINK_PATH, json=_body(amount_usd))
        )
