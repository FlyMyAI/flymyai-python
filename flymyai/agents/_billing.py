"""Adding funds to the account (``flymy.top-up-link.v1``).

``client.billing.create_top_up_link(amount_usd)`` makes a one-time payment link for the
account owner. Nothing is charged until the owner pays, and an unpaid link expires, so
the call takes no idempotency key. Show ``checkout_url`` to the owner; a program never
pays it. Unknown answer fields are kept on the models (``extra="allow"``).
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

_TOP_UP_LINK_PATH = "/api/v1/agents/billing/top-up-link/"

Amount = Union[Decimal, int, float, str]


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


def top_up_amount(amount_usd: Amount) -> str:
    """The amount as the exact decimal string the API parses (never a binary float)."""
    if isinstance(amount_usd, bool) or not isinstance(
        amount_usd, (Decimal, int, float, str)
    ):
        raise ValueError(
            "amount_usd must be a number of US dollars, such as 10 or 7.25."
        )
    try:
        amount = Decimal(str(amount_usd).strip())
    except (InvalidOperation, ValueError):
        raise ValueError(
            "amount_usd must be a number of US dollars, such as 10 or 7.25."
        ) from None
    if not amount.is_finite() or amount <= 0:
        raise ValueError("amount_usd must be more than zero.")
    return format(amount, "f")


def _body(amount_usd: Amount) -> dict:
    return {"amount_usd": top_up_amount(amount_usd)}


class Billing:
    """``client.billing``: add funds to the account."""

    def __init__(self, client: Any) -> None:
        self._c = client

    def create_top_up_link(self, amount_usd: Amount) -> TopUpLink:
        """A one-time payment link that adds ``amount_usd`` US dollars to the balance.

        Raises :class:`~flymyai.agents.InvalidAmountError` (with ``minimum_usd`` and
        ``maximum_usd``) for an amount outside the account's limits. After an
        :class:`~flymyai.agents.InsufficientBalanceError`, ask for at least its
        ``shortfall_usd``."""
        return TopUpLink.model_validate(
            self._c._request("POST", _TOP_UP_LINK_PATH, json=_body(amount_usd))
        )


class AsyncBilling:
    """``client.billing`` of :class:`AsyncAgentClient`."""

    def __init__(self, client: Any) -> None:
        self._c = client

    async def create_top_up_link(self, amount_usd: Amount) -> TopUpLink:
        return TopUpLink.model_validate(
            await self._c._request("POST", _TOP_UP_LINK_PATH, json=_body(amount_usd))
        )
