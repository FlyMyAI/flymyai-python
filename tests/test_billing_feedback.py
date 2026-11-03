import asyncio
import json
from decimal import Decimal

import httpx
import pytest

from flymyai import (
    FeedbackRateLimitedError,
    FlyMyAIAgentError,
    InsufficientBalanceError,
    InvalidAmountError,
    RateLimitedError,
)
from flymyai.agents import (
    FEEDBACK_CATEGORIES,
    AsyncBilling,
    AsyncFeedback,
    Billing,
    Feedback,
    FeedbackReceipt,
    TopUpLink,
)
from flymyai.agents._client import AsyncAgentClient
from tests.test_mcp_teams import client_with, response

LINK = {
    "schema": "flymy.top-up-link.v1",
    "checkout_url": "https://checkout.example.test/pay/cs_1",
    "session_id": "5a0e4b4c-2f7b-4f0e-9f52-3c3f1f6f2a10",
    "amount_usd": "7.25",
    "minimum_usd": "5.00",
    "maximum_usd": "1000.00",
    "expires_at": "2026-11-04T00:00:00Z",
}
RECEIPT = {
    "schema": "flymy.feedback.v1",
    "feedback_id": "0b7f6c1e-8d3a-4f55-a2c9-1e2d3c4b5a69",
    "status": "received",
    "deduplicated": False,
}


def test_a_top_up_link_sends_the_exact_amount_once_without_a_key():
    seen = []

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/agents/billing/top-up-link/"
        assert "Idempotency-Key" not in request.headers
        seen.append(json.loads(request.content))
        return response(LINK, 201)

    client = client_with(handler)
    link = client.billing.create_top_up_link(7.25)
    assert isinstance(link, TopUpLink)
    assert link.checkout_url == LINK["checkout_url"]
    assert link.amount_usd == Decimal("7.25")
    assert link.minimum_usd == Decimal("5.00")
    assert link.maximum_usd == Decimal("1000.00")
    assert link.schema_version == "flymy.top-up-link.v1"
    # a field added later in v1 is kept
    assert link.model_extra == {"expires_at": "2026-11-04T00:00:00Z"}

    for amount in (Decimal("10"), 10, "12.50", 0.1):
        client.billing.create_top_up_link(amount)
    assert seen == [
        {"amount_usd": "7.25"},
        {"amount_usd": "10"},
        {"amount_usd": "10"},
        {"amount_usd": "12.50"},
        {"amount_usd": "0.1"},
    ]


@pytest.mark.parametrize("amount", [0, -1, "ten", "", True, None, float("nan"), [5]])
def test_a_top_up_amount_that_is_not_a_positive_number_is_refused_locally(amount):
    def handler(request):
        raise AssertionError("nothing is sent")

    with pytest.raises(ValueError):
        client_with(handler).billing.create_top_up_link(amount)


def test_an_amount_outside_the_limits_names_them():
    body = {
        "code": "invalid_amount",
        "detail": "The amount must be from $5.00 to $1000.00.",
        "minimum_usd": "5.00",
        "maximum_usd": None,
    }
    client = client_with(lambda request: response(body, 400))
    with pytest.raises(InvalidAmountError) as raised:
        client.billing.create_top_up_link(1)
    assert raised.value.status_code == 400
    assert raised.value.minimum_usd == Decimal("5.00")
    assert raised.value.maximum_usd is None
    assert str(raised.value) == "HTTP 400: The amount must be from $5.00 to $1000.00."
    assert isinstance(raised.value, FlyMyAIAgentError)


def test_too_many_links_wait_the_time_the_answer_gives():
    body = {
        "code": "rate_limited",
        "detail": "Too many payment links.",
        "retry_after_seconds": 60,
    }
    client = client_with(lambda request: response(body, 429))
    with pytest.raises(RateLimitedError) as raised:
        client.billing.create_top_up_link(5)
    assert raised.value.code == "rate_limited"
    assert raised.value.retry_after_seconds == 60
    assert not isinstance(raised.value, FeedbackRateLimitedError)


def test_an_empty_balance_names_the_shortfall_and_the_top_up_page():
    body = {
        "code": "insufficient_balance",
        "detail": (
            "Insufficient balance: this call holds up to $0.50 before it runs. Top up"
            " your balance to continue."
        ),
        "needed_usd": "0.50",
        "shortfall_usd": "0.35",
        "top_up_url": "https://app.example.test/billing?top_up=5.00",
    }

    def handler(request):
        return response(body, 402)

    client = client_with(handler)
    with pytest.raises(InsufficientBalanceError) as raised:
        client.feedback.send("anything")
    assert raised.value.status_code == 402
    assert raised.value.shortfall_usd == Decimal("0.35")
    assert raised.value.top_up_url == "https://app.example.test/billing?top_up=5.00"
    assert str(raised.value) == f"HTTP 402: {body['detail']}"

    # an older answer without them: the same error, with nothing to show
    del body["shortfall_usd"], body["top_up_url"]
    with pytest.raises(InsufficientBalanceError) as raised:
        client.feedback.send("anything")
    assert raised.value.shortfall_usd is None and raised.value.top_up_url is None

    # a page that is not https is never handed on
    body["top_up_url"] = "http://app.example.test/billing"
    with pytest.raises(InsufficientBalanceError) as raised:
        client.feedback.send("anything")
    assert raised.value.top_up_url is None


def test_other_payment_refusals_stay_untyped():
    body = {"code": "app_budget_exhausted", "detail": "The app budget is used up."}
    client = client_with(lambda request: response(body, 402))
    with pytest.raises(FlyMyAIAgentError) as raised:
        client.feedback.send("anything")
    assert type(raised.value) is FlyMyAIAgentError


def test_feedback_sends_what_was_given_with_source_sdk():
    seen = []

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/agents/feedback/"
        assert "Idempotency-Key" not in request.headers
        seen.append(json.loads(request.content))
        return response(dict(RECEIPT, deduplicated=len(seen) > 1), 202)

    client = client_with(handler)
    first = client.feedback.send(
        "firecrawl_scrape answered 502 three times for a public page.",
        category="bug",
        execution_id="abc-defg-hij",
        tool="custom--firecrawl--firecrawl_scrape",
        client="nightly-report",
    )
    again = client.feedback.send("Only the message.")
    assert isinstance(first, FeedbackReceipt)
    assert first.feedback_id == RECEIPT["feedback_id"]
    assert first.status == "received" and first.deduplicated is False
    assert again.deduplicated is True
    assert seen == [
        {
            "message": "firecrawl_scrape answered 502 three times for a public page.",
            "category": "bug",
            "execution_id": "abc-defg-hij",
            "tool": "custom--firecrawl--firecrawl_scrape",
            "client": "nightly-report",
            "source": "sdk",
        },
        {"message": "Only the message.", "source": "sdk"},
    ]


@pytest.mark.parametrize(
    "message, options",
    [
        ("", {}),
        ("   ", {}),
        ("x" * 4001, {}),
        ("ok", {"category": "complaint"}),
        ("ok", {"tool": "t" * 256}),
        ("ok", {"client": "c" * 101}),
        ("ok", {"execution_id": ""}),
        (None, {}),
    ],
)
def test_feedback_is_checked_before_it_is_sent(message, options):
    def handler(request):
        raise AssertionError("nothing is sent")

    with pytest.raises(ValueError):
        client_with(handler).feedback.send(message, **options)


def test_too_much_feedback_is_a_typed_rate_limit():
    body = {
        "code": "feedback_rate_limited",
        "detail": "Too much feedback from this account; try again later.",
    }

    def handler(request):
        return httpx.Response(
            429,
            headers={"Retry-After": "1200"},
            stream=httpx.ByteStream(json.dumps(body).encode()),
        )

    with pytest.raises(FeedbackRateLimitedError) as raised:
        client_with(handler).feedback.send("one more")
    assert isinstance(raised.value, RateLimitedError)
    assert raised.value.code == "feedback_rate_limited"
    assert raised.value.retry_after_seconds == 1200


def test_the_categories_are_the_api_ones():
    assert FEEDBACK_CATEGORIES == (
        "bug",
        "wrong_result",
        "missing_tool",
        "billing",
        "docs",
        "idea",
        "praise",
        "other",
    )


def test_the_async_client_has_the_same_billing_and_feedback():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if request.url.path.endswith("/top-up-link/"):
            return response(LINK, 201)
        return response(RECEIPT, 202)

    async def go():
        client = AsyncAgentClient(
            api_key="personal-key", base_url="https://example.test"
        )
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="https://example.test", transport=httpx.MockTransport(handler)
        )
        try:
            link = await client.billing.create_top_up_link("7.25")
            sent = await client.feedback.send("Async report.", category="docs")
            with pytest.raises(ValueError):
                await client.feedback.send("")
        finally:
            await client.close()
        return link, sent

    link, sent = asyncio.run(go())
    assert link.amount_usd == Decimal("7.25") and sent.status == "received"
    assert bodies == [
        {"amount_usd": "7.25"},
        {"message": "Async report.", "category": "docs", "source": "sdk"},
    ]


# The v1 methods stay (a frozen program keeps the calls it was written with): one may be
# added, none renamed or removed.
V1_METHODS = {Billing: {"create_top_up_link"}, Feedback: {"send"}}


@pytest.mark.parametrize(
    "resource, methods",
    [
        (Billing, V1_METHODS[Billing]),
        (AsyncBilling, V1_METHODS[Billing]),
        (Feedback, V1_METHODS[Feedback]),
        (AsyncFeedback, V1_METHODS[Feedback]),
    ],
)
def test_the_v1_methods_stay(resource, methods):
    assert methods <= {name for name in dir(resource) if not name.startswith("_")}
