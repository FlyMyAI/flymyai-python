import asyncio
from decimal import Decimal

import httpx
import pytest

from flymyai.agents import AsyncAgentClient, RunPrice
from tests.test_mcp_teams import client_with, response

RUN = "aaa-bbbb-ccc"
# The backend's answer (docs/protocols/RUN_USAGE.md in fma/core/backend): money as
# strings with 12 decimal places, rows newest first.
PRICE = {
    "id": RUN,
    "run_seq": 3,
    "billing_settled": False,
    "tool_calls": [{
        "action_name": "mcpserver--crm--lookup",
        "price": "0.010000000000",
        "settled": False,
    }],
    "llm_usage": [
        {
            "provider": "openai",
            "model": "gpt-5.6-terra",
            "purpose": None,
            "input_tokens": 1000,
            "cache_read_tokens": 4000,
            "cache_write_tokens": 0,
            "total_input_tokens": 5000,
            "output_tokens": 300,
            "reasoning_tokens": 120,
            "price": "0.006500000000",
            "settled": True,
        },
        {
            "provider": "openai",
            "model": "gpt-5.6-terra",
            "purpose": "completion_review",
            "input_tokens": 500,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_input_tokens": 500,
            "output_tokens": 50,
            "price": "0.001500000000",
        },
    ],
    "total_price": "0.018000000000",
    "model_inference": [{
        "prediction_id": "p-1",
        "model": "nano-banana-2",
        "price": None,
        "settled": False,
    }],
    "total_with_inference": "0.018000000000",
    "future_field": "kept",
}


def _handler(seen):
    def answer(request):
        seen.append(request)
        if request.url.path.endswith("/executions/prices/"):
            return response([PRICE, {**PRICE, "id": "zzz-yyyy-xxx"}])
        return response(PRICE)

    return answer


def _check(price: RunPrice) -> None:
    assert price.id == RUN
    assert price.total_price == Decimal("0.018")
    assert price.billing_settled is False
    assert price.tool_calls[0].price == Decimal("0.01")
    assert price.tool_calls[0].settled is False
    own, review = price.llm_usage
    assert (own.purpose, own.total_input_tokens, own.reasoning_tokens) == (
        None,
        5000,
        120,
    )
    assert own.settled is True
    # An older backend without the newer fields still reads.
    assert (review.reasoning_tokens, review.settled) == (0, False)
    assert price.model_inference[0].price is None
    assert price.total_with_inference == Decimal("0.018")
    assert price.model_extra == {"future_field": "kept"}


def test_prices_reads_one_runs_usage_and_cost():
    seen = []
    client = client_with(_handler(seen))
    _check(client.runs.prices(RUN))
    assert [(r.method, r.url.path) for r in seen] == [
        ("GET", f"/api/v1/agents/executions/{RUN}/prices/")
    ]


def test_list_prices_by_ids_or_newest():
    seen = []
    client = client_with(_handler(seen))
    by_ids = client.runs.list_prices(ids=[RUN, "zzz-yyyy-xxx"])
    newest = client.runs.list_prices(limit=5)
    default = client.runs.list_prices()
    assert [p.id for p in by_ids] == [RUN, "zzz-yyyy-xxx"]
    _check(newest[0])
    assert len(default) == 2
    assert [dict(r.url.params) for r in seen] == [
        {"ids": f"{RUN},zzz-yyyy-xxx"},
        {"limit": "5"},
        {},
    ]
    assert {r.url.path for r in seen} == {"/api/v1/agents/executions/prices/"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ids": []},
        {"ids": "aaa-bbbb-ccc"},
        {"ids": [""]},
        {"ids": [str(n) for n in range(201)]},
        {"limit": 0},
        {"limit": 201},
        {"limit": True},
        {"limit": "5"},
    ],
)
def test_list_prices_refuses_bad_arguments_before_calling(kwargs):
    seen = []
    client = client_with(_handler(seen))
    with pytest.raises(ValueError):
        client.runs.list_prices(**kwargs)
    assert seen == []


def test_async_prices_match_the_sync_client():
    seen = []

    async def run():
        async with AsyncAgentClient(
            api_key="personal-key", base_url="https://example.test"
        ) as client:
            await client._http.aclose()
            client._http = httpx.AsyncClient(
                base_url="https://example.test",
                headers={"X-API-KEY": "personal-key"},
                transport=httpx.MockTransport(_handler(seen)),
            )
            return (
                await client.runs.prices(RUN),
                await client.runs.list_prices(ids=[RUN]),
            )

    one, many = asyncio.run(run())
    _check(one)
    _check(many[0])
    assert [(r.url.path, dict(r.url.params)) for r in seen] == [
        (f"/api/v1/agents/executions/{RUN}/prices/", {}),
        ("/api/v1/agents/executions/prices/", {"ids": RUN}),
    ]
