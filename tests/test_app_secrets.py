import asyncio
import json

import httpx
import pytest

from flymyai.agents import (
    AppSecret,
    AppSecretLimitError,
    AppSecretList,
    AppSecrets,
    AsyncAppSecrets,
)
from flymyai.agents._client import AsyncAgentClient
from tests.test_mcp_teams import client_with, response

URL = "/api/v1/agents/apps/secrets/"
VALUE = "placeholder-value-1"
ONE = {
    "schema": "flymy.app-secret.v1",
    "app": "denis/shop",
    "input_url": "https://app.uat.example/projects?secret_app=denis%2Fshop&secret_name=STRIPE_KEY",
    "name": "STRIPE_KEY",
    "set": True,
    "version": 1,
    "generated": False,
    "updated_at": "2026-10-07T12:00:00+00:00",
}
LISTING = {
    "schema": "flymy.app-secrets.v1",
    "app": "denis/shop",
    "input_url": "https://app.uat.example/projects?secret_app=denis%2Fshop",
    "secrets": [
        {
            **{k: v for k, v in ONE.items() if k not in ("schema", "app")},
            "declared": True,
            "used_by": ["modules.shop.with.env.STRIPE_KEY"],
        },
        {"name": "SESSION_KEY", "set": False, "declared": True, "used_by": []},
    ],
}

# v1 methods: within v1 only add
V1_METHODS = {"list", "set", "generate", "delete"}


def _seen(handler_answers):
    seen = []

    def handler(request):
        seen.append((
            request.method,
            request.url.path,
            dict(request.url.params),
            json.loads(request.content) if request.content else None,
        ))
        return handler_answers(request)

    return seen, handler


def test_values_go_in_and_never_come_out():
    def answers(request):
        if request.method == "GET":
            return response(LISTING)
        if request.method == "DELETE":
            return response({"deleted": True})
        body = json.loads(request.content)
        if "generate" in body:
            return response({**ONE, "name": "SESSION_KEY", "generated": True})
        return response(ONE)

    seen, handler = _seen(answers)
    secrets = client_with(handler).apps.secrets

    one = secrets.set("denis/shop", "STRIPE_KEY", VALUE)
    made = secrets.generate("denis/shop", "SESSION_KEY", length=48)
    again = secrets.generate("denis/shop", "SESSION_KEY")
    listing = secrets.list("denis/shop")
    gone = secrets.delete("denis/shop", "STRIPE_KEY")

    assert (
        isinstance(one, AppSecret) and one.version == 1 and one.schema_ == ONE["schema"]
    )
    assert made.generated is True and again.generated is True
    assert isinstance(listing, AppSecretList)
    assert [s.name for s in listing.secrets] == ["STRIPE_KEY", "SESSION_KEY"]
    assert listing.secrets[0].used_by == ["modules.shop.with.env.STRIPE_KEY"]
    assert listing.secrets[1].set is False
    assert gone is True
    assert listing.input_url == LISTING["input_url"]
    assert one.input_url == ONE["input_url"]
    # nothing the backend answers holds the value
    assert VALUE not in json.dumps([one.model_dump(), listing.model_dump()])
    assert seen == [
        ("PUT", URL, {}, {"app": "denis/shop", "name": "STRIPE_KEY", "value": VALUE}),
        (
            "PUT",
            URL,
            {},
            {"app": "denis/shop", "name": "SESSION_KEY", "generate": {"length": 48}},
        ),
        (
            "PUT",
            URL,
            {},
            {"app": "denis/shop", "name": "SESSION_KEY", "generate": True},
        ),
        ("GET", URL, {"app": "denis/shop"}, None),
        ("DELETE", URL, {"app": "denis/shop", "name": "STRIPE_KEY"}, None),
    ]


def test_unexpected_value_fields_never_enter_secret_models():
    extra = {"value": VALUE, "value_encrypted": "private-ciphertext"}
    _, handler = _seen(lambda request: response({**ONE, **extra}))
    one = client_with(handler).apps.secrets.set("denis/shop", "STRIPE_KEY", VALUE)
    listing = AppSecretList.model_validate({
        **LISTING,
        **extra,
        "secrets": [{**ONE, **extra}],
    })
    for result in (one, listing):
        encoded = result.model_dump_json()
        assert VALUE not in encoded
        assert "private-ciphertext" not in encoded


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.list("denis"),
        lambda s: s.list("Denis/shop"),
        lambda s: s.list("denis/shop\n"),
        lambda s: s.list("denis/" + "s" * 155),
        lambda s: s.set("denis/shop", "KEY\n", VALUE),
        lambda s: s.set("denis/shop", "1KEY", VALUE),
        lambda s: s.set("denis/shop", "BAD-NAME", VALUE),
        lambda s: s.set("denis/shop", "KEY", ""),
        lambda s: s.set("denis/shop", "KEY", "\ud800"),
        lambda s: s.set("denis/shop", "KEY", "a\x00b"),
        lambda s: s.set("denis/shop", "KEY", "a" * (8 * 1024 + 1)),
        lambda s: s.generate("denis/shop", "KEY", length=8),
        lambda s: s.generate("denis/shop", "KEY", length=True),
        lambda s: s.delete("denis/shop", "K E Y"),
    ],
)
def test_a_bad_argument_is_refused_before_any_request(call):
    def handler(request):
        raise AssertionError(f"unexpected request {request.url}")

    with pytest.raises(ValueError) as caught:
        call(client_with(handler).apps.secrets)
    assert "a\x00b" not in str(caught.value)


def test_the_limit_is_a_typed_error():
    answer = {"code": "too_many_secrets", "message": "an app holds at most 32 secrets"}
    _, handler = _seen(lambda request: response(answer, 409))
    with pytest.raises(AppSecretLimitError) as caught:
        client_with(handler).apps.secrets.set("denis/shop", "KEY", VALUE)
    assert caught.value.status_code == 409
    assert "32 secrets" in str(caught.value)


def test_the_async_client_has_the_same_secrets():
    def handler(request):
        if request.method == "GET":
            return response(LISTING)
        if request.method == "DELETE":
            return response({"deleted": False})
        return response(ONE)

    async def go():
        client = AsyncAgentClient(
            api_key="personal-key", base_url="https://example.test"
        )
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="https://example.test", transport=httpx.MockTransport(handler)
        )
        try:
            one = await client.apps.secrets.set("denis/shop", "STRIPE_KEY", VALUE)
            made = await client.apps.secrets.generate(
                "denis/shop", "SESSION_KEY", length=64
            )
            listing = await client.apps.secrets.list("denis/shop")
            gone = await client.apps.secrets.delete("denis/shop", "STRIPE_KEY")
        finally:
            await client.close()
        return one, made, listing, gone

    one, made, listing, gone = asyncio.run(go())
    assert made.input_url == ONE["input_url"]
    assert one.version == 1 and len(listing.secrets) == 2 and gone is False
    sync = {name for name in dir(AppSecrets) if not name.startswith("_")}
    assert sync == {name for name in dir(AsyncAppSecrets) if not name.startswith("_")}


@pytest.mark.parametrize("resource", [AppSecrets, AsyncAppSecrets])
def test_the_v1_methods_stay(resource):
    methods = {name for name in dir(resource) if not name.startswith("_")}
    assert V1_METHODS <= methods
