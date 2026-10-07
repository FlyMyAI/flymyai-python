import pytest
from tests.test_apps_composition import run_call
from tests.test_mcp_teams import response


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_sandbox_scope_key_and_one_dispatch(asynchronous):
    seen = []
    def handler(request):
        seen.append(request)
        return response({"output": {"sandbox": "lease", "status": "pending"}})
    result = run_call(asynchronous, handler, "sandboxes", "call", "sandbox_create", app="me/browser", resource="computer.tool", arguments={"ttl_seconds": 120}, idempotency_key="stable")
    assert result["output"]["sandbox"] == "lease" and len(seen) == 1
    assert seen[0].headers["Idempotency-Key"] == "stable"
    assert seen[0].url.path.endswith("/sandbox/sandbox_create/")
    with pytest.raises(ValueError):
        run_call(asynchronous, handler, "sandboxes", "call", "sandbox_create", app="me/browser", resource="computer.tool")
    assert len(seen) == 1
