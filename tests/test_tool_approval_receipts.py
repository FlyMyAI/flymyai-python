"""Client protocol tests; backend authority has separate integration coverage."""

import pytest

from tests.test_apps_composition import run_call
from tests.test_mcp_teams import response

OPERATION = "0957bd6e-901e-4324-b004-ebcd432d829a"


@pytest.mark.parametrize("async_client", [False, True])
def test_pending_receipt_is_preserved_and_reconciliation_does_not_resubmit(async_client):
    receipt = {"status": "awaiting_approval", "approval": {
        "operation_id": OPERATION, "status": "pending",
    }}
    current = {"operation_id": OPERATION, "status": "unknown", "attempts": 1,
               "approval": {"status": "consumed"}}
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.method == "POST":
            assert request.headers["Idempotency-Key"] == "one-logical-call"
            return response(receipt, status=202)
        return response(current)

    assert run_call(async_client, handler, "tools", "call", 17,
                    action="read", idempotency_key="one-logical-call") == receipt
    assert run_call(async_client, handler, "tools", "get_operation", OPERATION) == current
    assert calls == [("POST", "/api/v1/agents/tools/17/call/"),
                     ("GET", f"/api/v1/agents/tool-operations/{OPERATION}/")]
    with pytest.raises(ValueError, match="UUID"):
        run_call(async_client, handler, "tools", "get_operation", "../approve")
    assert len(calls) == 2
