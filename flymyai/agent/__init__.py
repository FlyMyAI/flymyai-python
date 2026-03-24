"""flymyai.agent — sandbox client for running code in isolated Kubernetes pods.

Wraps the agent-sandbox REST API:
  - Sandbox: create / exec / delete a single pod
  - SandboxClient: context-manager, sync and async
  - execute_code(): one-shot helper (create → exec → delete)

Sandbox API repo: gitlab.flymy.ai/fma/core/agent-sandbox

Quick start:

    from flymyai.agent import execute_code

    result = execute_code(
        sandbox_url="http://agent-sandbox.sandboxes.svc:8080",
        code="print(2 ** 32)",
        language="python",
    )
    assert result.exit_code == 0
    print(result.stdout)   # 4294967296
"""

from flymyai.agent.sandbox import (
    SandboxClient,
    AsyncSandboxClient,
    SandboxResult,
    execute_code,
    async_execute_code,
)

__all__ = [
    "SandboxClient",
    "AsyncSandboxClient",
    "SandboxResult",
    "execute_code",
    "async_execute_code",
]
