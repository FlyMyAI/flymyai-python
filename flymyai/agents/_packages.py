"""Fleet packages (``flymy.agent-package.v1``): hand agents to another account.

A package is one document - agents (one, a lead with the agents it starts, or several),
their files, the connections they need, example inputs and ``acceptance``, what success
means. The owner exports it or shares it by token; the receiver plans the import (nothing
changes; ``plan_token``), maps its own connections, applies the plan the user confirmed
(the agents are created and one run per acceptance case starts, billed to the receiver)
and reads the import report until it passed or failed. The routes live on the agents
host under ``/api/v1/agents/packages/`` (backend ``docs/protocols/FLEET_PACKAGES.md``).
Unknown answer fields are kept on the models (``extra="allow"``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from flymyai.agents._artifacts import _body, _key

_ROOT = "/api/v1/agents/packages/"
_IMPORT_ID = re.compile(r"pkgi_[A-Za-z0-9_-]{4,40}")


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)


class Package(_Model):
    """The package document (``flymy.agent-package.v1``): ``agents``, ``fleets``,
    ``requires``, ``example_inputs``, ``acceptance`` and the rest as the API sent
    it. :meth:`document` is what :meth:`Packages.plan` and :meth:`Packages.apply`
    take back."""

    name: str = ""
    agents: List[Dict[str, Any]] = []
    fleets: List[Dict[str, Any]] = []
    requires: Dict[str, Any] = {}
    acceptance: List[Dict[str, Any]] = []

    def document(self) -> Dict[str, Any]:
        """The whole document, ``schema`` included."""
        return self.model_dump()


class PackageShare(_Model):
    """A share: its ``token`` (``fpk_...``) is shown only in this answer."""

    id: int
    token: Optional[str] = None
    url: Optional[str] = None
    name: Optional[str] = None
    recipient_email: Optional[str] = None
    expires_at: Optional[str] = None


class PackageShareList(_Model):
    shares: List[Dict[str, Any]] = []


class PackagePlan(_Model):
    """What importing would create - each agent with its prompt (``system``) -, each
    connection's ``status`` (ready, mapped, platform, missing, ambiguous with
    ``candidates``, ...), the acceptance cases with their ``inputs``, earlier
    imports of the same package (``previous_imports``: applying again makes a
    second copy), ``blocking`` and the ``plan_token`` that applies exactly this
    plan. Nothing was created. Show the prompts and the inputs before applying:
    the acceptance runs use the importer's connections."""

    package: Dict[str, Any] = {}
    creates: List[Dict[str, Any]] = []
    connections: List[Dict[str, Any]] = []
    acceptance: List[Dict[str, Any]] = []
    previous_imports: List[Dict[str, Any]] = []
    warnings: List[str] = []
    blocking: bool = False
    plan_token: str = ""


class PackageImport(_Model):
    """An import's report (``flymy.agent-package-import.v1``): ``status`` (created,
    running, passed, failed), the created ``agents``, per acceptance case its
    ``run_id``, ``verdict`` and ``failures``, and ``frozen`` on pass. A frozen
    entry's ``compilation_id`` means its asynchronous build was requested;
    read :meth:`Packages.status` until ``ready`` is true or ``error`` is set.
    Acceptance ``passed`` alone does not mean the saved version is ready."""

    id: str
    status: str
    agents: List[Dict[str, Any]] = []
    cases: List[Dict[str, Any]] = []
    frozen: Dict[str, Any] = {}
    error: Optional[str] = None


class PackageImports(_Model):
    imports: List[PackageImport] = []


class FleetInstaller(_Model):
    """The account's Agent transfer agent (formerly Fleet installer), created on
    first use: in its app chat it sends your agents (a private link) and installs
    a package someone sent you."""

    agent: str
    name: Optional[str] = None
    model: Optional[str] = None
    created: Optional[bool] = None


def _source(
    document: Any, share_token: Optional[str], url: Optional[str]
) -> Dict[str, Any]:
    """Exactly one place the package comes from."""
    if isinstance(document, Package):
        document = document.document()
    given = {
        key: value
        for key, value in (
            ("document", document),
            ("share_token", share_token),
            ("url", url),
        )
        if value is not None and value != ""
    }
    if len(given) != 1:
        raise ValueError("pass exactly one of document, share_token or url")
    return given


def _export_body(
    agents: List[str],
    name: Optional[str],
    description: Optional[str],
    acceptance: Optional[List[Dict[str, Any]]],
    example_inputs: Optional[Dict[str, Dict[str, Any]]],
    **extra: Any,
) -> Dict[str, Any]:
    if isinstance(agents, str) or not agents:
        raise ValueError("agents is a list of your agent ids")
    return _body(
        agents=[str(agent) for agent in agents],
        name=name,
        description=description,
        acceptance=acceptance,
        example_inputs=example_inputs,
        **extra,
    )


def _apply_body(
    document: Any,
    share_token: Optional[str],
    url: Optional[str],
    plan_token: str,
    mappings: Optional[Dict[str, int]],
    run_acceptance: Optional[bool],
    freeze_on_pass: Optional[bool],
    acceptance_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not plan_token:
        raise ValueError("apply() takes the plan_token of the plan the user confirmed")
    return _body(
        source=_source(document, share_token, url),
        plan_token=plan_token,
        mappings=mappings,
        acceptance_inputs=acceptance_inputs,
        run_acceptance=run_acceptance,
        freeze_on_pass=freeze_on_pass,
    )


def _import_path(import_id: Optional[str]) -> str:
    if import_id is None:
        return f"{_ROOT}imports/"
    if not isinstance(import_id, str) or not _IMPORT_ID.fullmatch(import_id):
        raise ValueError("import_id must be the pkgi_... id apply() returned")
    return f"{_ROOT}imports/{import_id}/"


class Packages:
    """``client.packages``: export, share and import fleet packages."""

    def __init__(self, client: Any) -> None:
        self._c = client

    def export(
        self,
        agents: List[str],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        acceptance: Optional[List[Dict[str, Any]]] = None,
        example_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Package:
        """The package of your agents, each with the agents it starts."""
        body = _export_body(agents, name, description, acceptance, example_inputs)
        data = self._c._request("POST", f"{_ROOT}export/", json=body)
        return Package.model_validate(data)

    def share(
        self,
        agents: List[str],
        *,
        idempotency_key: str,
        recipient_email: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        acceptance: Optional[List[Dict[str, Any]]] = None,
        example_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> PackageShare:
        """A private token to the package as it is now (shown once)."""
        body = _export_body(
            agents,
            name,
            description,
            acceptance,
            example_inputs,
            recipient_email=recipient_email,
            expires_in_days=expires_in_days,
        )
        data = self._c._request(
            "POST", f"{_ROOT}shares/", json=body, headers=_key(idempotency_key)
        )
        return PackageShare.model_validate(data)

    def shares(self) -> PackageShareList:
        """Your shares, newest first (tokens are never shown again)."""
        return PackageShareList.model_validate(
            self._c._request("GET", f"{_ROOT}shares/")
        )

    def revoke_share(self, share_id: int) -> Dict[str, Any]:
        """Stop new imports through a share; imported agents stay."""
        return self._c._request("POST", f"{_ROOT}shares/{int(share_id)}/revoke/")

    def plan(
        self,
        *,
        document: Any = None,
        share_token: Optional[str] = None,
        url: Optional[str] = None,
        mappings: Optional[Dict[str, int]] = None,
        acceptance_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> PackagePlan:
        """What importing would create and which connections it needs; changes
        nothing. ``document`` is the package (a dict, its YAML/JSON text or a
        :class:`Package`)."""
        body = _body(
            source=_source(document, share_token, url),
            mappings=mappings,
            acceptance_inputs=acceptance_inputs,
        )
        return PackagePlan.model_validate(
            self._c._request("POST", f"{_ROOT}plan/", json=body)
        )

    def apply(
        self,
        *,
        plan_token: str,
        idempotency_key: str,
        document: Any = None,
        share_token: Optional[str] = None,
        url: Optional[str] = None,
        mappings: Optional[Dict[str, int]] = None,
        acceptance_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        run_acceptance: Optional[bool] = None,
        freeze_on_pass: Optional[bool] = None,
    ) -> PackageImport:
        """Import exactly the plan the user confirmed: the same source, mappings and
        ``acceptance_inputs`` plus its ``plan_token``. 409 ``stale_plan`` carries the
        new plan in ``response_body["data"]``."""
        body = _apply_body(
            document,
            share_token,
            url,
            plan_token,
            mappings,
            run_acceptance,
            freeze_on_pass,
            acceptance_inputs,
        )
        data = self._c._request(
            "POST", f"{_ROOT}apply/", json=body, headers=_key(idempotency_key)
        )
        return PackageImport.model_validate(data)

    def status(self, import_id: str) -> PackageImport:
        """The import's report; reading it grades the runs that ended."""
        return PackageImport.model_validate(
            self._c._request("GET", _import_path(import_id))
        )

    def imports(self) -> PackageImports:
        """Your 20 newest imports."""
        return PackageImports.model_validate(
            self._c._request("GET", _import_path(None))
        )

    def installer(self) -> FleetInstaller:
        """Your Fleet installer agent, created on first use."""
        return FleetInstaller.model_validate(
            self._c._request("POST", f"{_ROOT}installer/")
        )


class AsyncPackages:
    """``client.packages`` on the async client."""

    def __init__(self, client: Any) -> None:
        self._c = client

    async def export(
        self,
        agents: List[str],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        acceptance: Optional[List[Dict[str, Any]]] = None,
        example_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Package:
        body = _export_body(agents, name, description, acceptance, example_inputs)
        data = await self._c._request("POST", f"{_ROOT}export/", json=body)
        return Package.model_validate(data)

    async def share(
        self,
        agents: List[str],
        *,
        idempotency_key: str,
        recipient_email: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        acceptance: Optional[List[Dict[str, Any]]] = None,
        example_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> PackageShare:
        body = _export_body(
            agents,
            name,
            description,
            acceptance,
            example_inputs,
            recipient_email=recipient_email,
            expires_in_days=expires_in_days,
        )
        data = await self._c._request(
            "POST", f"{_ROOT}shares/", json=body, headers=_key(idempotency_key)
        )
        return PackageShare.model_validate(data)

    async def shares(self) -> PackageShareList:
        data = await self._c._request("GET", f"{_ROOT}shares/")
        return PackageShareList.model_validate(data)

    async def revoke_share(self, share_id: int) -> Dict[str, Any]:
        return await self._c._request("POST", f"{_ROOT}shares/{int(share_id)}/revoke/")

    async def plan(
        self,
        *,
        document: Any = None,
        share_token: Optional[str] = None,
        url: Optional[str] = None,
        mappings: Optional[Dict[str, int]] = None,
        acceptance_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> PackagePlan:
        body = _body(
            source=_source(document, share_token, url),
            mappings=mappings,
            acceptance_inputs=acceptance_inputs,
        )
        data = await self._c._request("POST", f"{_ROOT}plan/", json=body)
        return PackagePlan.model_validate(data)

    async def apply(
        self,
        *,
        plan_token: str,
        idempotency_key: str,
        document: Any = None,
        share_token: Optional[str] = None,
        url: Optional[str] = None,
        mappings: Optional[Dict[str, int]] = None,
        acceptance_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        run_acceptance: Optional[bool] = None,
        freeze_on_pass: Optional[bool] = None,
    ) -> PackageImport:
        body = _apply_body(
            document,
            share_token,
            url,
            plan_token,
            mappings,
            run_acceptance,
            freeze_on_pass,
            acceptance_inputs,
        )
        data = await self._c._request(
            "POST", f"{_ROOT}apply/", json=body, headers=_key(idempotency_key)
        )
        return PackageImport.model_validate(data)

    async def status(self, import_id: str) -> PackageImport:
        data = await self._c._request("GET", _import_path(import_id))
        return PackageImport.model_validate(data)

    async def imports(self) -> PackageImports:
        data = await self._c._request("GET", _import_path(None))
        return PackageImports.model_validate(data)

    async def installer(self) -> FleetInstaller:
        data = await self._c._request("POST", f"{_ROOT}installer/")
        return FleetInstaller.model_validate(data)
