"""The MCP tool-pending digest plumbing the Session wake runs on."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from mistralai_vibe_local_harness.vibe._mcp_models import (
    MCPAuthorizationRef,
    MCPDescriptorCachePolicy,
    MCPToolPendingDigestParams,
    ResolvedMCPServerConfig,
)
from mistralai_vibe_local_harness.vibe._mcp_pool import MCPStdioPool
from mistralai_vibe_local_harness.vibe._mcp_runtime import MCPRuntime
from mistralai_vibe_local_harness.vibe._mcp_transport import tool_pending_digest_binding


def _stdio_server(name: str = "ii-bridge") -> ResolvedMCPServerConfig:
    return ResolvedMCPServerConfig(
        name=name,
        transport="stdio",
        command="bridge",
        args=("serve",),
        authorization=MCPAuthorizationRef(name, "fingerprint", "none", "revision"),
    )


@pytest.mark.asyncio
async def test_a_digest_binding_delivers_its_params_and_swallows_sink_errors() -> None:
    """*Prepare*: A sink that records and one that raises.
    *Do*: Run the binding's handler for both.
    *Assert*: The sink receives the configured server name and the digest
        text, and a failing sink never escapes into the transport's loop.
    """
    # Prepare
    delivered: list[tuple[str, str]] = []

    async def record(server_name: str, digest: str) -> None:
        delivered.append((server_name, digest))

    async def explode(server_name: str, digest: str) -> None:
        raise RuntimeError("the session is being built")

    # Do
    await tool_pending_digest_binding("ii-bridge", record).handler(
        MCPToolPendingDigestParams(text="check: ii_inbox (2)")
    )
    await tool_pending_digest_binding("ii-bridge", explode).handler(
        MCPToolPendingDigestParams(text="check: ii_inbox (2)")
    )

    # Assert
    assert delivered == [("ii-bridge", "check: ii_inbox (2)")]


@pytest.mark.asyncio
async def test_one_digest_reaches_every_hold_on_the_shared_server_process() -> None:
    """*Prepare*: One pool, one server, two leases holding it - one with a sink
    recording deliveries, one without.
    *Do*: Dispatch one digest through the hold both leases share.
    *Assert*: Both are offered the digest - a sink-less hold is skipped, not a
    crash - and a lease that released sees later digests no longer.
    """
    # Prepare
    delivered: list[tuple[str, str]] = []
    pool = MCPStdioPool(cast(Any, None))
    server = _stdio_server()
    subscriber = pool.lease()
    holding = pool.lease(cast(Any, _recording_sink(delivered)))
    await pool.acquire(subscriber, server)
    await pool.acquire(holding, server)
    connection = await pool.acquire(holding, server)
    dispatch = connection._digest_sink

    # Do
    await cast(Any, dispatch)("a-server", "check: ii_inbox (2)")
    await pool.release(subscriber, server)
    await cast(Any, dispatch)("a-server", "check: comms_inbox (1)")

    # Assert
    assert delivered == [
        ("a-server", "check: ii_inbox (2)"),
        ("a-server", "check: comms_inbox (1)"),
    ]
    await pool.release_all(holding)
    await pool.aclose()


def _recording_sink(delivered: list[tuple[str, str]]) -> Any:
    async def record(server_name: str, digest: str) -> None:
        delivered.append((server_name, digest))

    return record


@pytest.mark.asyncio
@pytest.mark.parametrize("with_sink,expected", [(True, True), (False, False)])
async def test_stdio_discovery_holds_the_server_only_while_a_digest_sink_is_bound(
    tmp_path: Path, with_sink: bool, expected: bool
) -> None:
    """*Prepare*: One Runtime with a stubbed stdio lease, digest sink optional.
    *Do*: Discover the stdio server's tools.
    *Assert*: The list keeps the server's process - so its digest pushes
        remain reachable - exactly while a sink is bound; without one, the
        process is released with the answer as before.
    """
    # Prepare
    persistence: list[bool] = []

    class StubLease:
        async def list_tools(
            self, server: ResolvedMCPServerConfig, *, persistent: bool = False
        ) -> tuple[Any, ...]:
            persistence.append(persistent)
            return ()

        async def aclose(self) -> None: ...

    runtime = MCPRuntime(
        cache_root=tmp_path,
        cache_policy=MCPDescriptorCachePolicy(),
        authorization_provider=cast(Any, None),
        tool_digest_sink=(_recording_sink([]) if with_sink else None),
    )
    lease = StubLease()
    runtime._stdio_lease = lambda: lease  # type: ignore[method-assign]

    # Do
    await runtime._discover(_stdio_server(), cast(Any, None))
    await runtime.aclose()

    # Assert
    assert persistence == [expected]
