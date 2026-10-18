"""
Simple test to validate MCP auth header priority behavior.

Validates that:
1. auth_value is not required in config.yaml
2. Server-specific headers (x-mcp-server-name-authorization) take precedence over config auth_value
"""

import pytest
from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.types.mcp import MCPAuth, MCPTransport, MCPSpecVersion
from litellm.types.mcp_server.mcp_server_manager import MCPServer


@pytest.mark.asyncio
async def test_mcp_server_works_without_config_auth_value():
    """
    Test that MCP servers work without auth_value in config when headers are provided.
    This validates that auth_value is truly optional in config.yaml.
    """
    # Create a server WITHOUT config auth_value
    server_without_config_auth = MCPServer(
        server_id="test-server-no-config",
        name="Test MCP Server No Config Auth",
        server_name="test_server_no_config",
        alias="test_no_config",
        url="https://api.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.authorization,
        authentication_token=None,  # No config auth
    )

    manager = MCPServerManager()

    # Test that it works with only header auth
    client = await manager._create_mcp_client(
        server=server_without_config_auth,
        mcp_auth_header="Bearer token_from_header_only",
    )

    # Verify header token is used
    assert client._mcp_auth_value == "Bearer token_from_header_only"
    assert client.auth_type == MCPAuth.authorization


@pytest.mark.parametrize("token_key", ["authentication_token", "auth_value"])
async def test_mcp_server_config_auth_value_header_used(token_key, config_only_mcp_manager_factory):
    """Ensure the configured auth token is emitted as the upstream Authorization header.

    The token is resolved through the v2 credential resolver and rides on the client's
    httpx.Auth, so assert the header it writes onto the request rather than the (now
    credential-free) _get_auth_headers() dict.
    """
    import httpx2

    from litellm.proxy._experimental.mcp_server.outbound_credentials.httpx_auth import (
        StaticHeaderAuth,
    )

    config = {
        "test_server": {
            "url": "https://api.example.com/mcp",
            "transport": "http",
            "auth_type": "bearer_token",
            token_key: "example_token",
        }
    }

    manager = config_only_mcp_manager_factory()
    await manager.load_servers_from_config(config)

    server = next(iter(manager.config_mcp_servers.values()))
    client = await manager._create_mcp_client(server)

    assert isinstance(client._resolved_auth, StaticHeaderAuth)
    emitted = next(client._resolved_auth.auth_flow(httpx2.Request("POST", server.url)))
    assert emitted.headers["Authorization"] == "Bearer example_token"
    assert client.auth_type == MCPAuth.bearer_token


@pytest.mark.asyncio
async def test_mcp_server_byok_resolves_per_request_alias_authorization_header():
    """Ensure BYOK/static-auth servers without stored DB credentials resolve per-request
    x-mcp-{alias}-authorization headers passed via raw_headers (#46664).
    """
    byok_server = MCPServer(
        server_id="test-byok-server",
        name="Test BYOK Server",
        server_name="test_byok_server",
        alias="test_byok",
        url="https://api.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.bearer_token,
        authentication_token=None,
    )

    manager = MCPServerManager()

    # Pass per-request header in raw_headers with mcp_auth_header=None
    client = await manager.create_mcp_client(
        server=byok_server,
        mcp_auth_header=None,
        raw_headers={"x-mcp-test_byok-authorization": "Bearer token_from_byok_header"},
    )

    assert client._mcp_auth_value == {"Authorization": "Bearer token_from_byok_header"}
    assert client._get_auth_headers() == {"Authorization": "Bearer token_from_byok_header"}
    assert client.auth_type == MCPAuth.bearer_token


def test_prepare_mcp_server_headers_resolves_per_request_alias_header():
    """Ensure _prepare_mcp_server_headers extracts server_auth_header from raw_headers
    when mcp_server_auth_headers and mcp_auth_header are None (#46664).
    """
    from litellm.proxy._experimental.mcp_server.operations import _prepare_mcp_server_headers

    server = MCPServer(
        server_id="test-alias-server",
        name="Test Alias Server",
        server_name="github_mcp",
        alias="github",
        url="https://api.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.bearer_token,
        authentication_token=None,
    )

    server_auth, extra = _prepare_mcp_server_headers(
        server=server,
        mcp_server_auth_headers=None,
        mcp_auth_header=None,
        oauth2_headers=None,
        raw_headers={"x-mcp-github-authorization": "Bearer gh_secret_token_123"},
    )

    assert server_auth == {"Authorization": "Bearer gh_secret_token_123"}
