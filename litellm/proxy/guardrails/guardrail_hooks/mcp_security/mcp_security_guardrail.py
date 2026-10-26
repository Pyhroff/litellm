"""
MCP Security Guardrail for LiteLLM.

Validates that MCP servers referenced in request tools are registered
on the LiteLLM gateway. Blocks or alerts when unregistered servers are found.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, Literal

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.responses.mcp.litellm_proxy_mcp_handler import (
    LITELLM_PROXY_MCP_SERVER_URL_PREFIX,
)
from litellm.types.guardrails import GuardrailEventHooks

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache


class MCPSecurityGuardrail(CustomGuardrail):
    def __init__(
        self,
        on_violation: Literal["block", "alert"] = "block",
        allow_external_servers: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        super().__init__(**kwargs)
        self.on_violation = on_violation
        self.allow_external_servers = allow_external_servers

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [GuardrailEventHooks.pre_call]

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: "DualCache",
        data: dict,
        call_type: str,
    ) -> Exception | str | dict | None:
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call) is not True:
            return data

        unregistered: Final = await self._find_unregistered_mcp_servers(data)
        if not unregistered:
            return data

        message: Final = (
            f"MCP Security: request references unregistered MCP server(s): "
            f"{', '.join(sorted(unregistered))}. "
            f"Only servers registered on this gateway are allowed."
        )

        if self.on_violation == "block":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Violated guardrail policy",
                    "guardrail": "mcp_security",
                    "unregistered_servers": sorted(unregistered),
                    "detection_message": message,
                },
            )
        else:
            verbose_proxy_logger.warning(message)

        return data

    @classmethod
    def _extract_mcp_server_info_from_tools(
        cls,
        tools: list[dict],
        allow_external_servers: bool = False,
    ) -> tuple[set[str], set[str]]:
        """Extract MCP server names and unvetted external server URLs from tools."""
        server_names: Final[set[str]] = set()
        unregistered: Final[set[str]] = set()
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") != "mcp":
                continue
            server_url = tool.get("server_url", "")
            if not isinstance(server_url, str):
                continue
            if server_url.startswith(LITELLM_PROXY_MCP_SERVER_URL_PREFIX):
                name = server_url[len(LITELLM_PROXY_MCP_SERVER_URL_PREFIX) :]
                if name:
                    server_names.add(name)
            else:
                from litellm.responses.mcp.tool_precheck import proxy_path_mcp_name

                path_name = proxy_path_mcp_name(tool)
                if path_name:
                    server_names.add(path_name)
                elif not allow_external_servers:
                    unregistered.add(server_url)
        return server_names, unregistered

    @staticmethod
    def _extract_mcp_server_names_from_tools(tools: list[dict]) -> set[str]:
        """Extract MCP server names from tools with type=mcp and litellm_proxy server_url."""
        names, _ = MCPSecurityGuardrail._extract_mcp_server_info_from_tools(
            tools, allow_external_servers=True
        )
        return names

    async def _find_unregistered_mcp_servers(self, data: Mapping[str, object]) -> frozenset[str]:
        """Check tools in data against the MCP server registry. Returns set of unregistered server names."""
        tools: Final = data.get("tools")
        if not tools or not isinstance(tools, list):
            return frozenset()

        requested_servers, external_unregistered = self._extract_mcp_server_info_from_tools(
            tools, allow_external_servers=self.allow_external_servers
        )
        if not requested_servers and not external_unregistered:
            return frozenset()

        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        async with global_mcp_server_manager.catalog.operation():
            registry: Final = global_mcp_server_manager.get_registry()
            registered_names: Final = set(registry.keys())

            unregistered_gateway = requested_servers - registered_names
            return frozenset(unregistered_gateway | external_unregistered)
