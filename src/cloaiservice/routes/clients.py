"""List clients (each configured LLM is a client)."""

import cloai
from fastapi import APIRouter

from cloaiservice.config import get_config
from cloaiservice.models.clients import AvailableClientsResponse, ClientInfo

router = APIRouter()


@router.get("", response_model=AvailableClientsResponse)
async def list_clients() -> AvailableClientsResponse:
    """List all available LLM clients and their configurations."""
    config = get_config()
    client_info = {}

    for client_id, llm in config.clients.items():
        base_client = llm.client

        # Determine client type and extract relevant info
        if isinstance(base_client, cloai.AnthropicBedrockLlm):
            client_info[client_id] = ClientInfo(
                provider="Anthropic", model=base_client.model, type="Bedrock"
            )
        elif isinstance(base_client, cloai.OpenAiLlm):
            client_info[client_id] = ClientInfo(
                provider="OpenAI", model=base_client.model, type="OpenAI"
            )
        elif isinstance(base_client, cloai.AzureLlm):
            client_info[client_id] = ClientInfo(
                provider="OpenAI", model=base_client.model, type="Azure"
            )
        else:
            client_info[client_id] = ClientInfo(
                provider="Unknown",
                model=str(base_client.__class__.__name__),
                type="Custom",
            )

    return AvailableClientsResponse(clients=client_info)
