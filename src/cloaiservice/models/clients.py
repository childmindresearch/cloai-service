"""Client list data models."""

from pydantic import BaseModel, Field


class ClientInfo(BaseModel):
    """Information about an LLM client."""

    provider: str = Field(
        ..., description="The provider of the LLM (e.g., Anthropic, OpenAI)"
    )
    model: str = Field(..., description="The model identifier")
    type: str = Field(..., description="The type of client (Bedrock, Azure, OpenAI)")


class AvailableClientsResponse(BaseModel):
    """Response model for the /clients endpoint."""

    clients: dict[str, ClientInfo] = Field(
        ..., description="Map of client IDs to their information"
    )
