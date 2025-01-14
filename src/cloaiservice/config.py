"""App configuration."""

import pathlib
from functools import lru_cache
from os import environ
from typing import Literal

import cloai
import fastapi
import pydantic
from cloai.llm import bedrock as cloai_bedrock
from fastapi import status
from openai.types import chat_model
from pydantic import BaseModel, Field


class BedrockAnthropicConfig(BaseModel):
    """Bedrock Anthropic client configuration."""

    type: Literal["bedrock-anthropic"]
    model: cloai_bedrock.ANTHROPIC_BEDROCK_MODELS
    aws_access_key: pydantic.SecretStr = pydantic.Field(
        ..., min_length=20, max_length=20
    )
    aws_secret_key: pydantic.SecretStr = pydantic.Field(
        ..., min_length=40, max_length=40
    )
    region: str

    def create_client(self) -> cloai.LargeLanguageModel:
        """Create an Anthropic client instance."""
        return cloai.LargeLanguageModel(
            client=cloai.AnthropicBedrockLlm(
                model=self.model,
                aws_access_key=self.aws_access_key.get_secret_value(),
                aws_secret_key=self.aws_secret_key.get_secret_value(),
                region=self.region,
            )
        )


class OpenAIConfig(BaseModel):
    """OpenAI client configuration."""

    type: Literal["openai"]
    model: chat_model.ChatModel | str
    api_key: pydantic.SecretStr
    base_url: str | None = Field(default=None)

    def create_client(self) -> cloai.LargeLanguageModel:
        """Create an OpenAI client instance."""
        return cloai.LargeLanguageModel(
            client=cloai.OpenAiLlm(
                model=self.model,
                api_key=self.api_key.get_secret_value(),
                base_url=self.base_url,
            )
        )


class AzureConfig(BaseModel):
    """Azure client configuration."""

    type: Literal["azure"]
    api_key: pydantic.SecretStr
    endpoint: str
    deployment: str
    api_version: str

    def create_client(self) -> cloai.LargeLanguageModel:
        """Create an Azure client instance."""
        return cloai.LargeLanguageModel(
            client=cloai.AzureLlm(
                api_key=self.api_key.get_secret_value(),
                endpoint=self.endpoint,
                deployment=self.deployment,
                api_version=self.api_version,
            )
        )


class ClientConfig(BaseModel):
    """Configuration for all clients."""

    clients: dict[str, BedrockAnthropicConfig | OpenAIConfig | AzureConfig]

    def create_clients(self) -> dict[str, cloai.LargeLanguageModel]:
        """Create all client instances."""
        return {name: config.create_client() for name, config in self.clients.items()}


class Config:
    """Service configuration."""

    def __init__(self, clients: dict[str, cloai.LargeLanguageModel]) -> None:
        """Initialize service configuration."""
        self.clients = clients


@lru_cache()
def get_config() -> Config:
    """Load config from environment or config file.

    Precedence:
    1. CONFIG_JSON environment variable (containing JSON string)
    2. File specified by CONFIG_PATH environment variable
    3. config.json in current directory
    """
    # First try loading from CONFIG_JSON environment variable
    config_path = pathlib.Path(environ.get("CONFIG_PATH", "config.json"))
    config_json = environ.get("CONFIG_JSON")
    if not config_json:
        try:
            config_json = config_path.read_text()
        except (FileNotFoundError, IsADirectoryError):
            raise fastapi.HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Config file not found and CONFIG_JSON environment variable not set.",
            )

    try:
        client_config = ClientConfig.model_validate_json(config_json)
    except pydantic.ValidationError:
        raise fastapi.HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "Invalid model JSON configuration."
        )
    return Config(clients=client_config.create_clients())
