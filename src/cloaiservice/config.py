"""App configuration."""

import pathlib
from functools import lru_cache
from os import environ
from typing import Literal

import cloai
from pydantic import BaseModel, Field


class AnthropicConfig(BaseModel):
    """Anthropic client configuration."""

    type: Literal["anthropic"]
    model: str
    aws_access_key: str
    aws_secret_key: str
    region: str

    def create_client(self) -> cloai.LargeLanguageModel:
        """Create an Anthropic client instance."""
        return cloai.LargeLanguageModel(
            client=cloai.AnthropicBedrockLlm(
                model=self.model,  # type: ignore # don't want to need to keep literal union in sync with cloai
                aws_access_key=self.aws_access_key,
                aws_secret_key=self.aws_secret_key,
                region=self.region,
            )
        )


class OpenAIConfig(BaseModel):
    """OpenAI client configuration."""

    type: Literal["openai"]
    model: str
    api_key: str
    base_url: str | None = Field(default=None)

    def create_client(self) -> cloai.LargeLanguageModel:
        """Create an OpenAI client instance."""
        return cloai.LargeLanguageModel(
            client=cloai.OpenAiLlm(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
            )
        )


class AzureConfig(BaseModel):
    """Azure client configuration."""

    type: Literal["azure"]
    api_key: str
    endpoint: str
    deployment: str
    api_version: str

    def create_client(self) -> cloai.LargeLanguageModel:
        """Create an Azure client instance."""
        return cloai.LargeLanguageModel(
            client=cloai.AzureLlm(
                api_key=self.api_key,
                endpoint=self.endpoint,
                deployment=self.deployment,
                api_version=self.api_version,
            )
        )


class ClientConfig(BaseModel):
    """Configuration for all clients."""

    clients: dict[str, AnthropicConfig | OpenAIConfig | AzureConfig]

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
    config_json = environ.get("CONFIG_JSON")
    if config_json:
        client_config = ClientConfig.model_validate_json(config_json)
        return Config(clients=client_config.create_clients())

    # Fall back to config file
    config_path = pathlib.Path(environ.get("CONFIG_PATH", "config.json"))
    if not config_path.exists():
        raise FileNotFoundError(
            (
                f"Config file not found at {config_path} and CONFIG_JSON environment " 
                "variable not set."
            )
        )

    client_config = ClientConfig.model_validate_json(config_path.read_text("utf8"))
    return Config(clients=client_config.create_clients())
