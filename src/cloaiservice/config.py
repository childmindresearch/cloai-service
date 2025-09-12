"""App configuration."""

import json
import pathlib
from functools import lru_cache
from os import environ
from typing import Any, Literal

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
    model: cloai_bedrock.ANTHROPIC_BEDROCK_MODELS | str
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


def create_clients(
    config: dict[str, dict[str, Any]],
) -> dict[str, cloai.LargeLanguageModel]:
    """Creates the LLM clients.

    This function will run for all clients, even when one fails, so that
    all errors in the config can be returned to the user.

    Args:
        config: The JSON formatted configurations.

    Returns:
        A dictionary of LLM clients.

    Raises:
        500: For malformed configurations.
    """
    clients: dict[str, cloai.LargeLanguageModel] = {}
    errors = []

    type_constructors = {
        "azure": AzureConfig,
        "bedrock-anthropic": BedrockAnthropicConfig,
        "openai": OpenAIConfig,
    }

    for name, args in config["clients"].items():
        if "type" not in args:
            errors.append(f"Client {name} is missing a 'type' property.")
            continue

        if args["type"] not in type_constructors:
            errors.append(
                (
                    f"Unknown client type: {args['type']}. "
                    f"Valid types: {type_constructors.keys()}"
                )
            )
            continue

        try:
            clients[name] = type_constructors[args["type"]](**args).create_client()
        except pydantic.ValidationError as exc_info:
            # Report only the type and the location, as further contents may contain
            # secrets.
            validation_errors = [
                f"Error type: {error['type']}, Location: {error['loc']}"
                for error in exc_info.errors()
            ]
            errors.append(
                f"Error validating client {name}: " + " ".join(validation_errors)
            )

    if errors:
        raise fastapi.HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="; ".join(errors)
        )

    return clients


class Config(pydantic.BaseModel):
    """Service configuration."""

    clients: dict[str, cloai.LargeLanguageModel]


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

    clients = create_clients(json.loads(config_json))
    return Config(clients=clients)
