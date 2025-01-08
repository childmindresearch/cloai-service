""".. include:: ../../README.md"""  # noqa: D415

import dataclasses
import json
from functools import lru_cache
from typing import Annotated, Any, Generic, TypeVar

import cloai
import cloai.llm.utils
from fastapi import Body, Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.config import environ

from cloaiservice import schemaconverter

T = TypeVar("T")


@dataclasses.dataclass
class Config:
    """Service configuration."""
    clients: dict[str, cloai.LargeLanguageModel]


# Request/Response Models
class PromptRequest(BaseModel):
    """Prompt request."""
    system_prompt: str = Field(..., description="The system prompt to send to the LLM")
    user_prompt: str = Field(..., description="The user prompt to send to the LLM")


class InstructorRequest(PromptRequest, Generic[T]):
    """Instructor request."""
    response_model: Any = Field(..., description="JSON schema response model")
    max_tokens: int = Field(4096, description="Maximum tokens to generate")


class ChainOfVerificationRequest(InstructorRequest[T]):
    """Chain of verification request."""
    statements: list[str] | None = Field(None, description="Verification statements")
    max_verifications: int = Field(3, description="Maximum verification iterations")
    create_new_statements: bool = Field(
        False, description="Whether to generate new verification statements"
    )
    error_on_iteration_limit: bool = Field(
        False, description="Whether to raise an error on hitting iteration limit"
    )


class LLMResponse(BaseModel):
    """LLM Response base class."""
    result: Any = Field(..., description="The LLM response")


@lru_cache()
def get_config() -> Config:
    """Load config from environment or config file."""
    # TODO/TBD: load this from environment variables or a config file
    return Config(
        clients={
            "sonnet": cloai.LargeLanguageModel(
                client=cloai.AnthropicBedrockLlm(
                    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                    aws_access_key="hello",
                    aws_secret_key="hello",
                    region="us-east-1",
                )
            ),
            "gpt": cloai.LargeLanguageModel(
                client=cloai.OpenAiLlm(
                    model="gpt-4o",
                    api_key=environ.get("OAI_API_KEY"),
                )
            ),
            "azure": cloai.LargeLanguageModel(
                client=cloai.AzureLlm(
                    api_key="hello",
                    endpoint="hello",
                    deployment="hello",
                    api_version="hello",
                )
            ),
        }
    )


async def get_llm_client(id: str) -> cloai.LargeLanguageModel | None:
    """Get an LLM client by its ID."""
    return get_config().clients.get(id)


app = FastAPI(
    title="cloai API Service",
    description="API service for interacting with various LLM providers",
    version="0.1.0",
)


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


@app.get("/clients", response_model=AvailableClientsResponse)
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


@app.post("/run", response_model=LLMResponse)
async def run_prompt(
    request: Annotated[PromptRequest, Body(...)],
    llm: Annotated[cloai.LargeLanguageModel, Depends(get_llm_client)],
) -> LLMResponse:
    """Run a basic prompt against the LLM."""
    try:
        result = await llm.run(
            system_prompt=request.system_prompt, user_prompt=request.user_prompt
        )
        return LLMResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/instructor", response_model=LLMResponse)
async def run_instructor(
    request: Annotated[InstructorRequest[T], Body(...)],
    llm: Annotated[cloai.LargeLanguageModel, Depends(get_llm_client)],
) -> LLMResponse:
    """Run a structured query using instructor."""
    try:
        model = schemaconverter.create_model_from_schema(request.response_model)

        result = await llm.call_instructor(
            response_model=model,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            max_tokens=request.max_tokens,
        )
        return LLMResponse(result=result)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON schema: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cov", response_model=LLMResponse)
async def chain_of_verification(
    request: Annotated[ChainOfVerificationRequest[T], Body(...)],
    llm: Annotated[cloai.LargeLanguageModel, Depends(get_llm_client)],
) -> LLMResponse:
    """Run chain of verification on a prompt."""
    try:
        model = schemaconverter.create_model_from_schema(request.response_model)

        result = await llm.chain_of_verification(  # type: ignore # pycharm is confused
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            response_model=model,
            statements=request.statements,
            max_verifications=request.max_verifications,
            create_new_statements=request.create_new_statements,
            error_on_iteration_limit=request.error_on_iteration_limit,
        )
        return LLMResponse(result=result)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON schema: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}
