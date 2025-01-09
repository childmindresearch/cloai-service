"""LLM endpoint data models."""

from typing import Any

from pydantic import BaseModel, Field


class PromptRequest(BaseModel):
    """Prompt request."""

    system_prompt: str = Field(..., description="The system prompt to send to the LLM")
    user_prompt: str = Field(..., description="The user prompt to send to the LLM")


class InstructorRequest(PromptRequest):
    """Instructor request."""

    response_model: Any = Field(..., description="JSON schema response model")
    max_tokens: int = Field(4096, description="Maximum tokens to generate")


class ChainOfVerificationRequest(BaseModel):
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
