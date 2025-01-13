"""LLM endpoint data models."""

from typing import Any, Self

import fastapi
import pydantic
from fastapi import status
from pydantic import BaseModel, Field


class PromptRequest(BaseModel):
    """Prompt request."""

    system_prompt: str = Field(..., description="The system prompt to send to the LLM")
    user_prompt: str = Field(..., description="The user prompt to send to the LLM")


class InstructorRequest(PromptRequest):
    """Instructor request."""

    response_model: Any = Field(..., description="JSON schema response model")
    max_tokens: int = Field(4096, description="Maximum tokens to generate")


class ChainOfVerificationRequest(PromptRequest):
    """Chain of verification request."""

    max_verifications: int = Field(3, description="Maximum verification iterations")
    statements: list[str] | None = Field(
        None,
        description=(
            "Verification statements. If None are provided, "
            "create_new_statements must be set to True."
        ),
    )
    create_new_statements: bool = Field(
        False,
        description="Whether to generate new verification statements. If no statements"
        "are provided, then this must be set to True.",
    )
    error_on_iteration_limit: bool = Field(
        False, description="Whether to raise an error on hitting iteration limit"
    )

    @pydantic.model_validator(mode="after")
    def validate_create_statements_if_none_provided(self) -> Self:
        """Raises 400 if no statements are provided or generated."""
        if not self.statements and not self.create_new_statements:
            raise fastapi.HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "If no statements are provided then create_new_statements "
                    "must be set to True."
                ),
            )
        return self


class LLMResponse(BaseModel):
    """LLM Response base class."""

    result: Any = Field(..., description="The LLM response")
