"""Endpoints talking to LLMs."""

from typing import Annotated

import cloai
import fastapi
from fastapi import APIRouter, Body, Depends, HTTPException, status

from cloaiservice.config import get_config
from cloaiservice.models.llm import (
    ChainOfVerificationRequest,
    InstructorRequest,
    LLMResponse,
    PromptRequest,
)
from cloaiservice.services import schemaconverter

router = APIRouter()


def get_llm_client(id: str) -> cloai.LargeLanguageModel:
    """Get an LLM client by its ID."""
    client = get_config().clients.get(id)
    if client is None:
        raise fastapi.HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Client not found"
        )
    return client


@router.post("/run", response_model=LLMResponse)
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
    except Exception as exc_info:
        raise HTTPException(status_code=500, detail=str(exc_info)) from exc_info


@router.post("/instructor", response_model=LLMResponse)
async def run_instructor(
    request: Annotated[InstructorRequest, Body(...)],
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
    except Exception as exc_info:
        raise HTTPException(status_code=500, detail=str(exc_info)) from exc_info


@router.post("/cov", response_model=LLMResponse)
async def chain_of_verification(
    request: Annotated[ChainOfVerificationRequest, Body(...)],
    llm: Annotated[cloai.LargeLanguageModel, Depends(get_llm_client)],
) -> LLMResponse:
    """Run chain of verification on a prompt."""
    try:
        result = await llm.chain_of_verification(  # type: ignore[call-arg] # pycharm is confused
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            response_model=str,
            statements=request.statements,
            max_verifications=request.max_verifications,
            create_new_statements=request.create_new_statements,
            error_on_iteration_limit=request.error_on_iteration_limit,
        )
        return LLMResponse(result=result)
    except Exception as exc_info:
        raise HTTPException(status_code=500, detail=str(exc_info)) from exc_info
