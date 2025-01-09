"""Health check."""

from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}
