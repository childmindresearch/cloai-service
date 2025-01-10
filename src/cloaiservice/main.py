"""App entrypoint."""

from fastapi import APIRouter, FastAPI

from cloaiservice.routes import clients, health, llm

app = FastAPI(
    title="cloai API Service",
    description="API service for interacting with various LLM providers",
    version="0.1.0",
)

version_router = APIRouter(prefix="/v1")
version_router.include_router(clients.router, prefix="/clients", tags=["clients"])
version_router.include_router(llm.router, prefix="/llm", tags=["llm"])
version_router.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(version_router)
