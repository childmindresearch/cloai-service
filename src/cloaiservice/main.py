"""App entrypoint."""

from fastapi import FastAPI

from cloaiservice.routes import clients, health, llm

app = FastAPI(
    title="cloai API Service",
    description="API service for interacting with various LLM providers",
    version="0.1.0",
)

app.include_router(clients.router, prefix="/clients", tags=["clients"])
app.include_router(llm.router, prefix="/llm", tags=["llm"])
app.include_router(health.router, prefix="/health", tags=["health"])
