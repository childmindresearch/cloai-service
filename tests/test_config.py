"""Unit tests for the config module."""

import functools
import os
import pathlib
from typing import Any, Callable

import fastapi
import pytest
from fastapi import status

from cloaiservice import config


@pytest.fixture(autouse=True, scope="function")
def reset_cache() -> None:
    """Resets the get_config cache on every test."""
    config.get_config.cache_clear()


@pytest.fixture
def config_json() -> str:
    """A JSON configuration used for the tests."""
    return """
    {
        "clients": {
            "test-model": { 
                "type": "bedrock-anthropic", 
                "model": "anthropic.claude-3-5-sonnet-20241022-v2:0", 
                "aws_access_key": "01234567890123456789", 
                "aws_secret_key": "0123456789012345678901234567890123456789", 
                "region": "us-west-2"
            }
        }
    }
    """


def reset_env_variables(*env_vars: str) -> Callable:
    """Saves environment variables and resets them after the test."""

    def decorator(function: Callable) -> Callable:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            cache = {var: os.environ.get(var, "") for var in env_vars}
            try:
                function(*args, **kwargs)
            finally:
                for key, value in cache.items():
                    os.environ[key] = value

        return wrapper

    return decorator


@reset_env_variables("CONFIG_PATH", "CONFIG_JSON")
def test_get_config_environment(config_json: str) -> None:
    """Get CONFIG_JSON from environment."""
    os.environ["CONFIG_PATH"] = ""
    os.environ["CONFIG_JSON"] = config_json

    result = config.get_config()

    assert len(result.clients) == 1
    assert (
        result.clients["test-model"].client.model
        == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    )


@reset_env_variables("CONFIG_PATH", "CONFIG_JSON")
def test_get_config_file(tmp_path: pathlib.Path, config_json: str) -> None:
    """Get CONFIG_JSON from file."""
    config_file = tmp_path / "config.json"
    config_file.write_text(config_json)
    os.environ["CONFIG_PATH"] = str(config_file)
    os.environ["CONFIG_JSON"] = ""

    result = config.get_config()

    assert len(result.clients) == 1
    assert (
        result.clients["test-model"].client.model
        == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    )


@reset_env_variables("CONFIG_PATH", "CONFIG_JSON")
def test_get_config_not_specified() -> None:
    """Test that an error is raised for no config specified."""
    os.environ["CONFIG_PATH"] = ""
    os.environ["CONFIG_JSON"] = ""

    with pytest.raises(fastapi.HTTPException) as exc_info:
        config.get_config()

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
