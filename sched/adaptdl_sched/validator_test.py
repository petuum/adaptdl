import pytest

from aiohttp.test_utils import TestClient, TestServer, loop_context
from unittest.mock import AsyncMock

from adaptdl_sched.validator import Validator


async def test_healthz(aiohttp_client, loop):
    app = Validator().get_app()
    client = await aiohttp_client(app)
    response = await client.get("/healthz")
    assert response.status == 200


async def test_mutate(aiohttp_client, loop):
    validator = Validator()
    validator._core_api = AsyncMock()
    app = validator.get_app()
    client = await aiohttp_client(app)
    response = await client.post("/mutate", json={
        "request": {
            "uid": "uid",
            "namespace": "default",
            "object": {
                "spec": {
                    "template": {
                        "key": "val"
                    }
                }
            }
        }
    })
    assert response.status == 200
