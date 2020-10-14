import json
import kubernetes_asyncio as kubernetes
import uuid

from http import HTTPStatus
from unittest.mock import AsyncMock

from adaptdl_sched.validator import Validator


def _assert_response(response_json, request_json):
    assert response_json["apiVersion"] == "admission.k8s.io/v1"
    assert response_json["kind"] == "AdmissionReview"
    assert response_json["response"]["uid"] == request_json["request"]["uid"]


async def test_healthz(aiohttp_client, loop):
    app = Validator().get_app()
    client = await aiohttp_client(app)
    response = await client.get("/healthz")
    assert response.status == HTTPStatus.OK


async def test_create_invalid_template(aiohttp_client, loop):
    validator = Validator()
    # Set up mocks.
    exc = kubernetes.client.rest.ApiException(
        status=HTTPStatus.UNPROCESSABLE_ENTITY, reason="reason")
    exc.body = json.dumps({"message": str(uuid.uuid4())})
    mock = AsyncMock(side_effect=exc)
    validator._core_api.create_namespaced_pod_template = mock
    # Send request.
    app = validator.get_app()
    client = await aiohttp_client(app)
    template = {"key": str(uuid.uuid4())}
    request_json = {
        "request": {
            "uid": str(uuid.uuid4()),
            "operation": "CREATE",
            "namespace": str(uuid.uuid4()),
            "object": {"spec": {"template": template}},
        }
    }
    response = await client.post("/validate", json=request_json)
    # Check template dry run.
    assert mock.call_args.args[0] == request_json["request"]["namespace"]
    assert mock.call_args.args[1]["template"] == template
    assert mock.call_args.kwargs["dry_run"] == "All"
    # Check HTTP response.
    assert response.status == HTTPStatus.OK
    response_json = await response.json()
    _assert_response(response_json, request_json)
    # Check operation was disallowed.
    assert not response_json["response"]["allowed"]
    status = response_json["response"]["status"]
    assert status["code"] == HTTPStatus.UNPROCESSABLE_ENTITY
    assert status["reason"] == "Invalid"
    assert status["message"] == json.loads(exc.body)["message"]


async def test_create_invalid_replicas(aiohttp_client, loop):
    validator = Validator()
    validator._core_api.create_namespaced_pod_template = AsyncMock()
    # Send request.
    app = validator.get_app()
    client = await aiohttp_client(app)
    request_json = {
        "request": {
            "uid": str(uuid.uuid4()),
            "operation": "CREATE",
            "namespace": str(uuid.uuid4()),
            "object": {"spec": {"minReplicas": 4, "maxReplicas": 2,
                                "template": {}}},
        }
    }
    response = await client.post("/validate", json=request_json)
    # Check HTTP response.
    assert response.status == HTTPStatus.OK
    response_json = await response.json()
    _assert_response(response_json, request_json)
    # Check operation was disallowed.
    assert not response_json["response"]["allowed"]
    status = response_json["response"]["status"]
    assert status["code"] == HTTPStatus.UNPROCESSABLE_ENTITY
    assert status["reason"] == "Invalid"


async def test_update_spec(aiohttp_client, loop):
    validator = Validator()
    # Send request.
    app = validator.get_app()
    client = await aiohttp_client(app)
    request_json = {
        "request": {
            "uid": str(uuid.uuid4()),
            "operation": "UPDATE",
            "namespace": str(uuid.uuid4()),
            "object": {"spec": {"key": "value"}},
            "oldObject": {"spec": {"key": "oldValue"}},
        }
    }
    response = await client.post("/validate", json=request_json)
    # Check HTTP response.
    assert response.status == HTTPStatus.OK
    response_json = await response.json()
    _assert_response(response_json, request_json)
    # Check operation was disallowed.
    assert not response_json["response"]["allowed"]
    status = response_json["response"]["status"]
    assert status["code"] == HTTPStatus.UNPROCESSABLE_ENTITY
    assert status["reason"] == "Forbidden"
