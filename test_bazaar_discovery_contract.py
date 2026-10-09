"""Discovery contract for POST /evaluate/fast.

These tests read the extension the process publishes and call the route
handlers directly. They do not send a payment and do not start the hosted service.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest
from fastapi import HTTPException

_db = tempfile.NamedTemporaryFile(prefix="dcl-bazaar-", suffix=".db", delete=False)
_db.close()
os.environ["DCL_DB_PATH"] = _db.name

import bazaar_server as bazaar  # noqa: E402


FAST = "POST /evaluate/fast"
DIGEST = "ab" * 32
MIXED_DIGEST = "aB" * 32
OLD_ROUTES = (
    (bazaar.evaluate_strict, "strict", "POST /evaluate/strict"),
    (bazaar.evaluate_jailbreak, "jailbreak", "POST /evaluate/jailbreak"),
    (bazaar.evaluate_safety, "safety", "POST /evaluate/safety"),
    (bazaar.evaluate_quality, "quality", "POST /evaluate/quality"),
)


def _extension(route_key: str) -> dict:
    return bazaar.routes[route_key].extensions["bazaar"]


def _body_schema(route_key: str) -> dict:
    return _extension(route_key)["schema"]["properties"]["input"]["properties"]["body"]


def _output_example_schema(route_key: str) -> dict:
    return _extension(route_key)["schema"]["properties"]["output"]["properties"]["example"]


def _fast_request(**overrides) -> bazaar.EvaluateRequest:
    payload = {
        "response": "plain text",
        "agent_id": "agent-1",
        "task_type": "http_side_effect",
        "request_digest": MIXED_DIGEST,
    }
    payload.update(overrides)
    return bazaar.EvaluateRequest(**payload)


def _assert_fast_rejected(req: bazaar.EvaluateRequest) -> HTTPException:
    before = len(bazaar._chain)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(bazaar.evaluate_fast(req))
    assert caught.value.status_code == 400
    assert len(bazaar._chain) == before
    return caught.value


def test_fast_discovery_publishes_task_type_and_request_digest():
    info = _extension(FAST)["info"]
    body = info["input"]["body"]
    example = info["output"]["example"]
    schema = _body_schema(FAST)
    output_schema = _output_example_schema(FAST)

    assert info["input"]["method"] == "POST"
    assert info["input"]["bodyType"] == "json"
    assert body["task_type"] == "http_side_effect"
    assert body["request_digest"] == DIGEST
    assert "response" in body
    assert example["request_digest"] == DIGEST
    assert example["verdict"] == "COMMIT"
    assert isinstance(example["reason"], str)

    assert schema["properties"]["task_type"]["const"] == "http_side_effect"
    assert schema["properties"]["request_digest"]["pattern"] == bazaar._REQUEST_DIGEST_PATTERN.pattern
    assert schema["properties"]["request_digest"]["pattern"] == "^[0-9a-fA-F]{64}$"
    assert set(schema["required"]) >= {"response", "task_type", "request_digest"}
    assert "request_digest" in output_schema["properties"]
    assert output_schema["properties"]["request_digest"]["pattern"] == bazaar._REQUEST_DIGEST_PATTERN.pattern
    assert "request_digest" in output_schema["required"]
    assert output_schema["properties"]["verdict"]["type"] == "string"


def test_fast_payment_terms_stay_one_cent_on_the_configured_network():
    accept = bazaar.routes[FAST].accepts
    assert accept.price == "$0.01"
    assert accept.scheme == "exact"
    assert accept.network == bazaar.X402_NETWORK
    assert accept.pay_to == bazaar.X402_WALLET


def test_other_post_routes_keep_their_previous_input_example():
    for _handler, _tier, route_key in OLD_ROUTES:
        body = _extension(route_key)["info"]["input"]["body"]
        assert body == {"response": "example agent output", "agent_id": "agent-123"}
        schema = _body_schema(route_key)
        assert "task_type" not in schema["properties"]
        assert "request_digest" not in schema["properties"]
        assert "request_digest" not in schema.get("required", [])
        output_schema = _output_example_schema(route_key)
        assert "request_digest" not in output_schema.get("required", [])
        assert "request_digest" not in _extension(route_key)["info"]["output"]["example"]


def test_get_probe_does_not_publish_the_post_body():
    info = _extension("GET /evaluate/fast")["info"]
    body = info["input"].get("body")
    assert body in (None, {})
    assert "request_digest" not in info["output"]["example"]


def test_valid_fast_request_returns_the_original_digest():
    before = len(bazaar._chain)
    response = asyncio.run(bazaar.evaluate_fast(_fast_request()))
    assert response.verdict == "COMMIT"
    assert response.request_digest == MIXED_DIGEST
    assert response.request_digest != MIXED_DIGEST.lower()
    entry = bazaar._chain.get_by_tx(response.tx_hash)
    assert entry["task_type"] == "http_side_effect"
    assert len(bazaar._chain) == before + 1


def test_missing_request_digest_is_rejected_before_the_chain():
    error = _assert_fast_rejected(_fast_request(request_digest=None))
    assert error.detail == "request_digest must be 64 hex characters"


def test_invalid_request_digest_is_rejected_before_the_chain():
    invalid = (
        "",
        "abc",
        "ab" * 31,
        "ab" * 32 + "c",
        "g" * 64,
        " " + DIGEST,
        DIGEST + " ",
        " " * 64,
    )
    for digest in invalid:
        error = _assert_fast_rejected(_fast_request(request_digest=digest))
        assert error.detail == "request_digest must be 64 hex characters"


def test_missing_task_type_is_rejected_before_the_chain():
    error = _assert_fast_rejected(
        bazaar.EvaluateRequest(
            response="plain text",
            agent_id="agent-1",
            request_digest=DIGEST,
        )
    )
    assert error.detail == "task_type must be http_side_effect"
    error = _assert_fast_rejected(_fast_request(task_type=None))
    assert error.detail == "task_type must be http_side_effect"


def test_other_task_type_is_rejected_before_the_chain():
    for task_type in ("fast", "unknown", "HTTP_SIDE_EFFECT", "http_side_effect_extra"):
        error = _assert_fast_rejected(_fast_request(task_type=task_type))
        assert error.detail == "task_type must be http_side_effect"


def test_empty_or_padded_task_type_is_rejected_before_the_chain():
    for task_type in ("", " ", " http_side_effect", "http_side_effect ", "\thttp_side_effect", "http_side_effect\n"):
        error = _assert_fast_rejected(_fast_request(task_type=task_type))
        assert error.detail == "task_type must be http_side_effect"


def test_old_routes_keep_the_previous_contract():
    for handler, tier, _route_key in OLD_ROUTES:
        before = len(bazaar._chain)
        response = asyncio.run(
            handler(bazaar.EvaluateRequest(response="plain text", agent_id="agent-1"))
        )
        assert response.request_digest is None
        entry = bazaar._chain.get_by_tx(response.tx_hash)
        assert entry["task_type"] == tier
        assert len(bazaar._chain) == before + 1

        echoed = asyncio.run(
            handler(
                bazaar.EvaluateRequest(
                    response="plain text",
                    task_type="http_side_effect",
                    request_digest=MIXED_DIGEST,
                )
            )
        )
        assert echoed.request_digest == MIXED_DIGEST
        stored = bazaar._chain.get_by_tx(echoed.tx_hash)
        assert stored["task_type"] == "http_side_effect"

    unknown = asyncio.run(
        bazaar.evaluate_strict(
            bazaar.EvaluateRequest(response="plain text", task_type="unknown")
        )
    )
    assert bazaar._chain.get_by_tx(unknown.tx_hash)["task_type"] == "strict"

    before = len(bazaar._chain)
    with pytest.raises(HTTPException) as bad_digest:
        asyncio.run(
            bazaar.evaluate_strict(
                bazaar.EvaluateRequest(response="plain text", request_digest="abc")
            )
        )
    assert bad_digest.value.status_code == 400
    with pytest.raises(HTTPException) as padded:
        asyncio.run(
            bazaar.evaluate_strict(
                bazaar.EvaluateRequest(response="plain text", task_type="  http_side_effect")
            )
        )
    assert padded.value.status_code == 400
    assert len(bazaar._chain) == before


def test_discovery_example_matches_server_validation():
    body = _extension(FAST)["info"]["input"]["body"]
    output = _extension(FAST)["info"]["output"]["example"]
    assert bazaar._REQUEST_DIGEST_PATTERN.fullmatch(body["request_digest"])
    assert output["request_digest"] == body["request_digest"]
    assert body["task_type"] == "http_side_effect"

    before = len(bazaar._chain)
    response = asyncio.run(bazaar.evaluate_fast(bazaar.EvaluateRequest(**body)))
    assert response.request_digest == body["request_digest"]
    assert response.request_digest == output["request_digest"]
    assert response.verdict in {"COMMIT", "NO_COMMIT"}
    assert isinstance(response.reason, str)
    assert len(bazaar._chain) == before + 1

    rejected = _assert_fast_rejected(
        bazaar.EvaluateRequest(response=body["response"], agent_id=body["agent_id"])
    )
    assert rejected.status_code == 400
