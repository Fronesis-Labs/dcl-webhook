"""Discovery contract for POST /evaluate/fast.

These tests read the extension the process publishes and call the evaluation
function directly. They do not send a payment and do not start the hosted service.
"""

from __future__ import annotations

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


def _extension(route_key: str) -> dict:
    return bazaar.routes[route_key].extensions["bazaar"]


def _body_schema(route_key: str) -> dict:
    return _extension(route_key)["schema"]["properties"]["input"]["properties"]["body"]


def _output_example_schema(route_key: str) -> dict:
    return _extension(route_key)["schema"]["properties"]["output"]["properties"]["example"]


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
    assert schema["properties"]["request_digest"]["pattern"] == "^[0-9a-fA-F]{64}$"
    assert set(schema["required"]) >= {"response", "task_type", "request_digest"}
    assert "request_digest" in output_schema["properties"]
    assert "request_digest" in output_schema["required"]
    assert output_schema["properties"]["verdict"]["type"] == "string"


def test_fast_payment_terms_stay_one_cent_on_the_configured_network():
    accept = bazaar.routes[FAST].accepts
    assert accept.price == "$0.01"
    assert accept.scheme == "exact"
    assert accept.network == bazaar.X402_NETWORK
    assert accept.pay_to == bazaar.X402_WALLET


def test_other_post_routes_keep_their_previous_input_example():
    body = _extension("POST /evaluate/strict")["info"]["input"]["body"]
    assert body == {"response": "example agent output", "agent_id": "agent-123"}
    assert "task_type" not in _body_schema("POST /evaluate/strict")["properties"]


def test_get_probe_does_not_publish_the_post_body():
    body = _extension("GET /evaluate/fast")["info"]["input"].get("body")
    assert body in (None, {})


def test_client_task_type_is_stored_and_digest_is_echoed_unchanged():
    before = len(bazaar._chain)
    response = bazaar._process_evaluation(
        bazaar.EvaluateRequest(
            response="plain text",
            agent_id="agent-1",
            task_type="http_side_effect",
            request_digest=MIXED_DIGEST,
        ),
        "default",
        "fast",
    )
    assert response.request_digest == MIXED_DIGEST
    entry = bazaar._chain.get_by_tx(response.tx_hash)
    assert entry["task_type"] == "http_side_effect"
    assert len(bazaar._chain) == before + 1


def test_omitted_task_type_keeps_the_tier_and_omitted_digest_is_not_replaced():
    response = bazaar._process_evaluation(
        bazaar.EvaluateRequest(response="plain text", agent_id="agent-1"),
        "default",
        "fast",
    )
    assert response.request_digest is None
    entry = bazaar._chain.get_by_tx(response.tx_hash)
    assert entry["task_type"] == "fast"


def test_explicit_unknown_task_type_keeps_the_tier():
    response = bazaar._process_evaluation(
        bazaar.EvaluateRequest(response="plain text", task_type="unknown"),
        "default",
        "strict",
    )
    entry = bazaar._chain.get_by_tx(response.tx_hash)
    assert entry["task_type"] == "strict"


def test_invalid_digest_and_task_type_are_rejected_before_the_chain():
    before = len(bazaar._chain)
    with pytest.raises(HTTPException) as digest_error:
        bazaar._process_evaluation(
            bazaar.EvaluateRequest(response="plain text", request_digest="abc"),
            "default",
            "fast",
        )
    assert digest_error.value.status_code == 400
    with pytest.raises(HTTPException) as blank_task:
        bazaar._process_evaluation(
            bazaar.EvaluateRequest(response="plain text", task_type="  http_side_effect"),
            "default",
            "fast",
        )
    assert blank_task.value.status_code == 400
    with pytest.raises(HTTPException) as empty_response:
        bazaar._process_evaluation(
            bazaar.EvaluateRequest(response="  "),
            "default",
            "fast",
        )
    assert empty_response.value.status_code == 400
    assert len(bazaar._chain) == before
