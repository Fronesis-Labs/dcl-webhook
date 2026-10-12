"""Discovery and payment-route coverage for the 10 Bazaar resources.

Does not call the facilitator and does not write the default chain file.
`bazaar_server` probes the facilitator at import, so that probe is stubbed
before the import. The route table and handlers are unchanged.
"""
import asyncio
import inspect
import threading

import pytest
from fastapi import HTTPException
from dcl_audit_store import ensure_schema, find_by_tx_hash
from dcl_core import ChainState
from x402.http.facilitator_client import HTTPFacilitatorClient
from x402.http.types import HTTPRequestContext
from x402.http.x402_http_server import x402HTTPResourceServer

HTTPFacilitatorClient.get_supported = lambda self: None  # noqa: E731

import bazaar_server as bz


EXISTING = {
    "POST /evaluate/fast": "$0.01",
    "POST /evaluate/strict": "$0.05",
    "POST /evaluate/jailbreak": "$0.02",
    "POST /evaluate/safety": "$0.01",
    "POST /evaluate/quality": "$0.03",
}
ADDED = {
    "POST /evaluate/secrets": "$0.02",
    "POST /evaluate/pii": "$0.02",
    "POST /evaluate/batch": "$0.10",
    "GET /audit/:tx_hash": "$0.10",
    "GET /audit/:tx_hash/deep": "$0.50",
}
ALL = {**EXISTING, **ADDED}

USDC = {
    "$0.01": "10000",
    "$0.02": "20000",
    "$0.03": "30000",
    "$0.05": "50000",
    "$0.10": "100000",
    "$0.50": "500000",
}


def _extension(route_key: str) -> dict:
    return bz.routes[route_key].extensions["bazaar"]


def _body_schema(route_key: str) -> dict:
    return _extension(route_key)["schema"]["properties"]["input"]["properties"]["body"]


def _output_schema(route_key: str) -> dict:
    return _extension(route_key)["schema"]["properties"]["output"]["properties"]["example"]


def _path_schema(route_key: str) -> dict:
    return _extension(route_key)["schema"]["properties"]["input"]["properties"]["pathParams"]


def _manifest_by_url() -> dict:
    manifest = bz.x402_manifest()
    return {item["resource"]["url"]: item for item in manifest["resources"]}


def _payment_server() -> x402HTTPResourceServer:
    return x402HTTPResourceServer(bz.server, bz.routes)


def _requires(server: x402HTTPResourceServer, path: str, method: str) -> bool:
    return server.requires_payment(HTTPRequestContext(adapter=None, path=path, method=method))


def test_routes_dict_has_ten_paid_resources():
    assert list(bz.routes) == list(ALL)
    for key, price in ALL.items():
        cfg = bz.routes[key]
        assert cfg.accepts.price == price
        assert cfg.accepts.scheme == "exact"
        assert cfg.accepts.pay_to == bz.X402_WALLET
        assert cfg.accepts.network == bz.X402_NETWORK
        method, path = key.split(" ", 1)
        assert cfg.resource == f"{bz.PUBLIC_BASE_URL}{path}"
        assert _extension(key)["info"]["input"]["method"] == method


def test_payment_middleware_receives_the_same_routes():
    middleware = next(
        item for item in bz.app.user_middleware
        if item.cls.__name__ == "PaymentMiddlewareASGI"
    )
    assert middleware.kwargs["routes"] is bz.routes
    assert middleware.kwargs["server"] is bz.server


def test_payment_matcher_covers_concrete_paths_and_not_the_other_method():
    server = _payment_server()
    assert _requires(server, "/evaluate/fast", "POST")
    assert _requires(server, "/evaluate/secrets", "POST")
    assert _requires(server, "/evaluate/pii", "POST")
    assert _requires(server, "/evaluate/batch", "POST")
    assert not _requires(server, "/evaluate/secrets", "GET")
    assert not _requires(server, "/evaluate/fast", "GET")
    assert _requires(server, "/audit/0xabc", "GET")
    assert _requires(server, "/audit/0xabc/deep", "GET")
    assert not _requires(server, "/audit/0xabc", "POST")
    assert not _requires(server, "/audit/0xabc/deep", "POST")
    assert not _requires(server, "/health", "GET")


def test_well_known_lists_all_ten_with_method_path_and_price():
    manifest = bz.x402_manifest()
    assert manifest["x402Version"] == 2
    resources = manifest["resources"]
    assert len(resources) == 10
    by_url = {item["resource"]["url"]: item for item in resources}
    for key, price in ALL.items():
        method, path = key.split(" ", 1)
        item = by_url[f"{bz.PUBLIC_BASE_URL}{path}"]
        assert item["resource"]["method"] == method
        assert item["resource"]["mimeType"] == "application/json"
        accept = item["accepts"][0]
        assert accept["scheme"] == "exact"
        assert accept["network"] == bz.X402_NETWORK
        assert accept["asset"] == bz.USDC_BASE
        assert accept["payTo"] == bz.X402_WALLET
        assert accept["amount"] == USDC[price]
        assert accept["maxTimeoutSeconds"] == 300


def test_existing_five_keep_evaluate_schemas_and_strict_policy():
    for path, _price, description, _tags in bz._EVALUATE_PATHS:
        key = f"POST {path}"
        assert key in EXISTING
        assert bz.routes[key].description == description
        assert bz.routes[key].accepts.price == EXISTING[key]
        assert _body_schema(key) == bz._EVALUATE_INPUT_SCHEMA
        assert _output_schema(key)["properties"]["verdict"]["enum"] == ["COMMIT", "NO_COMMIT"]
        info = _extension(key)["info"]["input"]
        assert info["method"] == "POST"
        assert info["bodyType"] == "json"
    source = inspect.getsource(bz.evaluate_strict)
    assert '_process_evaluation(req, "default", "strict")' in source
    assert [path for path, *_rest in bz._EVALUATE_PATHS] == [
        "/evaluate/fast",
        "/evaluate/strict",
        "/evaluate/jailbreak",
        "/evaluate/safety",
        "/evaluate/quality",
    ]


def test_scan_resources_use_scan_schema_not_evaluate_schema():
    for key, categories in (
        ("POST /evaluate/secrets", ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]),
        ("POST /evaluate/pii", ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]),
    ):
        body = _body_schema(key)
        assert set(body["properties"]) == {"response", "agent_id", "task_type"}
        assert body["required"] == ["response"]
        assert body != bz._EVALUATE_INPUT_SCHEMA
        output = _output_schema(key)
        assert output != bz._EVALUATE_OUTPUT_SCHEMA
        assert "risk_score" in output["properties"]
        assert "findings" in output["properties"]
        assert "seal_text" in output["properties"]
        assert "risk_score" in output["required"]
        assert "seal_text" in output["required"]
        example = _extension(key)["info"]["output"]["example"]
        assert example["categories_checked"] == categories
        assert _extension(key)["info"]["input"]["bodyType"] == "json"


def test_batch_resource_describes_items_and_max_items():
    body = _body_schema("POST /evaluate/batch")
    assert body["required"] == ["items", "agent_id"]
    assert body["properties"]["max_items"]["default"] == 20
    item = body["properties"]["items"]["items"]
    assert item["required"] == ["response"]
    assert set(item["properties"]) == {"response", "policy", "task_type"}
    output = _output_schema("POST /evaluate/batch")
    assert output != bz._EVALUATE_OUTPUT_SCHEMA
    assert output["required"] == ["batch_id", "agent_id", "count", "results"]
    result_fields = output["properties"]["results"]["items"]["properties"]
    assert "pipeline_id" in result_fields
    assert "seal_text" in result_fields
    assert "confidence" in result_fields


def test_audit_resources_are_get_with_tx_hash_path_param():
    for key in ("GET /audit/:tx_hash", "GET /audit/:tx_hash/deep"):
        info = _extension(key)["info"]["input"]
        assert info["method"] == "GET"
        assert info["type"] == "http"
        assert "bodyType" not in info
        assert "body" not in info
        assert "queryParams" not in info
        input_props = _extension(key)["schema"]["properties"]["input"]["properties"]
        assert "queryParams" not in input_props
        path_schema = _path_schema(key)
        assert path_schema["required"] == ["tx_hash"]
        assert path_schema["properties"]["tx_hash"]["type"] == "string"
    basic = _output_schema("GET /audit/:tx_hash")
    assert "chain_integrity" in basic["properties"]
    assert "tampered_at_index" not in basic["properties"]
    deep = _output_schema("GET /audit/:tx_hash/deep")
    assert "tampered_at_index" in deep["properties"]
    assert "drift_context" in deep["properties"]
    urls = set(_manifest_by_url())
    assert f"{bz.PUBLIC_BASE_URL}/audit/:tx_hash" in urls
    assert f"{bz.PUBLIC_BASE_URL}/audit/:tx_hash/deep" in urls


def test_fastapi_paths_match_the_paid_resources():
    routes = {
        (next(iter(route.methods)), route.path)
        for route in bz.app.routes
        if getattr(route, "methods", None)
    }
    assert ("POST", "/evaluate/secrets") in routes
    assert ("POST", "/evaluate/pii") in routes
    assert ("POST", "/evaluate/batch") in routes
    assert ("GET", "/audit/{tx_hash}") in routes
    assert ("GET", "/audit/{tx_hash}/deep") in routes
    assert ("POST", "/evaluate/fast") in routes
    assert ("GET", "/.well-known/x402") in routes
    assert ("GET", "/.well-known/x402.json") in routes


def test_fast_keeps_production_audit_event_and_payment_log(tmp_path, monkeypatch):
    chain = ChainState(str(tmp_path / "chain.db"))
    ensure_schema(chain._conn, threading.RLock())
    monkeypatch.setattr(bz, "_chain", chain)
    monkeypatch.setattr(bz, "_commit_rate", [])
    logged = []

    def capture(*args, **kwargs):
        logged.append(args[3])

    monkeypatch.setattr(bz, "log_payment", capture)

    fast = asyncio.run(bz.evaluate_fast(bz.EvaluateRequest(response="hello from the agent")))
    strict = asyncio.run(bz.evaluate_strict(bz.EvaluateRequest(response="hello from the agent")))

    assert logged == ["/evaluate/fast", "/evaluate/strict"]
    fast_events = find_by_tx_hash(chain._conn, fast.tx_hash)
    assert len(fast_events) == 1
    assert fast_events[0]["producer"] == "bazaar"
    assert fast_events[0]["route"] == "/evaluate/fast"
    assert fast_events[0]["proof"] == {"tx_hash": fast.tx_hash}
    assert find_by_tx_hash(chain._conn, strict.tx_hash) == []
    assert '_process_evaluation(req, "default", "strict")' in inspect.getsource(bz.evaluate_strict)


def test_scan_batch_and_audit_handlers_follow_webhook_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(bz, "_chain", ChainState(str(tmp_path / "chain.db")))
    monkeypatch.setattr(bz, "_commit_rate", [])

    secrets = asyncio.run(bz.evaluate_secrets(bz.ScanRequest(response="hello from the agent")))
    assert secrets.verdict == "COMMIT"
    assert secrets.risk_score == 0.0
    assert secrets.findings == []
    assert secrets.categories_checked == ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    assert secrets.seal_text
    assert secrets.verify_url.startswith("https://x402.fronesislabs.com/verify/")

    pii = asyncio.run(bz.evaluate_pii(bz.ScanRequest(response="hello from the agent")))
    assert pii.verdict == "COMMIT"
    assert pii.categories_checked == ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]

    with pytest.raises(HTTPException) as limited:
        asyncio.run(bz.evaluate_batch(bz.BatchEvaluateRequest(
            items=[bz.BatchItem(response="one"), bz.BatchItem(response="two")],
            agent_id="batch-agent",
            max_items=1,
        )))
    assert limited.value.status_code == 400

    batch = asyncio.run(bz.evaluate_batch(bz.BatchEvaluateRequest(
        items=[bz.BatchItem(response="hello from the agent")],
        agent_id="batch-agent",
    )))
    assert batch.count == 1
    assert batch.agent_id == "batch-agent"
    assert batch.results[0].verdict == "COMMIT"
    assert batch.results[0].seal_text
    assert len(batch.batch_id) == 8

    decoded = asyncio.run(bz.audit_decode(batch.results[0].tx_hash))
    assert decoded["tx_hash"] == batch.results[0].tx_hash
    assert decoded["agent_id"] == "batch-agent"
    assert decoded["verdict"] == "COMMIT"
    assert "tampered_at_index" not in decoded

    deep = asyncio.run(bz.audit_decode_deep(secrets.tx_hash))
    assert deep["tx_hash"] == secrets.tx_hash
    assert "drift_context" in deep
    assert "tampered_at_index" in deep

    with pytest.raises(HTTPException) as missing:
        asyncio.run(bz.audit_decode("0xmissing"))
    assert missing.value.status_code == 404
