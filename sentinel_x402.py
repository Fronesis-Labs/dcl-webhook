"""Dynamic x402 payment helper for Sentinel scan pricing (fastapi_x402)."""

from __future__ import annotations

from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from fastapi_x402.core import get_config, get_facilitator_client
from fastapi_x402.models import PaymentRequirements
from fastapi_x402.networks import get_default_asset_config

SCAN_PRICES = {
    "update_rescan": "$0.50",
    "deep_scan": "$2",
    "forensic_audit": "$10",
}


def scan_price(scan_type: str) -> str:
    if scan_type not in SCAN_PRICES:
        raise ValueError(
            f"Invalid scan_type '{scan_type}'. Expected one of: {', '.join(SCAN_PRICES)}"
        )
    return SCAN_PRICES[scan_type]


def _build_payment_requirements(request: Request, amount: str) -> PaymentRequirements:
    config = get_config()
    if isinstance(config.network, list):
        selected_network = config.network[0]
    else:
        selected_network = config.network
    asset_config = get_default_asset_config(selected_network)
    resource = f"{request.url.scheme}://{request.url.netloc}{request.url.path}"
    price_str = amount
    if price_str.startswith("$"):
        price_float = float(price_str[1:])
        atomic_amount = str(int(price_float * (10**asset_config.decimals)))
    else:
        atomic_amount = str(int(float(price_str) * (10**asset_config.decimals)))
    return PaymentRequirements(
        scheme="exact",
        network=selected_network,
        maxAmountRequired=atomic_amount,
        resource=resource,
        description="DCL Update Sentinel pay-per-call scan",
        mimeType="application/json",
        payTo=config.pay_to,
        maxTimeoutSeconds=config.default_expires_in or 300,
        asset=asset_config.address,
        extra={"name": asset_config.eip712_name, "version": asset_config.eip712_version},
    )


async def require_x402_payment(request: Request, amount: str) -> Optional[JSONResponse]:
    """Return a 402 JSONResponse if payment is missing/invalid; None if verified."""
    payment_header = request.headers.get("X-PAYMENT")
    if not payment_header:
        requirements = _build_payment_requirements(request, amount)
        return JSONResponse(
            status_code=402,
            content={
                "x402Version": 1,
                "error": "X-PAYMENT header is required",
                "accepts": [requirements.model_dump()],
            },
        )

    facilitator = get_facilitator_client()
    requirements = _build_payment_requirements(request, amount)
    verify_response = await facilitator.verify_payment(
        payment_header=payment_header,
        payment_requirements=requirements,
    )
    if not verify_response.isValid:
        return JSONResponse(
            status_code=402,
            content={
                "x402Version": 1,
                "error": verify_response.error or "Payment verification failed",
                "accepts": [requirements.model_dump()],
            },
        )

    if verify_response.payment_id:
        settle = await facilitator.settle_payment(
            payment_header=payment_header,
            payment_requirements=requirements,
        )
        if not settle.success:
            return JSONResponse(
                status_code=402,
                content={
                    "x402Version": 1,
                    "error": settle.errorReason or "Payment settlement failed",
                    "accepts": [requirements.model_dump()],
                },
            )
    request.state.payment_verified = True
    request.state.payment_payer = verify_response.payer
    return None
