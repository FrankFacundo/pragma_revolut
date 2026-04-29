from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sqlalchemy as sa

from pragma.data.schema import EVENT_COLUMNS, PROFILE_COLUMNS, USER_COLUMNS


@dataclass(slots=True)
class GenerationSummary:
    users: int
    events: int
    profiles: int
    out_dir: Path
    mysql_uri: str | None = None


COUNTRIES = ["GB", "FR", "DE", "ES", "IE", "NL", "PL", "RO", "LT", "US"]
REGIONS = ["uk", "eea", "us", "global"]
PLANS = ["standard", "plus", "premium", "metal", "ultra"]
AGE_BANDS = ["18-24", "25-35", "36-45", "46-60", "60+"]
DEVICES = ["iphone18,2", "iphone17,1", "pixel9", "samsung_s25", "web", "android_low"]
CURRENCIES = ["gbp", "eur", "usd", "pln"]
SEGMENTS = ["student", "low_income", "middle", "affluent", "ultra_hnw"]
RISK_PERSONAS = ["normal", "thin_file", "credit_hungry", "fraud_ring", "dormant", "traveller"]
MCCS = ["5411", "5812", "6012", "7995", "4111", "4814", "5732", "5942", "7011", "4899"]
MERCHANTS = [
    "metal plan",
    "city grocery",
    "metro coffee",
    "streaming subscription",
    "gym monthly",
    "crypto exchange",
    "airport lounge",
    "utilities direct debit",
    "marketplace order",
    "gaming wallet",
]
VIEWS = [
    "home",
    "p2p_amount",
    "confirm_p2p_dialog",
    "junior_transfer",
    "card_controls",
    "stocks_search",
    "credit_offer",
    "subscriptions",
]
PRODUCTS = ["credit", "stocks_shares_isa", "crypto", "savings", "junior", "insurance"]
SYMBOLS = ["swda", "vwrl", "aapl", "nvda", "tsla", "btc", "eth"]


def generate_synthetic_dataset(
    out_dir: str | Path,
    *,
    users: int = 10_000,
    min_events: int = 8,
    max_events: int = 256,
    seed: int = 13,
    mysql_uri: str | None = None,
    mysql_if_exists: str = "replace",
) -> GenerationSummary:
    """Generate PRAGMA-ready synthetic banking data.

    The output is deliberately shaped as record-level histories with static profile
    state and heterogeneous event rows, mirroring the paper while remaining fully
    synthetic and safe to publish.
    """

    if users <= 0:
        raise ValueError("users must be positive")
    if min_events <= 0 or max_events < min_events:
        raise ValueError("Require 0 < min_events <= max_events")

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    user_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    base_eval = datetime(2025, 7, 22, 13, 49, 29, tzinfo=UTC)
    event_id = 1
    for user_idx in range(users):
        user_id = f"user_{user_idx:08d}"
        segment = _weighted_choice(
            rng,
            {
                "student": 0.18,
                "low_income": 0.22,
                "middle": 0.42,
                "affluent": 0.15,
                "ultra_hnw": 0.03,
            },
        )
        risk_persona = _weighted_choice(
            rng,
            {
                "normal": 0.72,
                "thin_file": 0.08,
                "credit_hungry": 0.08,
                "fraud_ring": 0.025,
                "dormant": 0.055,
                "traveller": 0.04,
            },
        )
        country = rng.choice(COUNTRIES)
        region = _region_for(country)
        currency = _currency_for(country)
        plan = _plan_for(segment, rng)
        age_band = _age_for(segment, rng)
        device = rng.choice(DEVICES)
        insurance_state = _weighted_choice(
            rng, {"none": 0.65, "quote": 0.12, "active": 0.18, "cancelled": 0.05}
        )
        evaluation_time = base_eval - timedelta(days=rng.randint(0, 90), hours=rng.randint(0, 23))
        account_created_at = evaluation_time - timedelta(days=rng.randint(20, 1800))
        event_count = _event_count_for(rng, np_rng, segment, risk_persona, min_events, max_events)
        balance = _balance_for(segment, risk_persona, np_rng)
        balance_quantile = _balance_quantile(balance)

        user_events, event_id = _generate_events(
            rng=rng,
            np_rng=np_rng,
            user_id=user_id,
            first_event_id=event_id,
            count=event_count,
            account_created_at=account_created_at,
            evaluation_time=evaluation_time,
            segment=segment,
            risk_persona=risk_persona,
            currency=currency,
            plan=plan,
        )
        event_rows.extend(user_events)
        labels = _labels_for(user_events, segment, risk_persona, plan, balance, rng)

        user_row = {
            "user_id": user_id,
            "evaluation_time": evaluation_time.replace(tzinfo=None),
            "country": country,
            "region": region,
            "plan": plan,
            "age_band": age_band,
            "device": device,
            "currency": currency,
            "balance": float(round(balance, 2)),
            "balance_quantile": balance_quantile,
            "account_created_at": account_created_at.replace(tzinfo=None),
            "insurance_state": insurance_state,
            "segment": segment,
            "risk_persona": risk_persona,
            **labels,
        }
        user_rows.append(user_row)
        profile_rows.extend(_profile_rows(user_row, user_events))

    users_df = pd.DataFrame(user_rows, columns=USER_COLUMNS)
    events_df = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    profiles_df = pd.DataFrame(profile_rows, columns=PROFILE_COLUMNS)

    users_df.to_parquet(out / "users.parquet", index=False)
    events_df.to_parquet(out / "events.parquet", index=False)
    profiles_df.to_parquet(out / "profiles.parquet", index=False)

    if mysql_uri:
        _write_mysql(mysql_uri, users_df, events_df, profiles_df, if_exists=mysql_if_exists)

    return GenerationSummary(
        users=len(users_df),
        events=len(events_df),
        profiles=len(profiles_df),
        out_dir=out,
        mysql_uri=mysql_uri,
    )


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    total = sum(weights.values())
    draw = rng.random() * total
    acc = 0.0
    for key, weight in weights.items():
        acc += weight
        if draw <= acc:
            return key
    return next(reversed(weights))


def _region_for(country: str) -> str:
    if country == "GB":
        return "uk"
    if country == "US":
        return "us"
    if country in {"FR", "DE", "ES", "IE", "NL", "PL", "RO", "LT"}:
        return "eea"
    return "global"


def _currency_for(country: str) -> str:
    return {
        "GB": "gbp",
        "US": "usd",
        "PL": "pln",
    }.get(country, "eur")


def _plan_for(segment: str, rng: random.Random) -> str:
    if segment == "ultra_hnw":
        return _weighted_choice(rng, {"metal": 0.4, "ultra": 0.5, "premium": 0.1})
    if segment == "affluent":
        return _weighted_choice(rng, {"premium": 0.4, "metal": 0.35, "ultra": 0.05, "plus": 0.2})
    if segment == "student":
        return _weighted_choice(rng, {"standard": 0.7, "plus": 0.25, "premium": 0.05})
    return _weighted_choice(rng, {"standard": 0.45, "plus": 0.28, "premium": 0.18, "metal": 0.09})


def _age_for(segment: str, rng: random.Random) -> str:
    if segment == "student":
        return _weighted_choice(rng, {"18-24": 0.7, "25-35": 0.25, "36-45": 0.05})
    if segment == "ultra_hnw":
        return _weighted_choice(rng, {"36-45": 0.25, "46-60": 0.45, "60+": 0.3})
    return rng.choice(AGE_BANDS)


def _event_count_for(
    rng: random.Random,
    np_rng: np.random.Generator,
    segment: str,
    risk_persona: str,
    min_events: int,
    max_events: int,
) -> int:
    mean = {
        "student": 35,
        "low_income": 25,
        "middle": 55,
        "affluent": 95,
        "ultra_hnw": 160,
    }[segment]
    if risk_persona == "dormant":
        mean *= 0.25
    if risk_persona == "fraud_ring":
        mean *= 1.8
    count = int(np_rng.lognormal(mean=math.log(max(mean, 2)), sigma=0.65))
    count += rng.randint(-4, 4)
    return max(min_events, min(max_events, count))


def _balance_for(segment: str, risk_persona: str, np_rng: np.random.Generator) -> float:
    loc = {
        "student": 350,
        "low_income": 600,
        "middle": 2500,
        "affluent": 12000,
        "ultra_hnw": 65000,
    }[segment]
    if risk_persona in {"thin_file", "credit_hungry"}:
        loc *= 0.45
    if risk_persona == "fraud_ring":
        loc *= 0.20
    return float(max(-1500.0, np_rng.normal(loc, max(150.0, loc * 0.35))))


def _balance_quantile(balance: float) -> int:
    boundaries = [-500, 0, 250, 750, 1500, 3000, 7000, 15000, 40000]
    return sum(balance > b for b in boundaries)


def _generate_events(
    *,
    rng: random.Random,
    np_rng: np.random.Generator,
    user_id: str,
    first_event_id: int,
    count: int,
    account_created_at: datetime,
    evaluation_time: datetime,
    segment: str,
    risk_persona: str,
    currency: str,
    plan: str,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    event_id = first_event_id
    span_seconds = max(1, int((evaluation_time - account_created_at).total_seconds()))
    recurrent_merchants = rng.sample(["streaming subscription", "gym monthly", "utilities direct debit"], k=2)

    for idx in range(count):
        created_at = account_created_at + timedelta(seconds=int(rng.random() * span_seconds))
        event_type = _event_type_for(rng, segment, risk_persona)
        row = {column: None for column in EVENT_COLUMNS}
        row.update(
            {
                "user_id": user_id,
                "event_id": event_id,
                "created_at": created_at.replace(tzinfo=None),
                "event_type": event_type,
                "currency": currency,
            }
        )
        event_id += 1

        if event_type == "topup":
            amount = _amount(np_rng, segment, multiplier=2.5)
            row.update(
                {
                    "direction": "in",
                    "amount": amount,
                    "fee": 0.0,
                    "description": "salary topup" if rng.random() < 0.55 else "bank transfer topup",
                }
            )
        elif event_type == "card_payment":
            merchant = rng.choice(MERCHANTS)
            if idx % 30 == 0 and recurrent_merchants:
                merchant = rng.choice(recurrent_merchants)
            if risk_persona == "fraud_ring" and rng.random() < 0.18:
                merchant = rng.choice(["crypto exchange", "gaming wallet"])
            amount = _amount(np_rng, segment, multiplier=0.55)
            row.update(
                {
                    "direction": "out",
                    "amount": amount,
                    "fee": round(amount * rng.choice([0.0, 0.002, 0.005]), 2),
                    "description": merchant,
                    "merchant": merchant,
                    "mcc": _mcc_for(merchant, rng),
                    "is_recurrent": str(merchant in recurrent_merchants).lower(),
                }
            )
        elif event_type == "p2p_transfer":
            direction = rng.choice(["in", "out"])
            amount = _amount(np_rng, segment, multiplier=0.9)
            row.update(
                {
                    "direction": direction,
                    "amount": amount,
                    "fee": 0.0,
                    "description": rng.choice(["rent split", "dinner split", "family transfer", "holiday fund"]),
                    "counterparty_region": rng.choice(REGIONS),
                }
            )
        elif event_type == "app_event":
            row.update({"view": rng.choice(VIEWS)})
        elif event_type == "communication":
            product = rng.choice(PRODUCTS)
            interacted = _communication_interaction(rng, product, plan, segment)
            row.update(
                {
                    "channel": rng.choice(["email", "push", "inbox", "sms"]),
                    "product": product,
                    "interact": "interacted" if interacted else "ignored",
                    "description": f"{product} campaign",
                }
            )
        elif event_type == "trading":
            amount = _amount(np_rng, segment, multiplier=1.7)
            row.update(
                {
                    "direction": rng.choice(["buy", "sell"]),
                    "amount": amount,
                    "price": round(float(np_rng.lognormal(mean=4.2, sigma=0.7)), 4),
                    "symbol": rng.choice(SYMBOLS),
                    "order_type": rng.choice(["market", "limit", "recurring"]),
                    "description": "investment order",
                }
            )
        rows.append(row)

    rows.sort(key=lambda x: (x["created_at"], x["event_id"]))
    for new_idx, row in enumerate(rows):
        row["event_id"] = first_event_id + new_idx
    return rows, first_event_id + len(rows)


def _event_type_for(rng: random.Random, segment: str, risk_persona: str) -> str:
    weights = {
        "card_payment": 0.45,
        "app_event": 0.18,
        "topup": 0.11,
        "p2p_transfer": 0.11,
        "communication": 0.10,
        "trading": 0.05,
    }
    if segment in {"affluent", "ultra_hnw"}:
        weights["trading"] += 0.09
        weights["card_payment"] -= 0.04
    if risk_persona == "fraud_ring":
        weights["p2p_transfer"] += 0.10
        weights["card_payment"] += 0.06
        weights["communication"] -= 0.04
    if risk_persona == "dormant":
        weights["app_event"] += 0.16
        weights["card_payment"] -= 0.12
    return _weighted_choice(rng, weights)


def _amount(np_rng: np.random.Generator, segment: str, multiplier: float) -> float:
    base = {
        "student": 18,
        "low_income": 28,
        "middle": 55,
        "affluent": 130,
        "ultra_hnw": 420,
    }[segment]
    value = np_rng.lognormal(mean=math.log(base * multiplier), sigma=0.75)
    return float(round(min(value, 50_000.0), 2))


def _mcc_for(merchant: str, rng: random.Random) -> str:
    if "grocery" in merchant:
        return "5411"
    if "coffee" in merchant:
        return "5812"
    if "subscription" in merchant:
        return "4899"
    if "gym" in merchant:
        return "7995"
    if "crypto" in merchant:
        return "6012"
    if "gaming" in merchant:
        return "7995"
    return rng.choice(MCCS)


def _communication_interaction(
    rng: random.Random, product: str, plan: str, segment: str
) -> bool:
    p = 0.06
    if product in {"savings", "stocks_shares_isa"} and segment in {"affluent", "ultra_hnw"}:
        p += 0.14
    if product == "credit" and segment in {"low_income", "middle"}:
        p += 0.08
    if plan in {"metal", "ultra"}:
        p += 0.04
    return rng.random() < p


def _labels_for(
    events: list[dict[str, Any]],
    segment: str,
    risk_persona: str,
    plan: str,
    balance: float,
    rng: random.Random,
) -> dict[str, int]:
    outbound = sum(float(e["amount"] or 0.0) for e in events if e.get("direction") == "out")
    inbound = sum(float(e["amount"] or 0.0) for e in events if e.get("direction") == "in")
    gambling_or_crypto = sum(
        1
        for e in events
        if e.get("mcc") in {"7995", "6012"} or str(e.get("description") or "").find("crypto") >= 0
    )
    interacted_products = {
        str(e.get("product"))
        for e in events
        if e.get("event_type") == "communication" and e.get("interact") == "interacted"
    }
    recurrent = any(e.get("is_recurrent") == "true" for e in events)

    credit_risk = 0.08
    if risk_persona in {"credit_hungry", "thin_file"}:
        credit_risk += 0.22
    if balance < 250:
        credit_risk += 0.18
    if outbound > inbound * 1.35:
        credit_risk += 0.08
    if gambling_or_crypto > 5:
        credit_risk += 0.06
    if segment in {"affluent", "ultra_hnw"}:
        credit_risk -= 0.07

    fraud_risk = 0.015 + (0.62 if risk_persona == "fraud_ring" else 0.0)
    fraud_risk += min(0.12, gambling_or_crypto * 0.01)

    ltv_risk = 0.18
    if plan in {"premium", "metal", "ultra"}:
        ltv_risk += 0.28
    if segment in {"affluent", "ultra_hnw"}:
        ltv_risk += 0.30
    if len(events) > 80:
        ltv_risk += 0.10
    if risk_persona == "dormant":
        ltv_risk -= 0.22

    return {
        "label_credit_default": int(rng.random() < max(0.01, min(0.85, credit_risk))),
        "label_external_fraud": int(rng.random() < max(0.005, min(0.9, fraud_risk))),
        "label_comm_engagement": int(bool(interacted_products) or rng.random() < 0.05),
        "label_ltv": int(rng.random() < max(0.02, min(0.95, ltv_risk))),
        "label_recurrent_txn": int(recurrent),
        "label_product_savings": int("savings" in interacted_products or (balance > 3500 and rng.random() < 0.25)),
        "label_product_crypto": int("crypto" in interacted_products or gambling_or_crypto > 3),
        "label_product_credit": int("credit" in interacted_products or risk_persona == "credit_hungry"),
    }


def _profile_rows(user_row: dict[str, Any], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    user_id = str(user_row["user_id"])
    eval_time = user_row["evaluation_time"]
    rows = []
    static_keys = [
        "country",
        "region",
        "plan",
        "age_band",
        "device",
        "currency",
        "balance",
        "balance_quantile",
        "insurance_state",
        "segment",
        "risk_persona",
    ]
    for key in static_keys:
        rows.append(
            {
                "user_id": user_id,
                "key": key,
                "value": str(user_row[key]),
                "timestamp": eval_time,
                "is_lifelong": False,
            }
        )
    rows.append(
        {
            "user_id": user_id,
            "key": "lifelong_account_created",
            "value": "true",
            "timestamp": user_row["account_created_at"],
            "is_lifelong": True,
        }
    )
    for event_type, key in [
        ("topup", "lifelong_first_topup"),
        ("card_payment", "lifelong_first_card_payment"),
        ("trading", "lifelong_first_trading"),
    ]:
        first = next((e for e in events if e["event_type"] == event_type), None)
        if first:
            rows.append(
                {
                    "user_id": user_id,
                    "key": key,
                    "value": "true",
                    "timestamp": first["created_at"],
                    "is_lifelong": True,
                }
            )
    return rows


def _write_mysql(
    mysql_uri: str,
    users_df: pd.DataFrame,
    events_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    *,
    if_exists: str,
) -> None:
    if if_exists not in {"fail", "replace", "append"}:
        raise ValueError("mysql_if_exists must be one of: fail, replace, append")
    engine = sa.create_engine(mysql_uri, pool_pre_ping=True)
    with engine.begin() as conn:
        users_df.to_sql("pragma_users", conn, index=False, if_exists=if_exists, chunksize=10_000)
        events_df.to_sql("pragma_events", conn, index=False, if_exists=if_exists, chunksize=10_000)
        profiles_df.to_sql("pragma_profiles", conn, index=False, if_exists=if_exists, chunksize=10_000)
