from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from pragma.data.schema import (
    CATEGORICAL_KEYS,
    EVENT_COLUMNS,
    NUMERIC_KEYS,
    PROFILE_KEYS,
    TEXT_KEYS,
)

SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[UNK]", "[USR]", "[EVT]"]
IGNORED_EVENT_COLUMNS = {"user_id", "event_id", "created_at"}
TOKEN_RE = re.compile(r"[a-z0-9]+|[^\w\s]", re.IGNORECASE)


@dataclass(slots=True)
class EncodedEvent:
    key_ids: list[int]
    value_ids: list[int]
    value_positions: list[int]
    time_to_last: float
    calendar: tuple[int, int, int]


@dataclass(slots=True)
class EncodedProfile:
    key_ids: list[int]
    value_ids: list[int]
    value_positions: list[int]
    time_since: list[float]


@dataclass(slots=True)
class EncodedRecord:
    user_id: str
    profile: EncodedProfile
    events: list[EncodedEvent]
    labels: dict[str, int | float]


class SimpleBPE:
    """Small deterministic BPE-style tokenizer used only for free-text fields."""

    def __init__(
        self,
        *,
        merges: list[tuple[str, str]] | None = None,
        pieces: set[str] | None = None,
        lowercase: bool = True,
    ) -> None:
        self.merges = merges or []
        self.pieces = pieces or set()
        self.lowercase = lowercase
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        max_pieces: int = 4096,
        num_merges: int = 2048,
        min_frequency: int = 2,
    ) -> "SimpleBPE":
        word_counts: Counter[tuple[str, ...]] = Counter()
        for text in texts:
            for word in _basic_tokens(text):
                word_counts[tuple(word) + ("</w>",)] += 1

        merges: list[tuple[str, str]] = []
        vocab = dict(word_counts)
        for _ in range(num_merges):
            pair_counts: Counter[tuple[str, str]] = Counter()
            for word, freq in vocab.items():
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i + 1])] += freq
            if not pair_counts:
                break
            pair, freq = pair_counts.most_common(1)[0]
            if freq < min_frequency:
                break
            merges.append(pair)
            vocab = {_merge_word(word, pair): count for word, count in vocab.items()}
            if len(_pieces_from_vocab(vocab)) >= max_pieces:
                break

        tokenizer = cls(merges=merges)
        piece_counts: Counter[str] = Counter()
        for text in texts:
            piece_counts.update(tokenizer.encode_to_pieces(text))
        pieces = {piece for piece, _ in piece_counts.most_common(max_pieces)}
        pieces.update([chr(i) for i in range(ord("a"), ord("z") + 1)])
        pieces.update([str(i) for i in range(10)])
        tokenizer.pieces = pieces
        tokenizer._merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        return tokenizer

    def encode_to_pieces(self, text: str) -> list[str]:
        pieces: list[str] = []
        for token in _basic_tokens(text.lower() if self.lowercase else text):
            word = tuple(token) + ("</w>",)
            word = self._apply_merges(word)
            for piece in word:
                if piece == "</w>":
                    continue
                if piece.endswith("</w>"):
                    piece = piece[: -len("</w>")]
                if piece:
                    pieces.append(piece)
        return pieces

    def to_dict(self) -> dict[str, Any]:
        return {
            "merges": [list(pair) for pair in self.merges],
            "pieces": sorted(self.pieces),
            "lowercase": self.lowercase,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimpleBPE":
        return cls(
            merges=[tuple(pair) for pair in data.get("merges", [])],
            pieces=set(data.get("pieces", [])),
            lowercase=bool(data.get("lowercase", True)),
        )

    def _apply_merges(self, word: tuple[str, ...]) -> tuple[str, ...]:
        current = word
        while True:
            candidates = [
                (self._merge_ranks[pair], pair)
                for pair in zip(current, current[1:], strict=False)
                if pair in self._merge_ranks
            ]
            if not candidates:
                return current
            _, pair = min(candidates)
            current = _merge_word(current, pair)


@dataclass(slots=True)
class NumericBucket:
    boundaries: list[float]

    def bucket(self, value: float) -> int:
        if value == 0:
            return 0
        return int(np.searchsorted(np.asarray(self.boundaries), value, side="right")) + 1


class PragmaTokenizer:
    """Key-value-time tokenizer for PRAGMA-style banking histories."""

    def __init__(
        self,
        *,
        token_to_id: dict[str, int] | None = None,
        numeric_buckets: dict[str, NumericBucket] | None = None,
        text_tokenizer: SimpleBPE | None = None,
        categorical_keys: set[str] | None = None,
        text_keys: set[str] | None = None,
        numeric_keys: set[str] | None = None,
    ) -> None:
        self.token_to_id = token_to_id or {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.numeric_buckets = numeric_buckets or {}
        self.text_tokenizer = text_tokenizer or SimpleBPE()
        self.categorical_keys = categorical_keys or set(CATEGORICAL_KEYS)
        self.text_keys = text_keys or set(TEXT_KEYS)
        self.numeric_keys = numeric_keys or set(NUMERIC_KEYS)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["[PAD]"]

    @property
    def mask_token_id(self) -> int:
        return self.token_to_id["[MASK]"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["[UNK]"]

    @property
    def usr_token_id(self) -> int:
        return self.token_to_id["[USR]"]

    @property
    def evt_token_id(self) -> int:
        return self.token_to_id["[EVT]"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @classmethod
    def fit(
        cls,
        data_dir: str | Path,
        *,
        numeric_bins: int = 128,
        categorical_threshold: int = 20_000,
        text_vocab_size: int = 4096,
        bpe_merges: int = 2048,
    ) -> "PragmaTokenizer":
        data = Path(data_dir)
        users = pd.read_parquet(data / "users.parquet")
        events = pd.read_parquet(data / "events.parquet")
        profiles = pd.read_parquet(data / "profiles.parquet")

        tokenizer = cls()
        all_keys = set(PROFILE_KEYS)
        all_keys.update(column for column in EVENT_COLUMNS if column not in IGNORED_EVENT_COLUMNS)
        for key in sorted(all_keys):
            tokenizer._add_token(_key_token(key))

        text_values: list[str] = []
        numeric_values: dict[str, list[float]] = defaultdict(list)

        for key in sorted(tokenizer.numeric_keys):
            if key in events.columns:
                numeric_values[key].extend(_clean_numeric(events[key]))
            if key in users.columns:
                numeric_values[key].extend(_clean_numeric(users[key]))
            profile_values = profiles.loc[profiles["key"] == key, "value"]
            if not profile_values.empty:
                numeric_values[key].extend(_clean_numeric(profile_values))

        for key, values in numeric_values.items():
            nonzero = np.asarray([value for value in values if np.isfinite(value) and value != 0])
            if nonzero.size == 0:
                boundaries: list[float] = []
            else:
                qs = np.linspace(0, 1, numeric_bins + 1)[1:-1]
                boundaries = sorted(set(float(x) for x in np.quantile(nonzero, qs)))
            tokenizer.numeric_buckets[key] = NumericBucket(boundaries=boundaries)
            for bucket_idx in range(len(boundaries) + 2):
                tokenizer._add_token(_numeric_token(key, bucket_idx))

        for key in sorted(tokenizer.categorical_keys):
            values = []
            if key in events.columns:
                values.extend(_clean_strings(events[key]))
            if key in users.columns:
                values.extend(_clean_strings(users[key]))
            values.extend(_clean_strings(profiles.loc[profiles["key"] == key, "value"]))
            counts = Counter(values)
            for value, _ in counts.most_common(categorical_threshold):
                tokenizer._add_token(_category_token(key, value))

        for key in sorted(tokenizer.text_keys):
            if key in events.columns:
                text_values.extend(_clean_strings(events[key]))
            text_values.extend(_clean_strings(profiles.loc[profiles["key"] == key, "value"]))
        tokenizer.text_tokenizer = SimpleBPE.train(
            text_values,
            max_pieces=text_vocab_size,
            num_merges=bpe_merges,
            min_frequency=2,
        )
        for piece in sorted(tokenizer.text_tokenizer.pieces):
            tokenizer._add_token(_text_token(piece))
        return tokenizer

    def encode_record(
        self,
        user_row: pd.Series | dict[str, Any],
        events: pd.DataFrame,
        profiles: pd.DataFrame,
        *,
        max_events: int = 6500,
        max_event_tokens: int = 24,
        max_profile_tokens: int = 200,
    ) -> EncodedRecord:
        user = dict(user_row)
        user_id = str(user["user_id"])
        evaluation_time = _to_datetime(user["evaluation_time"])
        events = events.sort_values("created_at").tail(max_events)
        profiles = profiles.sort_values(["is_lifelong", "key"], ascending=[False, True])

        profile = self.encode_profile(
            profiles,
            evaluation_time=evaluation_time,
            max_tokens=max_profile_tokens,
        )
        encoded_events = self.encode_events(
            events,
            max_event_tokens=max_event_tokens,
        )
        return EncodedRecord(
            user_id=user_id,
            profile=profile,
            events=encoded_events,
            labels=_labels_from_user(user),
        )

    def encode_profile(
        self,
        profiles: pd.DataFrame,
        *,
        evaluation_time: datetime,
        max_tokens: int,
    ) -> EncodedProfile:
        key_ids: list[int] = []
        value_ids: list[int] = []
        value_positions: list[int] = []
        time_since: list[float] = []
        for _, row in profiles.iterrows():
            key = str(row["key"])
            values = self.encode_value(key, row["value"])
            if not values:
                continue
            key_id = self.key_id(key)
            timestamp = _to_datetime(row["timestamp"])
            delta = max(0.0, (evaluation_time - timestamp).total_seconds())
            t = soft_log_seconds(delta) if bool(row.get("is_lifelong", False)) else 0.0
            for pos, value_id in enumerate(values):
                key_ids.append(key_id)
                value_ids.append(value_id)
                value_positions.append(pos)
                time_since.append(t)
                if len(key_ids) >= max_tokens:
                    return EncodedProfile(key_ids, value_ids, value_positions, time_since)
        return EncodedProfile(key_ids, value_ids, value_positions, time_since)

    def encode_events(self, events: pd.DataFrame, *, max_event_tokens: int) -> list[EncodedEvent]:
        if events.empty:
            return []
        created = pd.to_datetime(events["created_at"])
        last = _to_datetime(created.max())
        out: list[EncodedEvent] = []
        for idx, row in events.iterrows():
            event_time = _to_datetime(row["created_at"])
            key_ids: list[int] = []
            value_ids: list[int] = []
            value_positions: list[int] = []
            for key in _event_key_order():
                if key not in row.index:
                    continue
                raw_value = row[key]
                if _is_missing(raw_value):
                    continue
                values = self.encode_value(key, raw_value)
                if not values:
                    continue
                key_id = self.key_id(key)
                for pos, value_id in enumerate(values):
                    key_ids.append(key_id)
                    value_ids.append(value_id)
                    value_positions.append(pos)
                    if len(key_ids) >= max_event_tokens:
                        break
                if len(key_ids) >= max_event_tokens:
                    break
            time_to_last = soft_log_seconds(max(0.0, (last - event_time).total_seconds()))
            out.append(
                EncodedEvent(
                    key_ids=key_ids,
                    value_ids=value_ids,
                    value_positions=value_positions,
                    time_to_last=time_to_last,
                    calendar=(event_time.hour, event_time.weekday(), event_time.day),
                )
            )
        return out

    def encode_value(self, key: str, raw_value: Any) -> list[int]:
        if _is_missing(raw_value):
            return []
        if key in self.numeric_buckets:
            value = _to_float(raw_value)
            bucket = self.numeric_buckets[key].bucket(value)
            return [self.token_to_id.get(_numeric_token(key, bucket), self.unk_token_id)]
        if key in self.text_keys:
            ids = [
                self.token_to_id.get(_text_token(piece), self.unk_token_id)
                for piece in self.text_tokenizer.encode_to_pieces(str(raw_value))
            ]
            return ids or [self.unk_token_id]
        token = _category_token(key, str(raw_value).strip().lower())
        return [self.token_to_id.get(token, self.unk_token_id)]

    def key_id(self, key: str) -> int:
        return self.token_to_id.get(_key_token(key), self.unk_token_id)

    def decode_token(self, token_id: int) -> str:
        return self.id_to_token.get(int(token_id), "[UNK]")

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token_to_id": self.token_to_id,
            "numeric_buckets": {k: asdict(v) for k, v in self.numeric_buckets.items()},
            "text_tokenizer": self.text_tokenizer.to_dict(),
            "categorical_keys": sorted(self.categorical_keys),
            "text_keys": sorted(self.text_keys),
            "numeric_keys": sorted(self.numeric_keys),
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PragmaTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            token_to_id={k: int(v) for k, v in data["token_to_id"].items()},
            numeric_buckets={
                key: NumericBucket(boundaries=list(value.get("boundaries", [])))
                for key, value in data.get("numeric_buckets", {}).items()
            },
            text_tokenizer=SimpleBPE.from_dict(data.get("text_tokenizer", {})),
            categorical_keys=set(data.get("categorical_keys", CATEGORICAL_KEYS)),
            text_keys=set(data.get("text_keys", TEXT_KEYS)),
            numeric_keys=set(data.get("numeric_keys", NUMERIC_KEYS)),
        )

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.token_to_id)
            self.id_to_token[self.token_to_id[token]] = token
        return self.token_to_id[token]


def soft_log_seconds(seconds: float) -> float:
    return float(8.0 * math.log1p(max(0.0, seconds) / 8.0))


def _basic_tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(str(text)) if match.group(0).strip()]


def _merge_word(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    merged: list[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def _pieces_from_vocab(vocab: dict[tuple[str, ...], int]) -> set[str]:
    pieces: set[str] = set()
    for word in vocab:
        for piece in word:
            if piece != "</w>":
                pieces.add(piece.removesuffix("</w>"))
    return pieces


def _key_token(key: str) -> str:
    return f"key:{key}"


def _category_token(key: str, value: str) -> str:
    return f"cat:{key}:{value.strip().lower()}"


def _numeric_token(key: str, bucket: int) -> str:
    return f"num:{key}:bucket_{bucket:04d}"


def _text_token(piece: str) -> str:
    return f"txt:{piece}"


def _clean_numeric(series: pd.Series) -> list[float]:
    values = pd.to_numeric(series, errors="coerce")
    return [float(x) for x in values.dropna().tolist()]


def _clean_strings(series: pd.Series) -> list[str]:
    if series.empty:
        return []
    out = []
    for value in series.dropna().tolist():
        if _is_missing(value):
            continue
        text = str(value).strip().lower()
        if text:
            out.append(text)
    return out


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _to_float(value: Any) -> float:
    if _is_missing(value):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    return pd.to_datetime(value).to_pydatetime().replace(tzinfo=None)


def _event_key_order() -> list[str]:
    first = ["event_type", "direction", "amount", "currency", "fee", "description", "mcc"]
    rest = [column for column in EVENT_COLUMNS if column not in IGNORED_EVENT_COLUMNS and column not in first]
    return first + rest


def _labels_from_user(user: dict[str, Any]) -> dict[str, int | float]:
    return {
        key.removeprefix("label_"): int(value)
        for key, value in user.items()
        if key.startswith("label_") and not _is_missing(value)
    }
