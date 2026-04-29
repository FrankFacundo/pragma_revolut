from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from pragma.config import MaskingConfig, ModelConfig
from pragma.data.tokenizer import EncodedRecord, PragmaTokenizer

TASK_LABELS = {
    "credit_default": ["credit_default"],
    "external_fraud": ["external_fraud"],
    "comm_engagement": ["comm_engagement"],
    "ltv": ["ltv"],
    "recurrent_txn": ["recurrent_txn"],
    "product_rec": ["product_savings", "product_crypto", "product_credit"],
}


class BankingEventDataset(Dataset[EncodedRecord]):
    """In-memory record-level dataset backed by generated Parquet files."""

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer: PragmaTokenizer,
        *,
        split: str = "train",
        val_fraction: float = 0.10,
        test_fraction: float = 0.10,
        seed: int = 13,
        max_events: int = 6500,
        max_event_tokens: int = 24,
        max_profile_tokens: int = 200,
    ) -> None:
        if split not in {"train", "val", "test", "all"}:
            raise ValueError("split must be one of train, val, test, all")
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_events = max_events
        self.max_event_tokens = max_event_tokens
        self.max_profile_tokens = max_profile_tokens

        users = pd.read_parquet(self.data_dir / "users.parquet")
        self.events = pd.read_parquet(self.data_dir / "events.parquet")
        self.profiles = pd.read_parquet(self.data_dir / "profiles.parquet")
        users = users.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if split != "all":
            n = len(users)
            n_test = int(n * test_fraction)
            n_val = int(n * val_fraction)
            if split == "test":
                users = users.iloc[:n_test]
            elif split == "val":
                users = users.iloc[n_test : n_test + n_val]
            else:
                users = users.iloc[n_test + n_val :]
        self.users = users.reset_index(drop=True)
        self._events_by_user = {
            user_id: df.copy()
            for user_id, df in self.events.groupby("user_id", sort=False)
        }
        self._profiles_by_user = {
            user_id: df.copy()
            for user_id, df in self.profiles.groupby("user_id", sort=False)
        }

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> EncodedRecord:
        user = self.users.iloc[idx]
        user_id = str(user["user_id"])
        events = self._events_by_user.get(user_id, self.events.iloc[0:0])
        profiles = self._profiles_by_user.get(user_id, self.profiles.iloc[0:0])
        return self.tokenizer.encode_record(
            user,
            events,
            profiles,
            max_events=self.max_events,
            max_event_tokens=self.max_event_tokens,
            max_profile_tokens=self.max_profile_tokens,
        )


def collate_records(
    records: list[EncodedRecord],
    *,
    tokenizer: PragmaTokenizer,
    config: ModelConfig,
) -> dict[str, Any]:
    batch_size = len(records)
    max_profile = max(1, min(config.max_profile_tokens, max(len(r.profile.key_ids) for r in records)))
    max_events = max(1, min(config.max_events, max(len(r.events) for r in records)))
    max_event_tokens = max(
        1,
        min(
            config.max_field_tokens,
            max((len(event.key_ids) for record in records for event in record.events), default=1),
        ),
    )
    pad = tokenizer.pad_token_id

    profile_key_ids = torch.full((batch_size, max_profile), pad, dtype=torch.long)
    profile_value_ids = torch.full((batch_size, max_profile), pad, dtype=torch.long)
    profile_positions = torch.zeros((batch_size, max_profile), dtype=torch.long)
    profile_times = torch.zeros((batch_size, max_profile), dtype=torch.float32)
    profile_mask = torch.zeros((batch_size, max_profile), dtype=torch.bool)

    event_key_ids = torch.full((batch_size, max_events, max_event_tokens), pad, dtype=torch.long)
    event_value_ids = torch.full((batch_size, max_events, max_event_tokens), pad, dtype=torch.long)
    event_positions = torch.zeros((batch_size, max_events, max_event_tokens), dtype=torch.long)
    event_token_mask = torch.zeros((batch_size, max_events, max_event_tokens), dtype=torch.bool)
    event_times = torch.zeros((batch_size, max_events), dtype=torch.float32)
    calendar = torch.zeros((batch_size, max_events, 3), dtype=torch.float32)
    event_mask = torch.zeros((batch_size, max_events), dtype=torch.bool)
    user_ids: list[str] = []
    labels: list[dict[str, int | float]] = []

    for b, record in enumerate(records):
        user_ids.append(record.user_id)
        labels.append(record.labels)
        p_len = min(max_profile, len(record.profile.key_ids))
        if p_len:
            profile_key_ids[b, :p_len] = torch.tensor(record.profile.key_ids[:p_len])
            profile_value_ids[b, :p_len] = torch.tensor(record.profile.value_ids[:p_len])
            profile_positions[b, :p_len] = torch.tensor(record.profile.value_positions[:p_len])
            profile_times[b, :p_len] = torch.tensor(record.profile.time_since[:p_len])
            profile_mask[b, :p_len] = True

        for e_idx, event in enumerate(record.events[:max_events]):
            t_len = min(max_event_tokens, len(event.key_ids))
            if not t_len:
                continue
            event_key_ids[b, e_idx, :t_len] = torch.tensor(event.key_ids[:t_len])
            event_value_ids[b, e_idx, :t_len] = torch.tensor(event.value_ids[:t_len])
            event_positions[b, e_idx, :t_len] = torch.tensor(event.value_positions[:t_len])
            event_token_mask[b, e_idx, :t_len] = True
            event_times[b, e_idx] = float(event.time_to_last)
            calendar[b, e_idx] = torch.tensor(event.calendar, dtype=torch.float32)
            event_mask[b, e_idx] = True

    return {
        "profile_key_ids": profile_key_ids,
        "profile_value_ids": profile_value_ids,
        "profile_positions": profile_positions,
        "profile_times": profile_times,
        "profile_mask": profile_mask,
        "event_key_ids": event_key_ids,
        "event_value_ids": event_value_ids,
        "event_positions": event_positions,
        "event_token_mask": event_token_mask,
        "event_times": event_times,
        "calendar": calendar,
        "event_mask": event_mask,
        "user_ids": user_ids,
        "labels_dict": labels,
    }


def apply_mlm_masking(
    batch: dict[str, Any],
    *,
    mask_cfg: MaskingConfig,
    mask_token_id: int,
    unk_token_id: int,
) -> dict[str, Any]:
    values = batch["event_value_ids"].clone()
    labels = torch.full_like(values, -100)
    token_mask = batch["event_token_mask"]
    selected = (torch.rand_like(values.float()) < mask_cfg.token_prob) & token_mask

    event_selected = (torch.rand(values.shape[:2]) < mask_cfg.event_prob) & batch["event_mask"]
    selected |= event_selected.unsqueeze(-1) & token_mask

    keys = batch["event_key_ids"]
    for b in range(values.shape[0]):
        for e in range(values.shape[1]):
            if not bool(batch["event_mask"][b, e]):
                continue
            unique_keys = torch.unique(keys[b, e][token_mask[b, e]])
            for key_id in unique_keys.tolist():
                if torch.rand(()) < mask_cfg.key_prob:
                    selected[b, e] |= (keys[b, e] == key_id) & token_mask[b, e]

    labels[selected] = values[selected]
    replace_draw = torch.rand_like(values.float())
    replace_mask = selected & (replace_draw < mask_cfg.mask_replace_prob)
    replace_unk = selected & (
        replace_draw >= mask_cfg.mask_replace_prob
    ) & (replace_draw < mask_cfg.mask_replace_prob + mask_cfg.unk_replace_prob)

    values[replace_mask] = mask_token_id
    values[replace_unk] = unk_token_id
    labels[replace_unk] = -100
    batch = dict(batch)
    batch["event_value_ids"] = values
    batch["mlm_labels"] = labels
    return batch


def collate_pretrain(
    records: list[EncodedRecord],
    *,
    tokenizer: PragmaTokenizer,
    config: ModelConfig,
    mask_cfg: MaskingConfig,
) -> dict[str, Any]:
    batch = collate_records(records, tokenizer=tokenizer, config=config)
    return apply_mlm_masking(
        batch,
        mask_cfg=mask_cfg,
        mask_token_id=tokenizer.mask_token_id,
        unk_token_id=tokenizer.unk_token_id,
    )


def collate_downstream(
    records: list[EncodedRecord],
    *,
    tokenizer: PragmaTokenizer,
    config: ModelConfig,
    task: str,
) -> dict[str, Any]:
    if task not in TASK_LABELS:
        raise ValueError(f"Unknown task {task}. Expected one of {sorted(TASK_LABELS)}")
    batch = collate_records(records, tokenizer=tokenizer, config=config)
    label_names = TASK_LABELS[task]
    labels = [
        [float(record.labels.get(label, 0.0)) for label in label_names]
        for record in records
    ]
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    if len(label_names) == 1:
        label_tensor = label_tensor.squeeze(-1)
    batch["labels"] = label_tensor
    return batch


def build_dataloader(
    dataset: BankingEventDataset,
    collate_fn: Callable[[list[EncodedRecord]], dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
