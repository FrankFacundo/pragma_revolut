from __future__ import annotations

from functools import partial

import torch

from pragma.config import MaskingConfig, model_config_for_variant
from pragma.data.dataset import BankingEventDataset, collate_downstream, collate_pretrain
from pragma.data.synthetic import generate_synthetic_dataset
from pragma.data.tokenizer import PragmaTokenizer
from pragma.model.lora import apply_lora, lora_trainable_parameters
from pragma.model.pragma import PragmaForMaskedModeling, PragmaForSequenceClassification
from pragma.train.losses import masked_mlm_loss


def test_generate_tokenize_and_pretrain_forward(tmp_path):
    data_dir = tmp_path / "data"
    generate_synthetic_dataset(data_dir, users=24, min_events=4, max_events=12, seed=3)
    tokenizer = PragmaTokenizer.fit(
        data_dir,
        numeric_bins=8,
        text_vocab_size=128,
        bpe_merges=64,
    )
    assert tokenizer.vocab_size > 32

    config = model_config_for_variant(
        "tiny",
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        unk_token_id=tokenizer.unk_token_id,
        usr_token_id=tokenizer.usr_token_id,
        evt_token_id=tokenizer.evt_token_id,
    )
    config.max_events = 16
    dataset = BankingEventDataset(data_dir, tokenizer, split="train", max_events=16)
    records = [dataset[0], dataset[1]]
    batch = collate_pretrain(
        records,
        tokenizer=tokenizer,
        config=config,
        mask_cfg=MaskingConfig(token_prob=0.5, event_prob=0.1, key_prob=0.1),
    )
    model = PragmaForMaskedModeling(config)
    out = model(batch)
    assert out["mlm_logits"].shape[:3] == batch["event_value_ids"].shape
    loss = masked_mlm_loss(out["mlm_logits"], batch["mlm_labels"])
    assert torch.isfinite(loss)


def test_downstream_lora_forward(tmp_path):
    data_dir = tmp_path / "data"
    generate_synthetic_dataset(data_dir, users=20, min_events=4, max_events=10, seed=4)
    tokenizer = PragmaTokenizer.fit(data_dir, numeric_bins=8, text_vocab_size=128, bpe_merges=32)
    config = model_config_for_variant("tiny", vocab_size=tokenizer.vocab_size)
    config.pad_token_id = tokenizer.pad_token_id
    config.mask_token_id = tokenizer.mask_token_id
    config.unk_token_id = tokenizer.unk_token_id
    config.usr_token_id = tokenizer.usr_token_id
    config.evt_token_id = tokenizer.evt_token_id
    config.max_events = 16

    dataset = BankingEventDataset(data_dir, tokenizer, split="train", max_events=16)
    records = [dataset[0], dataset[1], dataset[2]]
    collate = partial(collate_downstream, tokenizer=tokenizer, config=config, task="product_rec")
    batch = collate(records)
    model = PragmaForSequenceClassification(config, num_labels=3)
    apply_lora(model, rank=2, alpha=2)
    trainable, total, pct = lora_trainable_parameters(model)
    assert 0 < trainable < total
    assert pct < 50
    out = model(batch)
    assert out["logits"].shape == (3, 3)
