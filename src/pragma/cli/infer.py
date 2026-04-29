from __future__ import annotations

import argparse
import json

import torch

from pragma.config import model_config_for_variant
from pragma.data.dataset import (
    TASK_LABELS,
    BankingEventDataset,
    collate_downstream,
    collate_records,
)
from pragma.data.tokenizer import PragmaTokenizer
from pragma.model.lora import apply_lora
from pragma.model.pragma import PragmaForSequenceClassification
from pragma.train.checkpoint import checkpoint_metadata, load_checkpoint
from pragma.train.device import move_batch, resolve_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PRAGMA embedding or prediction inference.")
    parser.add_argument("--data-dir", default="data/synth")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--variant", choices=["tiny", "s", "m", "l"], default="s")
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--task", choices=sorted(TASK_LABELS), default=None)
    parser.add_argument("--representation", choices=["usr", "last_event", "both"], default="both")
    parser.add_argument("--max-events", type=int, default=512)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    parser.add_argument("--lora", action="store_true", help="Force LoRA graph before loading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = resolve_runtime(args.device, None)
    tokenizer = PragmaTokenizer.load(args.tokenizer)
    config = model_config_for_variant(
        args.variant,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        unk_token_id=tokenizer.unk_token_id,
        usr_token_id=tokenizer.usr_token_id,
        evt_token_id=tokenizer.evt_token_id,
    )
    config.max_events = args.max_events
    task = args.task or "credit_default"
    model = PragmaForSequenceClassification(
        config,
        num_labels=len(TASK_LABELS[task]),
        representation=args.representation,  # type: ignore[arg-type]
    )
    if args.checkpoint:
        checkpoint_meta = checkpoint_metadata(args.checkpoint)
        model_meta = checkpoint_meta.get("meta", {})
        if args.lora or model_meta.get("lora"):
            apply_lora(
                model,
                rank=int(model_meta.get("lora_rank", 8)),
                alpha=float(model_meta.get("lora_alpha", 8.0)),
                dropout=float(model_meta.get("lora_dropout", 0.0)),
            )
        load_checkpoint(model, args.checkpoint, strict=False, map_location="cpu")
    model = model.to(ctx.device)
    model.eval()

    dataset = BankingEventDataset(
        args.data_dir,
        tokenizer,
        split="all",
        max_events=config.max_events,
    )
    idx = 0
    if args.user_id:
        matches = dataset.users.index[dataset.users["user_id"] == args.user_id].tolist()
        if not matches:
            raise ValueError(f"user_id not found: {args.user_id}")
        idx = int(matches[0])
    record = dataset[idx]
    if args.task:
        batch = collate_downstream([record], tokenizer=tokenizer, config=config, task=args.task)
    else:
        batch = collate_records([record], tokenizer=tokenizer, config=config)
    batch = move_batch(batch, ctx.device)
    with torch.no_grad():
        out = model(batch)
    payload = {
        "user_id": record.user_id,
        "embedding_dim": int(out["sequence_embedding"].shape[-1]),
        "embedding_preview": [round(float(x), 6) for x in out["sequence_embedding"][0, :10].cpu()],
    }
    if args.task:
        probs = torch.sigmoid(out["logits"]).detach().cpu().reshape(-1)
        payload["task"] = args.task
        payload["labels"] = TASK_LABELS[args.task]
        payload["probabilities"] = [round(float(x), 6) for x in probs]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
