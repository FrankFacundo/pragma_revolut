from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import torch

from pragma.config import TrainingConfig, model_config_for_variant
from pragma.data.dataset import (
    TASK_LABELS,
    BankingEventDataset,
    build_dataloader,
    collate_downstream,
)
from pragma.data.tokenizer import PragmaTokenizer
from pragma.model.lora import apply_lora, lora_trainable_parameters
from pragma.model.pragma import PragmaForSequenceClassification
from pragma.train.checkpoint import load_checkpoint, save_checkpoint
from pragma.train.device import autocast_context, move_batch, resolve_runtime, seed_everything
from pragma.train.losses import binary_or_multilabel_loss
from pragma.train.optim import build_optimizer, learning_rate, set_lr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune PRAGMA on synthetic downstream tasks.")
    parser.add_argument("--data-dir", default="data/synth")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--task", choices=sorted(TASK_LABELS), default="credit_default")
    parser.add_argument("--variant", choices=["tiny", "s", "m", "l"], default="s")
    parser.add_argument("--out-dir", default="runs/finetune")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-events", type=int, default=512)
    parser.add_argument("--max-event-tokens", type=int, default=24)
    parser.add_argument("--max-profile-tokens", type=int, default=200)
    parser.add_argument("--representation", choices=["usr", "last_event", "both"], default="both")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    ctx = resolve_runtime(args.device, args.dtype)
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
    config.max_field_tokens = args.max_event_tokens
    config.max_profile_tokens = args.max_profile_tokens
    num_labels = len(TASK_LABELS[args.task])
    model = PragmaForSequenceClassification(
        config,
        num_labels=num_labels,
        representation=args.representation,  # type: ignore[arg-type]
    )
    if args.checkpoint:
        info = load_checkpoint(model, args.checkpoint, strict=False, map_location="cpu")
        if info["unexpected"]:
            print(f"checkpoint unexpected keys ignored: {info['unexpected'][:6]}")
    if args.lora:
        apply_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        trainable, total, pct = lora_trainable_parameters(model)
        print(f"LoRA trainable={trainable:,}/{total:,} ({pct:.2f}%)")
    model = model.to(ctx.device)
    if args.compile:
        model = torch.compile(model)

    train_ds = BankingEventDataset(
        args.data_dir,
        tokenizer,
        split="train",
        seed=args.seed,
        max_events=config.max_events,
        max_event_tokens=config.max_field_tokens,
        max_profile_tokens=config.max_profile_tokens,
    )
    val_ds = BankingEventDataset(
        args.data_dir,
        tokenizer,
        split="val",
        seed=args.seed,
        max_events=config.max_events,
        max_event_tokens=config.max_field_tokens,
        max_profile_tokens=config.max_profile_tokens,
    )
    collate = partial(collate_downstream, tokenizer=tokenizer, config=config, task=args.task)
    train_loader = build_dataloader(
        train_ds,
        collate,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_dataloader(
        val_ds,
        collate,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        num_workers=args.num_workers,
        log_every=args.log_every,
        save_every=args.save_every,
    )
    out_dir = Path(args.out_dir) / args.task / "checkpoints"
    param_count = sum(p.numel() for p in model.parameters())
    print(
        f"finetune task={args.task} variant={args.variant} params={param_count:,} "
        f"device={ctx.device} dtype={ctx.dtype} train_records={len(train_ds)}"
    )
    iterator = iter(train_loader)
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = move_batch(batch, ctx.device)
        start = time.time()
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with autocast_context(ctx):
            out = model(batch)
            loss = binary_or_multilabel_loss(out["logits"], batch["labels"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr = learning_rate(
            step,
            base_lr=args.lr,
            total_steps=args.steps,
            warmup_steps=args.warmup_steps,
        )
        set_lr(optimizer, lr)
        optimizer.step()
        if step % args.log_every == 0:
            dt = time.time() - start
            print(f"step={step:06d} loss={loss.item():.4f} lr={lr:.2e} dt={dt:.2f}s")
        if args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
            metrics = evaluate(model, val_loader, ctx)
            print("val " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_checkpoint(
                out_dir,
                model,
                config=config,
                optimizer=optimizer,
                step=step,
                meta={
                    "train": train_cfg.to_dict(),
                    "task": args.task,
                    "lora": args.lora,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                },
            )
    save_checkpoint(
        out_dir,
        model,
        config=config,
        optimizer=optimizer,
        step=args.steps,
        meta={
            "train": train_cfg.to_dict(),
            "task": args.task,
            "lora": args.lora,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
        name="model_last",
    )
    print(f"saved {out_dir / 'model_last.safetensors'}")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    ctx,
    *,
    max_batches: int = 32,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        batch = move_batch(batch, ctx.device)
        with autocast_context(ctx):
            out = model(batch)
            loss = binary_or_multilabel_loss(out["logits"], batch["labels"])
        losses.append(float(loss.detach().cpu()))
        logits_all.append(out["logits"].detach().float().cpu())
        labels_all.append(batch["labels"].detach().float().cpu())
    metrics: dict[str, float] = {"loss": sum(losses) / max(1, len(losses))}
    metrics.update(_sklearn_metrics(torch.cat(logits_all), torch.cat(labels_all)))
    return metrics


def _sklearn_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:
        return {}
    probs = torch.sigmoid(logits).numpy()
    y = labels.numpy()
    out: dict[str, float] = {}
    try:
        out["pr_auc"] = float(average_precision_score(y, probs, average="macro"))
    except ValueError:
        pass
    try:
        out["roc_auc"] = float(roc_auc_score(y, probs, average="macro"))
    except ValueError:
        pass
    return out


if __name__ == "__main__":
    main()
