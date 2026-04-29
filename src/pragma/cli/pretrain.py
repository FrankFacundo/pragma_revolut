from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import torch

from pragma.config import MaskingConfig, TrainingConfig, model_config_for_variant
from pragma.data.dataset import BankingEventDataset, build_dataloader, collate_pretrain
from pragma.data.tokenizer import PragmaTokenizer
from pragma.model.pragma import PragmaForMaskedModeling
from pragma.train.checkpoint import save_checkpoint
from pragma.train.device import autocast_context, move_batch, resolve_runtime, seed_everything
from pragma.train.losses import masked_mlm_loss
from pragma.train.optim import build_optimizer, learning_rate, set_lr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain PRAGMA with masked modelling.")
    parser.add_argument("--data-dir", default="data/synth")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--variant", choices=["tiny", "s", "m", "l"], default="s")
    parser.add_argument("--out-dir", default="runs/pretrain")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--max-events", type=int, default=512)
    parser.add_argument("--max-event-tokens", type=int, default=24)
    parser.add_argument("--max-profile-tokens", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None)
    parser.add_argument("--save-every", type=int, default=500)
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
        label_smoothing=args.label_smoothing,
        masking=MaskingConfig(),
    )

    dataset = BankingEventDataset(
        args.data_dir,
        tokenizer,
        split="train",
        seed=args.seed,
        max_events=config.max_events,
        max_event_tokens=config.max_field_tokens,
        max_profile_tokens=config.max_profile_tokens,
    )
    collate = partial(
        collate_pretrain,
        tokenizer=tokenizer,
        config=config,
        mask_cfg=train_cfg.masking,
    )
    loader = build_dataloader(
        dataset,
        collate,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    model = PragmaForMaskedModeling(config).to(ctx.device)
    if args.compile:
        model = torch.compile(model)
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.out_dir) / "checkpoints"
    print(
        f"pretrain variant={args.variant} params={sum(p.numel() for p in model.parameters()):,} "
        f"device={ctx.device} dtype={ctx.dtype} records={len(dataset)}"
    )

    iterator = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch(batch, ctx.device)
        start = time.time()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(ctx):
            out = model(batch)
            loss = masked_mlm_loss(
                out["mlm_logits"],
                batch["mlm_labels"],
                label_smoothing=args.label_smoothing,
            )
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
            masked = int(batch["mlm_labels"].ne(-100).sum().item())
            print(
                f"step={step:06d} loss={loss.item():.4f} lr={lr:.2e} "
                f"masked={masked} dt={time.time() - start:.2f}s"
            )
        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_checkpoint(
                out_dir,
                model,
                config=config,
                optimizer=optimizer,
                step=step,
                meta={"train": train_cfg.to_dict(), "variant": args.variant},
            )
    save_checkpoint(
        out_dir,
        model,
        config=config,
        optimizer=optimizer,
        step=args.steps,
        meta={"train": train_cfg.to_dict(), "variant": args.variant},
        name="model_last",
    )
    print(f"saved {out_dir / 'model_last.safetensors'}")


if __name__ == "__main__":
    main()
