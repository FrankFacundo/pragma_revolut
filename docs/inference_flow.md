# PRAGMA inference - function call graph and tensor dimensions

This document traces the inference path invoked from:

```bash
pragma-infer --data-dir ... --tokenizer ... --checkpoint ... --variant s --user-id ...
```

or equivalently:

```bash
python scripts/infer.py --data-dir ... --tokenizer ... --checkpoint ... --variant s --user-id ...
```

The project is an encoder-only PRAGMA implementation for multi-source banking
event histories. Inference loads one encoded user record, runs the profile,
event, and history transformer encoders, then emits a sequence embedding and,
when `--task` is provided, sigmoid probabilities for the requested downstream
task.

Two call-graph views are provided:

1. **ASCII tree** - quickest to read top-to-bottom.
2. **Mermaid graph** - renders in GitHub / VS Code preview.

A shape walkthrough follows the graphs. It gives generic dimensions and the
concrete variant widths used by this repository.

---

## 1. ASCII tree

```text
main()                                                       src/pragma/cli/infer.py
|
+-- parse_args()
+-- resolve_runtime(device, dtype=None)                      train/device.py
|
+-- PragmaTokenizer.load(tokenizer.json)                     data/tokenizer.py
+-- model_config_for_variant(variant, vocab_size, tokens)    config.py
|
+-- PragmaForSequenceClassification(config, num_labels, repr) model/pragma.py
|     +-- PragmaBackbone(config)
|     |     +-- nn.Embedding(vocab_size, d_model)
|     |     +-- profile_encoder: TransformerEncoder          model/layers.py
|     |     |     +-- TransformerBlock x profile_layers
|     |     |           +-- LayerNorm
|     |     |           +-- MultiHeadSelfAttention
|     |     |           |     +-- qkv: Linear(D -> 3D)
|     |     |           |     +-- RotaryEmbedding            model/rotary.py
|     |     |           |     |     +-- apply_rotary(q, k)
|     |     |           |     +-- scaled_dot_product_attention
|     |     |           |     |     +-- softmax((q @ k.T) / sqrt(Dh)) @ v
|     |     |           |     +-- out_proj: Linear(D -> D)
|     |     |           +-- LayerNorm
|     |     |           +-- FeedForward
|     |     |                 +-- fc1: Linear(D -> F)
|     |     |                 +-- GELU(approximate="tanh")
|     |     |                 +-- fc2: Linear(F -> D)
|     |     +-- event_encoder: TransformerEncoder
|     |     |     +-- TransformerBlock x event_layers
|     |     |           +-- same block, without RoPE
|     |     +-- calendar_encoder: CalendarEncoder
|     |     |     +-- time features: hour/weekday/monthday -> sin/cos
|     |     |     +-- Linear(6 -> calendar_hidden or D)
|     |     |     +-- GELU
|     |     |     +-- Linear(hidden -> D)
|     |     +-- history_encoder: TransformerEncoder
|     |           +-- TransformerBlock x history_layers
|     |                 +-- same block, with temporal RoPE
|     |
|     +-- classifier
|           +-- sequence_representation(history, event_mask)
|           +-- LayerNorm(R)
|           +-- Dropout
|           +-- Linear(R -> C)
|
+-- checkpoint_metadata(checkpoint)                          train/checkpoint.py
+-- optionally apply_lora(model)                             model/lora.py
|     +-- replace qkv/out_proj/fc1/fc2 with LoRALinear
+-- load_checkpoint(model, checkpoint)
|
+-- BankingEventDataset(data_dir, tokenizer, split="all")    data/dataset.py
|     +-- read users.parquet, events.parquet, profiles.parquet
|
+-- dataset[idx]
|     +-- PragmaTokenizer.encode_record
|           +-- encode_profile
|           |     +-- encode_value per profile field
|           |           +-- numeric bucket / categorical token / SimpleBPE text
|           +-- encode_events
|                 +-- encode_value per event field
|                 +-- soft_log_seconds(time_to_last)
|
+-- collate_records or collate_downstream                    data/dataset.py
|     +-- pad profile tensors
|     +-- pad event tensors
|     +-- attach labels for downstream task
|
+-- move_batch(batch, device)                                train/device.py
+-- torch.no_grad()
      +-- model(batch)
      |     +-- PragmaForSequenceClassification.forward
      |           +-- PragmaBackbone.forward
      |           |     +-- _encode_profile
      |           |     |     +-- _pair_embeddings
      |           |     |     +-- prepend [USR]
      |           |     |     +-- profile_encoder(..., rope_positions=profile_times)
      |           |     +-- _encode_events
      |           |     |     +-- flatten B x E events to (B*E)
      |           |     |     +-- _pair_embeddings
      |           |     |     +-- prepend [EVT]
      |           |     |     +-- event_encoder(...)
      |           |     |     +-- split token and [EVT] embeddings
      |           |     +-- calendar_encoder(calendar) + event_repr
      |           |     +-- concat [USR] profile repr + event reprs
      |           |     +-- history_encoder(..., rope_positions=event_times)
      |           +-- sequence_representation(usr/last_event/both)
      |           +-- classifier(rep)
      |
      +-- JSON payload
            +-- embedding_dim
            +-- embedding_preview
            +-- optional task labels and probabilities
```

---

## 2. Mermaid call graph

```mermaid
flowchart TD
    main[main<br/>src/pragma/cli/infer.py]

    main --> parse_args
    main --> runtime[resolve_runtime]
    main --> tok_load[PragmaTokenizer.load]
    main --> cfg[model_config_for_variant]
    main --> build_model[PragmaForSequenceClassification]
    main --> ckpt_meta[checkpoint_metadata]
    main --> maybe_lora{checkpoint uses LoRA?}
    maybe_lora -->|yes| lora[apply_lora<br/>qkv/out_proj/fc1/fc2]
    maybe_lora -->|no| load_ckpt[load_checkpoint]
    lora --> load_ckpt
    main --> dataset[BankingEventDataset]
    main --> record[dataset[idx]]
    main --> collate[collate_records<br/>or collate_downstream]
    main --> move[move_batch]
    main --> no_grad[torch.no_grad]
    no_grad --> forward[model(batch)]
    forward --> json[JSON output]

    dataset --> parquet[read users/events/profiles parquet]
    record --> enc_record[PragmaTokenizer.encode_record]
    enc_record --> enc_profile[encode_profile]
    enc_record --> enc_events[encode_events]
    enc_profile --> enc_value_p[encode_value]
    enc_events --> enc_value_e[encode_value]
    enc_value_p --> val_routes[numeric bucket<br/>categorical token<br/>SimpleBPE text]
    enc_value_e --> val_routes
    enc_events --> tlog[soft_log_seconds]
    collate --> pad_profile[pad profile tensors]
    collate --> pad_events[pad event tensors]
    collate --> labels[optional labels]

    build_model --> cls[Classifier head]
    build_model --> bb[PragmaBackbone]
    bb --> embed[nn.Embedding]
    bb --> prof[profile_encoder<br/>TransformerEncoder]
    bb --> event[event_encoder<br/>TransformerEncoder]
    bb --> cal[CalendarEncoder]
    bb --> hist[history_encoder<br/>TransformerEncoder]

    forward --> bb_fwd[PragmaBackbone.forward]
    bb_fwd --> pair_p[_pair_embeddings profile]
    pair_p --> usr[prepend USR]
    usr --> prof_fwd[profile_encoder forward]
    bb_fwd --> pair_e[_pair_embeddings events]
    pair_e --> evt[prepend EVT]
    evt --> event_fwd[event_encoder forward]
    event_fwd --> split[split event tokens<br/>and EVT repr]
    split --> cal_add[add calendar embedding]
    prof_fwd --> hist_in[concat USR repr + events]
    cal_add --> hist_in
    hist_in --> hist_fwd[history_encoder forward]
    hist_fwd --> seq_rep[sequence_representation]
    seq_rep --> cls_fwd[classifier]

    prof --> block_p[TransformerBlock x profile_layers]
    event --> block_e[TransformerBlock x event_layers]
    hist --> block_h[TransformerBlock x history_layers]
    block_p --> attn[MultiHeadSelfAttention]
    block_e --> attn
    block_h --> attn
    block_p --> ffn[FeedForward]
    block_e --> ffn
    block_h --> ffn

    attn --> qkv[Linear D to 3D]
    attn --> rope{use_rope?}
    rope -->|profile/history| rotary[RotaryEmbedding<br/>apply_rotary]
    rope -->|event| sdpa[scaled_dot_product_attention]
    rotary --> sdpa
    sdpa --> scores[QK^T / sqrt Dh]
    scores --> softmax[softmax]
    softmax --> values[attention weights @ V]
    values --> out_proj[Linear D to D]

    ffn --> fc1[Linear D to F]
    fc1 --> gelu[GELU tanh approximation]
    gelu --> fc2[Linear F to D]
```

---

## 3. Shape legend

| Symbol | Meaning |
| --- | --- |
| `B` | Batch size. `pragma-infer` builds a one-record batch, so `B=1`. |
| `P` | Number of padded profile key/value tokens, `1 <= P <= max_profile_tokens`. Default cap: `200`. |
| `E` | Number of padded events, `1 <= E <= max_events`. CLI default cap: `512`. |
| `T` | Number of padded key/value tokens inside each event, `1 <= T <= max_field_tokens`. Default cap: `24`. |
| `V` | Tokenizer vocabulary size. |
| `D` | Model width, `config.d_model`. |
| `F` | Feed-forward width, `config.d_ffn`. |
| `H` | Number of attention heads. |
| `Dh` | Per-head width, `D / H`. Must be even for RoPE-enabled encoders. |
| `Ch` | Calendar encoder hidden size, `config.calendar_hidden or D`. |
| `C` | Number of task labels for the classifier. |
| `R` | Sequence representation width: `D` for `usr` or `last_event`, `2D` for `both`. |

Variant dimensions:

| Variant | `D` | `F` | Profile layers | Event layers | History layers | `H` | `Dh` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny` | 64 | 192 | 1 | 1 | 1 | 4 | 16 |
| `s` | 192 | 768 | 1 | 5 | 2 | 3 | 64 |
| `m` | 512 | 2048 | 3 | 16 | 6 | 8 | 64 |
| `l` | 1024 | 4096 | 9 | 45 | 18 | 16 | 64 |

Default CLI example for `--variant s --representation both --max-events 512`:

| Stage | Symbolic shape | Max-size shape |
| --- | --- | --- |
| Profile encoder input | `[1, P+1, 192]` | `[1, 201, 192]` |
| Event encoder input after flattening | `[E, T+1, 192]` | `[512, 25, 192]` |
| Event representations | `[1, E, 192]` | `[1, 512, 192]` |
| History encoder input/output | `[1, E+1, 192]` | `[1, 513, 192]` |
| Sequence representation, `both` | `[1, 384]` | `[1, 384]` |
| Classifier logits | `[1, C]`, then `[1]` when `C=1` | task-dependent |

---

## 4. Tensor dimensions from input to output

### 4.1 Collated batch

`collate_records` and `collate_downstream` produce:

| Tensor | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `profile_key_ids` | `[B, P]` | `long` | Semantic key token per profile value token. |
| `profile_value_ids` | `[B, P]` | `long` | Encoded profile values. |
| `profile_positions` | `[B, P]` | `long` | Position inside multi-token values. |
| `profile_times` | `[B, P]` | `float32` | Soft-log time since lifelong profile milestone, otherwise `0`. |
| `profile_mask` | `[B, P]` | `bool` | Valid profile token mask. |
| `event_key_ids` | `[B, E, T]` | `long` | Semantic key token per event value token. |
| `event_value_ids` | `[B, E, T]` | `long` | Encoded event values. |
| `event_positions` | `[B, E, T]` | `long` | Position inside multi-token event values. |
| `event_token_mask` | `[B, E, T]` | `bool` | Valid token mask inside each event. |
| `event_times` | `[B, E]` | `float32` | Soft-log seconds from each event to the last event. |
| `calendar` | `[B, E, 3]` | `float32` | Hour, weekday, monthday. |
| `event_mask` | `[B, E]` | `bool` | Valid event mask. |
| `labels` | `[B]` or `[B, C]` | `float32` | Only added by `collate_downstream`. |

### 4.2 Shared pair embedding

`PragmaBackbone._pair_embeddings(key_ids, value_ids, value_positions)` is used by
both the profile and event branches:

| Operation | Input shape | Output shape |
| --- | --- | --- |
| `embedding(key_ids)` | `[..., S]` | `[..., S, D]` |
| `embedding(value_ids)` | `[..., S]` | `[..., S, D]` |
| Add key and value embeddings | `[..., S, D]` | `[..., S, D]` |
| `sinusoidal_positions(value_positions, D)` | `[..., S]` | `[..., S, D]` |
| Add positional encoding + dropout | `[..., S, D]` | `[..., S, D]` |

For the profile branch, `S=P`.
For flattened event encoding, `S=T` and the leading dimension is `B*E`.

### 4.3 Profile branch

| Operation | Input shape | Output shape |
| --- | --- | --- |
| Profile pair embeddings | `[B, P]` ids | `[B, P, D]` |
| Embed `[USR]` token | `[B, 1]` ids | `[B, 1, D]` |
| Concatenate `[USR]` + profile tokens | `[B, 1, D]`, `[B, P, D]` | `[B, P+1, D]` |
| Build profile attention mask | `[B, P]` | `[B, P+1]` |
| Build profile RoPE positions | `[B, P]` | `[B, P+1]` |
| `profile_encoder` | `[B, P+1, D]` | `[B, P+1, D]` |
| Take `[USR]` representation | `[B, P+1, D]` | `[B, 1, D]` |

### 4.4 Event branch

| Operation | Input shape | Output shape |
| --- | --- | --- |
| Flatten events | `[B, E, T]` ids | `[B*E, T]` ids |
| Event pair embeddings | `[B*E, T]` ids | `[B*E, T, D]` |
| Embed `[EVT]` token | `[B*E, 1]` ids | `[B*E, 1, D]` |
| Concatenate `[EVT]` + event tokens | `[B*E, 1, D]`, `[B*E, T, D]` | `[B*E, T+1, D]` |
| Build event attention mask | `[B*E, T]` | `[B*E, T+1]` |
| `event_encoder` | `[B*E, T+1, D]` | `[B*E, T+1, D]` |
| Token embeddings | `[B*E, T, D]` | `[B, E, T, D]` |
| `[EVT]` event representation | `[B*E, D]` | `[B, E, D]` |
| Apply `event_mask` and `event_token_mask` | `[B, E, D]`, `[B, E, T, D]` | same shapes |

### 4.5 Calendar branch

| Operation | Input shape | Output shape |
| --- | --- | --- |
| Normalize hour/weekday/monthday | `[B, E, 3]` | `[B, E, 3]` |
| Sin/cos features | `[B, E, 3]` | `[B, E, 6]` |
| `Linear(6 -> Ch)` + GELU | `[B, E, 6]` | `[B, E, Ch]` |
| `Linear(Ch -> D)` | `[B, E, Ch]` | `[B, E, D]` |
| Mask and add to event repr | `[B, E, D]` | `[B, E, D]` |

### 4.6 History branch

| Operation | Input shape | Output shape |
| --- | --- | --- |
| Concatenate profile `[USR]` + event reprs | `[B, 1, D]`, `[B, E, D]` | `[B, E+1, D]` |
| Build history mask | `[B, E]` | `[B, E+1]` |
| Build history RoPE positions | `[B, E]` | `[B, E+1]` |
| `history_encoder` | `[B, E+1, D]` | `[B, E+1, D]` |
| `user_embedding = history[:, 0]` | `[B, E+1, D]` | `[B, D]` |
| `history_event_embeddings = history[:, 1:]` | `[B, E+1, D]` | `[B, E, D]` |

The backbone returns:

| Output key | Shape |
| --- | --- |
| `history` | `[B, E+1, D]` |
| `history_mask` | `[B, E+1]` |
| `user_embedding` | `[B, D]` |
| `event_token_embeddings` | `[B, E, T, D]` |
| `history_event_embeddings` | `[B, E, D]` |
| `event_embeddings` | `[B, E, D]` |

### 4.7 TransformerEncoder internals

Each profile, event, and history encoder is a stack of identical
`TransformerBlock`s followed by a final `LayerNorm`. Let `S` be the sequence
length for the active encoder:

| Encoder | `S` | RoPE |
| --- | ---: | --- |
| Profile encoder | `P+1` | Yes, using `profile_times`. |
| Event encoder | `T+1` | No. |
| History encoder | `E+1` | Yes, using `event_times`. |

One `TransformerBlock`:

| Operation | Input shape | Output shape |
| --- | --- | --- |
| `LayerNorm(D)` | `[B0, S, D]` | `[B0, S, D]` |
| `qkv: Linear(D -> 3D)` | `[B0, S, D]` | `[B0, S, 3D]` |
| View into heads | `[B0, S, 3D]` | `[B0, S, 3, H, Dh]` |
| Split and transpose `q,k,v` | `[B0, S, 3, H, Dh]` | each `[B0, H, S, Dh]` |
| `RotaryEmbedding(rope_positions)` | `[B0, S]` | `cos,sin` each `[B0, S, Dh]` |
| `apply_rotary(q,k)` | `q,k [B0, H, S, Dh]` | `q,k [B0, H, S, Dh]` |
| Attention mask broadcast | `[B0, S]` | `[B0, 1, 1, S]` |
| Conceptual attention scores | `q [B0, H, S, Dh]`, `k.T [B0, H, Dh, S]` | `[B0, H, S, S]` |
| Mask + softmax | scores `[B0, H, S, S]`, mask `[B0, 1, 1, S]` | `[B0, H, S, S]` |
| Weighted value sum | weights `[B0, H, S, S]`, `v [B0, H, S, Dh]` | `[B0, H, S, Dh]` |
| Fused `scaled_dot_product_attention(q,k,v)` | each `[B0, H, S, Dh]` | `[B0, H, S, Dh]` |
| Merge heads | `[B0, H, S, Dh]` | `[B0, S, D]` |
| `out_proj: Linear(D -> D)` | `[B0, S, D]` | `[B0, S, D]` |
| Attention residual | `[B0, S, D]` | `[B0, S, D]` |
| `LayerNorm(D)` | `[B0, S, D]` | `[B0, S, D]` |
| `fc1: Linear(D -> F)` | `[B0, S, D]` | `[B0, S, F]` |
| `GELU` + dropout | `[B0, S, F]` | `[B0, S, F]` |
| `fc2: Linear(F -> D)` | `[B0, S, F]` | `[B0, S, D]` |
| FFN residual + mask | `[B0, S, D]` | `[B0, S, D]` |

`B0` is `B` for profile/history and `B*E` for event encoding. The RoPE rows
apply only to the profile and history encoders; the event encoder skips them.

The final encoder norm maps `[B0, S, D] -> [B0, S, D]` and zeros padded
positions with the attention mask.

### 4.8 Sequence classifier head

| Representation | Operation | Shape |
| --- | --- | --- |
| `usr` | `history[:, 0]` | `[B, D]` |
| `last_event` | gather final valid event from `history[:, 1:]` | `[B, D]` |
| `both` | concatenate `usr` and `last_event` | `[B, 2D]` |

Then:

| Operation | Input shape | Output shape |
| --- | --- | --- |
| `LayerNorm(R)` | `[B, R]` | `[B, R]` |
| Dropout | `[B, R]` | `[B, R]` |
| `Linear(R -> C)` | `[B, R]` | `[B, C]` |
| Squeeze singleton label dimension | `[B, 1]` | `[B]` |
| `sigmoid(logits)` in CLI when `--task` is set | `[B]` or `[B, C]` | probabilities |

For `--representation both` on variant `s`, `R = 2D = 384`.
For `product_rec`, `C=3`; other built-in tasks use `C=1`.

### 4.9 Masked-modeling head, for pretraining context

`pragma-infer` does not use this head, but it shares the same backbone and is
the output path for `PragmaForMaskedModeling`.

| Operation | Input shape | Output shape |
| --- | --- | --- |
| Local event token embeddings | from backbone | `[B, E, T, D]` |
| History event embeddings | `[B, E, D]` | `[B, E, T, D]` after expand |
| User embedding | `[B, D]` | `[B, E, T, D]` after expand |
| Concatenate token/history/user | three `[B, E, T, D]` tensors | `[B, E, T, 3D]` |
| `Linear(3D -> D)` + GELU + `LayerNorm(D)` | `[B, E, T, 3D]` | `[B, E, T, D]` |
| Tied embedding projection | `[B, E, T, D] @ [D, V]` | `[B, E, T, V]` |
| Add `mlm_bias` | `[B, E, T, V]` | `[B, E, T, V]` |

### 4.10 Optional LoRA adapter dimensions

When a checkpoint was trained with LoRA, `apply_lora` wraps the selected
`qkv`, `out_proj`, `fc1`, and `fc2` linear modules. It does not change any
external tensor shape.

For a wrapped `Linear(I -> O)`:

| Operation | Shape |
| --- | --- |
| Base projection | `[..., I] -> [..., O]` |
| `lora_a` parameter | `[rank, I]` |
| `lora_b` parameter | `[O, rank]` |
| Adapter hidden | `[..., I] @ [I, rank] -> [..., rank]` |
| Adapter output | `[..., rank] @ [rank, O] -> [..., O]` |
| Final wrapped output | base `[..., O]` + scaled adapter `[..., O]` |

---

## 5. Hot path, compressed

For downstream inference on one user:

```text
PragmaTokenizer.encode_record
  -> collate_records/collate_downstream
  -> PragmaForSequenceClassification.forward
       -> PragmaBackbone.forward
            -> _encode_profile
                 embedding(key) + embedding(value) + sinusoidal value positions
                 prepend [USR]
                 TransformerEncoder with temporal RoPE
            -> _encode_events
                 flatten events to B*E
                 embedding(key) + embedding(value) + sinusoidal value positions
                 prepend [EVT]
                 TransformerEncoder without RoPE
                 take [EVT] as event embedding
            -> CalendarEncoder(calendar) + event embedding
            -> concat [USR] + event embeddings
            -> history TransformerEncoder with temporal RoPE
       -> sequence_representation(history, event_mask)
       -> LayerNorm -> Dropout -> Linear -> logits
  -> optional sigmoid(logits)
  -> JSON embedding/probabilities
```

The main attention cost is in the history encoder because its sequence length is
`E+1`, while event encoding is applied to `B*E` short sequences of length `T+1`.
With the CLI defaults, `E<=512` and `T<=24`; with the paper-style configuration
limits, `E` can be raised up to `6500`.
