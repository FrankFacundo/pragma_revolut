# Architecture

This implementation follows the PRAGMA paper at the level needed to train and
serve a reusable encoder backbone.

## Data Representation

Each user record has:

- `profiles.parquet`: static profile state and life-long milestones
- `events.parquet`: ordered heterogeneous events
- `users.parquet`: evaluation point and downstream labels

Every non-empty field is decomposed into:

- a semantic key token, for example `key:amount`
- one or more value tokens
- a time coordinate

Numerical values are bucketised by training-set percentiles with a dedicated
zero bucket. Categorical values map to a single token. Text values use a small
local BPE-style tokenizer, not a Hugging Face tokenizer.

## Model

`PragmaBackbone` has three encoders:

- Profile State Encoder: encodes profile key-value tokens and life-long-event
  time deltas with RoPE.
- Event Encoder: independently encodes tokens inside each event and emits an
  `[EVT]` representation per event.
- History Encoder: fuses the `[USR]` profile representation and all event
  representations, using log-seconds-to-last-event as continuous RoPE positions.

The pretraining head reconstructs masked event value tokens from the
concatenation of:

- local event-token embedding
- contextual history embedding for the event
- contextual user embedding

The downstream classifier uses `[USR]`, final `[EVT]`, or both.

## Training

Pretraining uses three corruption paths:

- individual token masking
- whole-event masking
- key-level masking within an event

Selected tokens are usually replaced by `[MASK]`; a fraction are replaced by
`[UNK]` and excluded from the loss, matching the paper's input-dropout behavior.

Fine-tuning supports full updates or LoRA on attention and feed-forward
projection layers.
