# HellaSwag Chat-Only Harness

This repo contains a small manual harness for testing HellaSwag with AI products
that only expose a chat UI. It never calls a model API. Instead, it creates
copy/paste prompts, accepts the model's pasted JSON answers, and scores them
locally against a labeled HellaSwag JSONL file.

## Dataset Format

The harness expects the standard HellaSwag JSONL shape:

```json
{"ind": 0, "ctx": "A person opens a cabinet.", "endings": ["...", "...", "...", "..."], "label": 2}
```

It also accepts records that use `ctx_a` and `ctx_b` instead of `ctx`. Each row
must have exactly four `endings`. Scoring requires `label` values from `0` to
`3`, so use a labeled split such as validation.

## Manual Evaluation Flow

For a quick real run, fetch a small labeled validation sample and generate a
copy/paste prompt:

```bash
python3 manual_chat_harness.py fetch --limit 20 --output data/hellaswag_val_20.jsonl
python3 manual_chat_harness.py make-prompt data/hellaswag_val_20.jsonl \
  --output prompts/real-val-20.txt
```

Open `prompts/real-val-20.txt`, paste the whole thing into your chat-only AI
app, and ask it to respond as instructed. Save the model's response as
`answers/real-val-20.json`, then score it:

```bash
python3 manual_chat_harness.py score data/hellaswag_val_20.jsonl answers/real-val-20.json \
  --output scores/real-val-20.score.json
```

To fetch the full official validation split instead of a small sample:

```bash
python3 manual_chat_harness.py fetch --output data/hellaswag_val.jsonl
```

The full download is checksum-verified against the official HellaSwag validation
SHA-256 published in the `Rowan/hellaswag` Hugging Face dataset metadata.

Create a prompt batch:

```bash
python3 manual_chat_harness.py make-prompt path/to/hellaswag_val.jsonl \
  --limit 20 \
  --offset 0 \
  --output prompts/batch-000.txt
```

Paste `prompts/batch-000.txt` into the chat app. The prompt asks the model to
return only JSON like this:

```json
{"answers":[{"id":"0","choice":2},{"id":"1","choice":0}]}
```

Paste the model response into an answer file, then score it:

```bash
python3 manual_chat_harness.py score path/to/hellaswag_val.jsonl answers/batch-000.json \
  --limit 20 \
  --offset 0 \
  --output scores/batch-000.score.json
```

Use the same `--limit`, `--offset`, and `--seed` values for scoring that you
used to generate the prompt. Increase `--offset` by the batch size for the next
batch.

## Randomized Batches

For a deterministic shuffled sample:

```bash
python3 manual_chat_harness.py make-prompt path/to/hellaswag_val.jsonl \
  --seed 1234 \
  --limit 50 \
  --output prompts/shuffled-050.txt
```

Score with the same seed:

```bash
python3 manual_chat_harness.py score path/to/hellaswag_val.jsonl answers/shuffled-050.json \
  --seed 1234 \
  --limit 50
```

## Try It With the Sample

```bash
python3 manual_chat_harness.py make-prompt sample_hellaswag.jsonl --output /tmp/hellaswag_prompt.txt
python3 manual_chat_harness.py score sample_hellaswag.jsonl sample_answers.json
```

The sample answer file should report `1.0` accuracy.

## Included Real Prompt

This checkout includes `data/hellaswag_val_10.jsonl`, a ten-row slice from the
official validation set, and `prompts/real-val-10.txt`, a ready-to-paste prompt
generated from it. To test a chat app immediately:

```bash
mkdir -p answers scores
# Paste the chat app's JSON response into answers/real-val-10.json
python3 manual_chat_harness.py score data/hellaswag_val_10.jsonl answers/real-val-10.json \
  --output scores/real-val-10.score.json
```
