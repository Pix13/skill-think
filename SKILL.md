---
name: controlling-local-llm-reasoning
description: Use when an agent needs to turn reasoning/thinking mode on, off, or set its depth on a local Qwen3-style model served by llama-server (llama.cpp) or vLLM — e.g. running the model with thinking OFF by default and enabling it per-request for hard subtasks, or applying reasoning-style/sampling presets. Keywords enable_thinking, reasoning_budget, thinking_budget, chat_template_kwargs, no_think.
---

# Controlling Local LLM Reasoning

## Overview

Qwen3-style models support a **hybrid thinking mode** that can be toggled per
request. The lever is the chat-template flag `enable_thinking`, plus a token
budget. This lets an agent run a cheap, fast model with **reasoning OFF by
default** and switch it **ON only for the hard subtasks** that need it.

Core principle: **thinking is a per-request property, not a server property.**
Start the server with thinking off; flip it on in the request body when needed.

This skill ships `think-chat.py`, a stdlib-only CLI that builds the correct
request body for either backend and POSTs it. Use it directly, or hand-roll the
request from the Quick Reference below.

## When to Use

- Delegating subtasks to a local model and you want to control its reasoning per task.
- You want a model serving fast/cheap by default, with deep reasoning on demand.
- You need Qwen3 reasoning **off** (it's on by default) for latency or tool-calling.
- You want to nudge reasoning *style* (first-principles, pre-mortem, …) or sampling.

Not for: hosted APIs (Claude/OpenAI manage thinking differently), or models without
a hybrid-thinking chat template.

Big thanks to https://github.com/iChristGit/OpenWebui-Tools

## The Toggle, Per Backend

Both backends read `chat_template_kwargs.enable_thinking`. They differ on the budget field.

| | **llama-server (llama.cpp)** | **vLLM** |
|---|---|---|
| Launch flag (required) | `--jinja` | `--reasoning-parser qwen3` (vLLM ≥ 0.9.0) |
| Default thinking OFF | `--reasoning-budget 0` | `--default-chat-template-kwargs '{"enable_thinking": false}'` |
| Per-request toggle | `chat_template_kwargs: {enable_thinking: true/false}` | same |
| Token budget field | `reasoning_budget` (−1 unlimited, 0 off, N cap) | `thinking_budget` (N cap; version-dependent) |
| Hard-disable | `reasoning_budget: 0` | `enable_thinking: false` |

Request-level `chat_template_kwargs` always overrides the server default, so a
server started thinking-off can still be told to think for one request.

**vLLM budget caveat:** `enable_thinking` works from vLLM 0.9.0; a per-request
*token cap* (`thinking_budget`) needs a recent build with reasoning-budget
support. If your vLLM rejects it, drop the cap — `enable_thinking: true` alone
still gives unlimited thinking. `--depth unlimited` (the default) never sends a cap.

## Quick Reference (think-chat.py)

```bash
# Fast answer, thinking OFF — the default
think-chat.py "what's 2+2?"

# Enable thinking — defaults to the general-coding profile
# (depth 'deep' / 8192 tokens + 'Thinking Precise' sampling, temp 0.6)
think-chat.py --thinking on "implement an LRU cache"

# Override the budget for a harder subtask
think-chat.py --thinking on --depth max "prove sqrt(2) is irrational"

# vLLM backend, also print the reasoning trace
think-chat.py --backend vllm --thinking on --show-thinking "plan this refactor"

# Reasoning style + sampling preset
think-chat.py --thinking on --think-style "First Principles" --sampling "Thinking Precise" "design a rate limiter"

# See the exact request body without sending (no server needed)
think-chat.py --thinking on --depth max --dry-run "hi"

# List every preset name
think-chat.py --list
```

Key flags: `--backend {llama,vllm}` · `--url` (default llama `:8080`, vllm `:8000`)
· `--model` · `--thinking {on,off}` · `--depth {unlimited,max,deep,normal,quick}`
· `--think-style` · `--present` · `--sampling` · `--show-thinking` · `--dry-run`.
Prompt is positional or piped via stdin. Run `--help` for all flags.

**Defaults:** thinking is OFF. Turning it on (`--thinking on`) defaults to the
general-coding profile — depth `deep` (8192) + `Thinking Precise` sampling
(temp 0.6, presence 0). Pass `--depth`/`--sampling` explicitly to override.

**Smoke-test the toggle on a real server** with `test_thinking.py` — it sends the
same prompt thinking-on then -off and asserts a reasoning trace appears only when
on (exit 0 pass / 1 fail / 2 unreachable):
```bash
URL=http://host:8080 MODEL=mymodel ./test_thinking.py
```

**Depth → budget tokens:** unlimited=−1 · max=16000 · deep=8192 · normal=3072 · quick=512.

## Hand-Rolled Requests

llama-server, thinking on with an 8k budget:
```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "local",
  "messages": [{"role": "user", "content": "..."}],
  "chat_template_kwargs": {"enable_thinking": true},
  "reasoning_budget": 8192
}'
```

vLLM, thinking off for this one request (server may default it on):
```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3-8B",
  "messages": [{"role": "user", "content": "..."}],
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Python OpenAI SDK (vLLM) — thinking kwargs go under `extra_body`:
```python
client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "..."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}, "thinking_budget": 8192},
)
```

## Presets

The CLI can inject prompt nudges into the system message and apply sampling bundles
(ported from the original Open-WebUI filter). Run `think-chat.py --list` for names.

- **Think styles** (only injected when thinking is on): First Principles, Pre-Mortem,
  Bayesian, Contrarian, Socratic, 10x Hypotheses, Step by Step, Devil's Advocate,
  Rubber Duck, Think Less/More, Extended/MAX Thinking.
- **Presentation:** ELI5, Be Concise, Bullet Points, TL;DR First, Teach Me, Expert
  Tone, Casual Chat, Debate Format, Analogies Only, Action Items, Socratic Reply.
- **Sampling** (Qwen3-recommended bundles): `Thinking General` (temp 1.0), `Thinking
  Precise` (temp 0.6, coding), `Instruct General` (temp 0.7), `Instruct Reasoning`.
  `Default` touches nothing. Sampling params are written at the body root **and**
  mirrored into `options{}` for Ollama-routed connections.

## Common Mistakes

- **Forgetting `--jinja` on llama-server** → `chat_template_kwargs` is ignored and
  the toggle silently does nothing.
- **Expecting `reasoning_budget` to work on vLLM** — it's a llama.cpp field. vLLM
  uses `thinking_budget`, and only on recent builds. Use `enable_thinking` there.
- **Setting Qwen3 sampling for the wrong mode** — Qwen recommends greedy-ish temp
  0.6 for thinking-precise/coding, but temp ~1.0 for general thinking. Don't reuse
  one bundle for both.
- **Assuming the server default sticks** — any request that sends
  `chat_template_kwargs` overrides it. Send the flag explicitly every time you care.
- **Reading the answer with thinking on but no parser** — without
  `--reasoning-parser` (vLLM), the `<think>` block lands in `content`, not
  `reasoning_content`; `--show-thinking` will then show nothing separate.
```
