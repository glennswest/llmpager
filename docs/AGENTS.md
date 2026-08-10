# Using llmpager as an agent backend

Findings from the Unsloth agent-harness deep dive (2026-08-09), and how
they map onto llmpager.

## What Unsloth actually ships (two separate things)

**1. Unsloth Start** — a launcher that points coding-agent CLIs at a
locally served model instead of a cloud API:
`unsloth start claude | codex | opencode | hermes | openclaw | pi`.
It spins up (or connects to) a local OpenAI-compatible server, injects
session-scoped provider config into the agent, and tears down on exit.
Nothing architecturally novel — the value is the packaging.

**2. ART (openpipe/art) + Unsloth RL** — an agent *training* harness:
agents run rollouts against an OpenAI-compatible backend, executions are
captured as trajectories (tool calls included), and RULER provides
automatic LLM-elicited rewards so you can RL-tune an agent without
hand-written reward functions. Open source; backend is any
Unsloth-supported model.

## What this means for llmpager

`llmpager-serve` is already the OpenAI-compatible endpoint both of these
patterns require. No integration code is needed for the serving side:

- **OpenAI-protocol agents** (Codex-style, OpenCode, anything using
  `OPENAI_BASE_URL`): point at the box and pick a model:

  ```bash
  export OPENAI_BASE_URL=http://ai.g8.lo:8090/v1
  export OPENAI_API_KEY=unused
  # interactive coding: qwen3-coder-30b-a3b (41 tok/s)
  # deep overnight tasks: kimi-k2.6 (~0.7 tok/s, 1T params)
  ```

- **Anthropic-protocol agents** (Claude Code) need a protocol
  translation layer; Unsloth Start provides one, as do LiteLLM-style
  proxies. We do not implement the Anthropic API ourselves.

- **RL training with ART**: llmpager can be the rollout backend
  (`LocalBackend` pointing at :8090). Batched decode (v0.15) matters
  here — rollout generation is throughput-bound, and prompt-array
  requests share expert fetches across parallel rollouts. Training
  itself (the gradient side) stays in Unsloth/ART on whatever GPU does
  the tuning; llmpager only serves inference.

## The model-tier pairing that makes this interesting

| Agent workload | Model | Why |
|---|---|---|
| Interactive coding loops | qwen3-coder-30b-a3b | 41 tok/s, low latency |
| Parallel rollout generation | qwen3 + batch API | fetch sharing, ~57 tok/s aggregate |
| Overnight deep analysis / planning | kimi-k2.6 | 1T-parameter quality at batch speeds |

One box, one endpoint, three tiers — the agent picks by `model:` field.
