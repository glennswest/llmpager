# YouTube upload kit — Kimi milestone video

Video file: docs/presentation/kimi-build/kimi-video.mp4 (~9:57)
Thumbnails: kimi-thumb-1.jpg (dark, "1 TRILLION PARAMETERS"),
            kimi-thumb-2.jpg (light, "571GB model / 16GB GPU")

## Title (pick one)

1. I Ran a 1-Trillion-Parameter LLM on a $429 GPU
2. 1 Trillion Parameters on a 16GB GPU — Here's How
3. Running Kimi K2.6 (1T params) on a Consumer GPU by Paging Experts from NVMe

## Description

A 1-trillion-parameter language model — Kimi K2.6, with 571GB of expert
weights — generating coherent text on a single 16GB RTX 5060 Ti.

The trick is treating the GPU like a CPU cache and the SSD like memory:
mixture-of-experts models only activate ~3% of their weights per token,
so llmpager keeps the shared core resident in VRAM and pages the routed
experts in from NVMe on demand, with LFU caches in VRAM and host RAM
holding the hot ones. The same open-source engine serves 30B models at
41 tokens/sec interactively, and the 1T tier for overnight agent work —
one OpenAI-compatible API on one home-lab box.

This talk covers the architecture, the DeepSeek-style latent attention
that makes a 256K context nearly free, the bit-exact int4 repack of
Moonshot's QAT weights, two debugging war stories (a tokenizer config
that lied, and a single sign-convention bug with 570 gigabytes of blast
radius), the first-light numbers, and the road from 0.35 to 2+ tokens
per second.

Everything is open source (Apache-2.0, Rust + CUDA):
https://github.com/glennswest/llmpager

Built in the open, two days from git init to first trillion-parameter
tokens. Inspired by turbo-fieldfare's expert-streaming work on Apple
Silicon.

Chapters:
0:00 Cold open — one trillion parameters
0:25 The headline numbers
1:01 Why this shouldn't work
1:53 The architecture
2:35 Kimi vs Qwen: what got harder
3:15 MLA: attention through a keyhole
4:00 Repack, don't requantize
4:39 War story #1: the tokenizer that lied
5:24 War story #2: one sign bit, 570GB blast radius
6:13 First light — the model speaks
6:48 Was it supposed to be that slow?
7:31 The RAM tier
8:13 One box, three model classes
8:48 What's next
9:27 The real headline

## Tags

llmpager, LLM, mixture of experts, MoE, Kimi K2, trillion parameters,
CUDA, Rust, GPU, RTX 5060 Ti, model offloading, expert paging, NVMe,
quantization, int4, DeepSeek, MLA, local LLM, self-hosted AI, homelab,
machine learning, AI on a budget
