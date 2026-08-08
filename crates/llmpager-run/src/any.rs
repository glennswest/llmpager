//! Architecture dispatch: one decode surface over the Qwen3-MoE and
//! Kimi/DeepSeek engines. The pack's embedded config decides which engine
//! loads (`kv_lora_rank` present => Kimi).

use std::path::Path;

use anyhow::Result;

use crate::{decode, kimi};

pub enum AnyDecoder {
    Qwen(decode::Decoder),
    Kimi(kimi::KimiDecoder),
}

impl AnyDecoder {
    /// Auto-detecting constructor. `core_q4` applies to the Qwen path only
    /// (the Kimi core is always q4-requantized — it has no other way to fit).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pack: &Path,
        core: &Path,
        slots: u32,
        io_threads: usize,
        max_seq: usize,
        core_q4: bool,
        direct: bool,
        ram_bytes: u64,
    ) -> Result<Self> {
        let meta = llmpager_core::pack::PackReader::open(pack)?.meta().clone();
        if kimi::KimiConfig::is_kimi(&meta.config) {
            Ok(AnyDecoder::Kimi(kimi::KimiDecoder::new(
                pack, core, slots, io_threads, max_seq, direct, ram_bytes,
            )?))
        } else {
            Ok(AnyDecoder::Qwen(decode::Decoder::new(
                pack, core, slots, io_threads, max_seq, core_q4, direct, ram_bytes,
            )?))
        }
    }

    pub fn is_kimi(&self) -> bool {
        matches!(self, AnyDecoder::Kimi(_))
    }

    pub fn step(&mut self, token: u32, pos: usize, want_logits: bool) -> Result<u32> {
        match self {
            AnyDecoder::Qwen(d) => d.step(token, pos, want_logits),
            AnyDecoder::Kimi(d) => d.step(token, pos, want_logits),
        }
    }

    pub fn step_chunk(&mut self, tokens: &[u32], start_pos: usize, want_logits: bool) -> Result<u32> {
        match self {
            AnyDecoder::Qwen(d) => d.step_chunk(tokens, start_pos, want_logits),
            AnyDecoder::Kimi(d) => d.step_chunk(tokens, start_pos, want_logits),
        }
    }

    pub fn chunk_cap(&self) -> usize {
        match self {
            AnyDecoder::Qwen(d) => d.chunk_cap(),
            AnyDecoder::Kimi(d) => d.chunk_cap(),
        }
    }

    pub fn last_logits(&self) -> Vec<f32> {
        match self {
            AnyDecoder::Qwen(d) => d.last_logits(),
            AnyDecoder::Kimi(d) => d.last_logits(),
        }
    }

    pub fn pager_metrics(&self) -> llmpager_cuda::pager::Metrics {
        match self {
            AnyDecoder::Qwen(d) => d.pager_metrics(),
            AnyDecoder::Kimi(d) => d.pager_metrics(),
        }
    }

    pub fn eos(&self) -> &[u32] {
        match self {
            AnyDecoder::Qwen(d) => &d.cfg.eos,
            AnyDecoder::Kimi(d) => &d.cfg.eos,
        }
    }

    pub fn resize_cache(&mut self, slots: u32) -> Result<()> {
        match self {
            AnyDecoder::Qwen(d) => d.resize_cache(slots),
            AnyDecoder::Kimi(d) => d.resize_cache(slots),
        }
    }

    pub fn set_prefetch_next(&mut self, on: bool) {
        if let AnyDecoder::Qwen(d) = self {
            d.prefetch_next = on;
        }
    }

    pub fn set_min_expert_weight(&mut self, w: f32) {
        if let AnyDecoder::Kimi(d) = self {
            d.min_expert_weight = w;
        }
    }
}
