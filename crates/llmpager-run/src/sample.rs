//! Token sampling: temperature / top-k / top-p / repetition penalty over
//! the final logits. Greedy (temperature 0) stays the default everywhere —
//! benchmarks and perplexity need determinism.

#[derive(Debug, Clone)]
pub struct Sampling {
    /// 0.0 => greedy argmax (default).
    pub temperature: f32,
    /// Nucleus mass; 1.0 disables.
    pub top_p: f32,
    /// Candidate cap before top-p; 0 disables.
    pub top_k: usize,
    /// >1.0 penalizes tokens present in `recent` (llama.cpp convention:
    /// positive logits divided, negative multiplied).
    pub repeat_penalty: f32,
    /// How many trailing tokens the penalty looks at.
    pub repeat_window: usize,
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
            repeat_window: 64,
            seed: 0x5eed,
        }
    }
}

impl Sampling {
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

/// Deterministic xorshift; one per generation stream.
pub struct SampleRng(u64);

impl SampleRng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn unit(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
}

pub fn sample(logits: &[f32], recent: &[u32], s: &Sampling, rng: &mut SampleRng) -> u32 {
    if s.is_greedy() && s.repeat_penalty == 1.0 {
        return argmax(logits);
    }

    let mut adjusted: Vec<f32> = logits.to_vec();
    if s.repeat_penalty != 1.0 {
        let start = recent.len().saturating_sub(s.repeat_window);
        for &t in &recent[start..] {
            let v = &mut adjusted[t as usize];
            *v = if *v > 0.0 { *v / s.repeat_penalty } else { *v * s.repeat_penalty };
        }
    }
    if s.is_greedy() {
        return argmax(&adjusted);
    }

    // Candidates sorted by logit, top-k capped.
    let mut idx: Vec<u32> = (0..adjusted.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| {
        adjusted[b as usize].partial_cmp(&adjusted[a as usize]).unwrap()
    });
    if s.top_k > 0 && s.top_k < idx.len() {
        idx.truncate(s.top_k);
    }

    // Softmax at temperature over the candidate set.
    let m = adjusted[idx[0] as usize];
    let mut probs: Vec<f32> =
        idx.iter().map(|&i| ((adjusted[i as usize] - m) / s.temperature).exp()).collect();
    let z: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= z;
    }

    // Nucleus cut.
    if s.top_p < 1.0 {
        let mut acc = 0.0f32;
        let mut keep = probs.len();
        for (i, p) in probs.iter().enumerate() {
            acc += p;
            if acc >= s.top_p {
                keep = i + 1;
                break;
            }
        }
        idx.truncate(keep);
        probs.truncate(keep);
        let z: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= z;
        }
    }

    let r = rng.unit();
    let mut acc = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return idx[i];
        }
    }
    idx[probs.len() - 1]
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut bestv = f32::MIN;
    for (i, v) in logits.iter().enumerate() {
        if *v > bestv {
            bestv = *v;
            best = i as u32;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_argmax() {
        let s = Sampling::default();
        let mut rng = SampleRng::new(1);
        assert_eq!(sample(&[0.1, 2.0, -1.0], &[], &s, &mut rng), 1);
    }

    #[test]
    fn repeat_penalty_flips_choice() {
        let s = Sampling { repeat_penalty: 2.0, ..Default::default() };
        let mut rng = SampleRng::new(1);
        // Token 1 leads but was just used; penalty halves it below token 0.
        assert_eq!(sample(&[1.5, 2.0, -1.0], &[1], &s, &mut rng), 0);
    }

    #[test]
    fn top_k_restricts_support() {
        let s = Sampling { temperature: 1.0, top_k: 2, seed: 7, ..Default::default() };
        let mut rng = SampleRng::new(7);
        for _ in 0..200 {
            let t = sample(&[5.0, 4.0, -10.0, -10.0], &[], &s, &mut rng);
            assert!(t == 0 || t == 1, "sampled outside top-k: {t}");
        }
    }

    #[test]
    fn top_p_prunes_tail() {
        let s = Sampling { temperature: 1.0, top_p: 0.5, ..Default::default() };
        let mut rng = SampleRng::new(3);
        // First candidate holds ~73% of mass; nucleus 0.5 keeps only it.
        for _ in 0..100 {
            assert_eq!(sample(&[2.0, 1.0, 0.0, -1.0], &[], &s, &mut rng), 0);
        }
    }

    #[test]
    fn deterministic_for_seed() {
        let s = Sampling { temperature: 0.9, top_k: 8, seed: 42, ..Default::default() };
        let logits: Vec<f32> = (0..64).map(|i| ((i * 37) % 13) as f32 * 0.3).collect();
        let mut a = SampleRng::new(42);
        let mut b = SampleRng::new(42);
        for _ in 0..50 {
            assert_eq!(sample(&logits, &[], &s, &mut a), sample(&logits, &[], &s, &mut b));
        }
    }
}
