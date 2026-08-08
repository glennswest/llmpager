//! With the `kernels` feature, compile kernels/*.cu to PTX at build time.
//! PTX is JIT-compiled by the driver for the actual GPU (compute_80 forward-
//! compatible through Blackwell), so no per-arch cubins are needed.

use std::path::PathBuf;
use std::process::Command;

const KERNELS: &[&str] = &["q4g64", "decode", "mla"];

fn main() {
    for k in KERNELS {
        println!("cargo:rerun-if-changed=kernels/{k}.cu");
    }
    println!("cargo:rerun-if-env-changed=NVCC");
    if std::env::var("CARGO_FEATURE_KERNELS").is_err() {
        return;
    }

    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| {
        for cand in ["/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13.3/bin/nvcc"] {
            if std::path::Path::new(cand).exists() {
                return cand.to_string();
            }
        }
        "nvcc".to_string()
    });

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    for k in KERNELS {
        let out = out_dir.join(format!("{k}.ptx"));
        let status = Command::new(&nvcc)
            .args(["--ptx", "-O3", "-arch=compute_80"])
            .arg(format!("kernels/{k}.cu"))
            .arg("-o")
            .arg(&out)
            .status()
            .unwrap_or_else(|e| panic!("running {nvcc}: {e} (set NVCC or install cuda-nvcc)"));
        assert!(status.success(), "nvcc failed compiling kernels/{k}.cu");
    }
}
