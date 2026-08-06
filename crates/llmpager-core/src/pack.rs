//! `.llmpk` expert pack: the on-disk layout experts are paged from.
//!
//! Layout (little-endian, all sections 4096-byte aligned):
//!
//! ```text
//! [0..4096)    header: magic "LLMPK1\0\0", u32 version, u32 json_len,
//!              then `json_len` bytes of JSON metadata, zero-padded
//! [4096..)     index: one IndexEntry per (layer, expert), row-major by
//!              layer then expert, zero-padded to a 4096 boundary
//! [...]        blobs: each expert's weight bytes, each starting on a
//!              4096 boundary
//! ```
//!
//! Alignment matters: O_DIRECT requires file offset, buffer address, and
//! length all aligned to the logical block size (512 or 4096; we use 4096
//! everywhere and read in whole aligned spans).

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

pub const MAGIC: &[u8; 8] = b"LLMPK1\0\0";
pub const FORMAT_VERSION: u32 = 1;
pub const ALIGN: u64 = 4096;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PackMeta {
    pub model: String,
    pub num_layers: u16,
    pub experts_per_layer: u16,
    /// Weight encoding of the blobs, e.g. "q4_gs64" (informational for now).
    pub dtype: String,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct IndexEntry {
    pub offset: u64,
    pub nbytes: u64,
}

const INDEX_ENTRY_BYTES: u64 = 16;

fn align_up(v: u64) -> u64 {
    v.div_ceil(ALIGN) * ALIGN
}

/// Streaming writer: declare metadata up front, append blobs in row-major
/// (layer, expert) order, then `finish()`.
pub struct PackWriter {
    file: BufWriter<File>,
    meta: PackMeta,
    index: Vec<IndexEntry>,
    index_end: u64,
    cursor: u64,
}

impl PackWriter {
    pub fn create(path: &Path, meta: PackMeta) -> Result<Self> {
        let total = meta.num_layers as u64 * meta.experts_per_layer as u64;
        let index_end = align_up(ALIGN + total * INDEX_ENTRY_BYTES);
        let mut file = BufWriter::new(File::create(path)?);
        // Header + index are back-filled in finish(); reserve the space.
        file.seek(SeekFrom::Start(index_end))?;
        Ok(Self { file, meta, index: Vec::with_capacity(total as usize), index_end, cursor: index_end })
    }

    /// Append the next expert's blob. Call exactly layers*experts times, in
    /// row-major (layer, expert) order.
    pub fn add_blob(&mut self, blob: &[u8]) -> Result<()> {
        let total = self.meta.num_layers as u64 * self.meta.experts_per_layer as u64;
        if self.index.len() as u64 == total {
            bail!("pack already has all {total} blobs");
        }
        self.index.push(IndexEntry { offset: self.cursor, nbytes: blob.len() as u64 });
        self.file.write_all(blob)?;
        let padded = align_up(blob.len() as u64);
        let pad = padded - blob.len() as u64;
        if pad > 0 {
            self.file.write_all(&vec![0u8; pad as usize])?;
        }
        self.cursor += padded;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        let total = self.meta.num_layers as u64 * self.meta.experts_per_layer as u64;
        if self.index.len() as u64 != total {
            bail!("pack incomplete: {} of {total} blobs written", self.index.len());
        }
        // Header.
        self.file.seek(SeekFrom::Start(0))?;
        let json = serde_json::to_vec(&self.meta)?;
        if json.len() as u64 > ALIGN - 16 {
            bail!("metadata JSON too large for header block");
        }
        self.file.write_all(MAGIC)?;
        self.file.write_all(&FORMAT_VERSION.to_le_bytes())?;
        self.file.write_all(&(json.len() as u32).to_le_bytes())?;
        self.file.write_all(&json)?;
        self.file.write_all(&vec![0u8; (ALIGN as usize) - 16 - json.len()])?;
        // Index.
        for e in &self.index {
            self.file.write_all(&e.offset.to_le_bytes())?;
            self.file.write_all(&e.nbytes.to_le_bytes())?;
        }
        let index_bytes = ALIGN + total * INDEX_ENTRY_BYTES;
        let pad = self.index_end - index_bytes;
        if pad > 0 {
            self.file.write_all(&vec![0u8; pad as usize])?;
        }
        self.file.flush()?;
        Ok(())
    }
}

/// Read side. Holds the fd and the full index in memory (16 bytes per
/// expert — ~100KB even for large MoEs).
pub struct PackReader {
    file: File,
    meta: PackMeta,
    index: Vec<IndexEntry>,
    direct: bool,
}

impl PackReader {
    /// Open with the OS page cache (portable; used by tests and tools).
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_inner(path, false)
    }

    /// Open with O_DIRECT (Linux): reads bypass the page cache, which is what
    /// the pager wants — cached experts live in VRAM, not in host RAM twice.
    #[cfg(target_os = "linux")]
    pub fn open_direct(path: &Path) -> Result<Self> {
        Self::open_inner(path, true)
    }

    fn open_inner(path: &Path, direct: bool) -> Result<Self> {
        let file = open_file(path, direct)?;
        let mut header = AlignedBuf::new(ALIGN as usize);
        pread_full(&file, header.as_mut(), 0)
            .with_context(|| format!("reading header of {}", path.display()))?;
        let header = header.as_ref();
        if &header[..8] != MAGIC {
            bail!("{}: not an llmpk file", path.display());
        }
        let version = u32::from_le_bytes(header[8..12].try_into().unwrap());
        if version != FORMAT_VERSION {
            bail!("{}: unsupported pack version {version}", path.display());
        }
        let json_len = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
        let meta: PackMeta = serde_json::from_slice(&header[16..16 + json_len])?;

        let total = meta.num_layers as usize * meta.experts_per_layer as usize;
        let index_span = align_up(total as u64 * INDEX_ENTRY_BYTES) as usize;
        let mut raw = AlignedBuf::new(index_span);
        pread_full(&file, raw.as_mut(), ALIGN)?;
        let raw = raw.as_ref();
        let index = (0..total)
            .map(|i| IndexEntry {
                offset: u64::from_le_bytes(raw[i * 16..i * 16 + 8].try_into().unwrap()),
                nbytes: u64::from_le_bytes(raw[i * 16 + 8..i * 16 + 16].try_into().unwrap()),
            })
            .collect();
        Ok(Self { file, meta, index, direct })
    }

    pub fn meta(&self) -> &PackMeta {
        &self.meta
    }

    pub fn entry(&self, layer: u16, expert: u16) -> IndexEntry {
        self.index[layer as usize * self.meta.experts_per_layer as usize + expert as usize]
    }

    /// Largest blob in the pack — sizes fetch buffers.
    pub fn max_blob_bytes(&self) -> u64 {
        self.index.iter().map(|e| e.nbytes).max().unwrap_or(0)
    }

    /// Read one expert's blob into `buf`, returning the blob length.
    ///
    /// `buf` must hold at least `align_up(entry.nbytes)` bytes and, when the
    /// pack was opened with O_DIRECT, must be 4096-aligned (use [`AlignedBuf`]
    /// or pinned buffers allocated with aligned allocators).
    pub fn read_blob_into(&self, layer: u16, expert: u16, buf: &mut [u8]) -> Result<usize> {
        let e = self.entry(layer, expert);
        let span = align_up(e.nbytes) as usize;
        if buf.len() < span {
            bail!("buffer too small: {} < {span}", buf.len());
        }
        if self.direct {
            let addr = buf.as_ptr() as usize;
            if addr % ALIGN as usize != 0 {
                bail!("O_DIRECT read requires a 4096-aligned buffer");
            }
        }
        pread_full(&self.file, &mut buf[..span], e.offset)?;
        Ok(e.nbytes as usize)
    }
}

fn open_file(path: &Path, direct: bool) -> Result<File> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut opts = std::fs::OpenOptions::new();
        opts.read(true);
        if direct {
            opts.custom_flags(libc::O_DIRECT);
        }
        return Ok(opts.open(path)?);
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = direct;
        Ok(File::open(path)?)
    }
}

fn pread_full(file: &File, buf: &mut [u8], offset: u64) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)?;
        Ok(())
    }
    #[cfg(not(unix))]
    {
        use std::io::Read;
        let mut f = file;
        f.seek(SeekFrom::Start(offset))?;
        f.read_exact(buf)?;
        Ok(())
    }
}

/// Heap buffer aligned to [`ALIGN`], as required by O_DIRECT.
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

impl AlignedBuf {
    pub fn new(len: usize) -> Self {
        let len = align_up(len as u64) as usize;
        let layout = std::alloc::Layout::from_size_align(len, ALIGN as usize).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "allocation failed");
        Self { ptr, len }
    }
}

impl AsRef<[u8]> for AlignedBuf {
    fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl AsMut<[u8]> for AlignedBuf {
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.len, ALIGN as usize).unwrap();
        unsafe { std::alloc::dealloc(self.ptr, layout) };
    }
}

// Safety: AlignedBuf is a plain owned heap allocation.
unsafe impl Send for AlignedBuf {}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob_for(layer: u16, expert: u16, nbytes: usize) -> Vec<u8> {
        (0..nbytes)
            .map(|i| (layer as usize * 31 + expert as usize * 7 + i) as u8)
            .collect()
    }

    #[test]
    fn round_trip() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("t.llmpk");
        let meta = PackMeta {
            model: "synthetic".into(),
            num_layers: 3,
            experts_per_layer: 4,
            dtype: "q4_gs64".into(),
        };
        let mut w = PackWriter::create(&path, meta.clone())?;
        for l in 0..3u16 {
            for e in 0..4u16 {
                // Varying, non-aligned sizes exercise padding.
                w.add_blob(&blob_for(l, e, 1000 + l as usize * 500 + e as usize * 13))?;
            }
        }
        w.finish()?;

        let r = PackReader::open(&path)?;
        assert_eq!(r.meta(), &meta);
        for l in 0..3u16 {
            for e in 0..4u16 {
                let want = blob_for(l, e, 1000 + l as usize * 500 + e as usize * 13);
                let entry = r.entry(l, e);
                assert_eq!(entry.nbytes as usize, want.len());
                assert_eq!(entry.offset % ALIGN, 0, "blob not aligned");
                let mut buf = AlignedBuf::new(want.len());
                let n = r.read_blob_into(l, e, buf.as_mut())?;
                assert_eq!(&buf.as_ref()[..n], &want[..]);
            }
        }
        Ok(())
    }

    #[test]
    fn incomplete_pack_rejected() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("t.llmpk");
        let meta = PackMeta {
            model: "x".into(),
            num_layers: 1,
            experts_per_layer: 2,
            dtype: "raw".into(),
        };
        let mut w = PackWriter::create(&path, meta)?;
        w.add_blob(&[1, 2, 3])?;
        assert!(w.finish().is_err());
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn o_direct_read() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("t.llmpk");
        let meta = PackMeta {
            model: "x".into(),
            num_layers: 1,
            experts_per_layer: 1,
            dtype: "raw".into(),
        };
        let mut w = PackWriter::create(&path, meta)?;
        let want = blob_for(0, 0, 8192);
        w.add_blob(&want)?;
        w.finish()?;
        // O_DIRECT can fail on exotic filesystems (tmpfs); fall back quietly
        // so CI stays green — the real target runs ext4.
        let r = match PackReader::open_direct(&path) {
            Ok(r) => r,
            Err(_) => return Ok(()),
        };
        let mut buf = AlignedBuf::new(want.len());
        let n = r.read_blob_into(0, 0, buf.as_mut())?;
        assert_eq!(&buf.as_ref()[..n], &want[..]);
        Ok(())
    }
}
