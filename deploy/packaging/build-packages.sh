#!/bin/bash
# Build .deb and .rpm packages for llmpager.
#
# Run on a Linux box with nvcc (kernels compile to PTX at build time),
# dpkg-deb, and rpmbuild (Debian: apt install rpm). The binaries load
# libcuda.so.1 at runtime, so one x86_64 build serves Debian and Fedora
# alike; the only hard runtime dependency is glibc.
set -euo pipefail
cd "$(dirname "$0")/../.."

VERSION=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
ARCH=amd64
DIST=deploy/packaging/dist
STAGE=$DIST/stage

echo "== building llmpager $VERSION =="
cargo build --release -p llmpager-serve -p llmpager-run -p llmpager-convert

rm -rf "$DIST"
mkdir -p "$STAGE/usr/bin" "$STAGE/lib/systemd/system" "$STAGE/etc/llmpager" \
         "$STAGE/usr/share/doc/llmpager" "$STAGE/var/lib/llmpager"
install -m 0755 target/release/llmpager-serve target/release/llmpager-run \
                target/release/llmpager-convert "$STAGE/usr/bin/"
install -m 0644 deploy/packaging/llmpager-packaged.service "$STAGE/lib/systemd/system/llmpager.service"
install -m 0644 deploy/packaging/serve.json.example "$STAGE/etc/llmpager/serve.json"
install -m 0644 README.md CHANGELOG.md docs/DESIGN.md docs/PERFORMANCE.md \
                "$STAGE/usr/share/doc/llmpager/"

# ---------- deb ----------
mkdir -p "$STAGE/DEBIAN"
cat > "$STAGE/DEBIAN/control" <<EOF
Package: llmpager
Version: $VERSION
Architecture: $ARCH
Maintainer: Glenn West <glennswest@neuralcloudcomputing.com>
Section: utils
Priority: optional
Depends: libc6 (>= 2.36)
Recommends: nvidia-driver
Homepage: https://github.com/glennswest/llmpager
Description: MoE expert-paging LLM inference engine for NVIDIA GPUs
 Runs Mixture-of-Experts language models whose weights exceed GPU VRAM by
 keeping the shared core resident and streaming routed experts from NVMe
 through a VRAM LFU cache. Includes an OpenAI-compatible HTTP server
 (multi-model, streaming), a checkpoint converter, and a decode CLI.
 Requires the NVIDIA driver (libcuda) at runtime; no CUDA toolkit needed.
EOF
echo "/etc/llmpager/serve.json" > "$STAGE/DEBIAN/conffiles"
cat > "$STAGE/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
systemctl daemon-reload >/dev/null 2>&1 || true
EOF
chmod 0755 "$STAGE/DEBIAN/postinst"
dpkg-deb --build --root-owner-group "$STAGE" "$DIST/llmpager_${VERSION}_${ARCH}.deb"

# ---------- rpm ----------
rpmbuild -bb \
  --define "_topdir $PWD/$DIST/rpmbuild" \
  --define "version $VERSION" \
  --define "stage $PWD/$STAGE" \
  --buildroot "$PWD/$DIST/rpmbuild/BUILDROOT" \
  deploy/packaging/llmpager.spec
cp "$DIST"/rpmbuild/RPMS/x86_64/llmpager-*.rpm "$DIST/"

echo "== artifacts =="
ls -la "$DIST"/*.deb "$DIST"/*.rpm
