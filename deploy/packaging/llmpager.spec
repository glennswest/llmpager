Name:           llmpager
Version:        %{version}
Release:        1%{?dist}
Summary:        MoE expert-paging LLM inference engine for NVIDIA GPUs
License:        Apache-2.0
URL:            https://github.com/glennswest/llmpager
AutoReqProv:    no

%description
llmpager runs Mixture-of-Experts language models whose weights exceed GPU
VRAM by keeping the shared core resident and streaming routed experts from
NVMe through a VRAM LFU cache. Includes an OpenAI-compatible HTTP server
(multi-model, streaming), a checkpoint converter, and a decode CLI.
Requires the NVIDIA driver (libcuda) at runtime; no CUDA toolkit needed.

%install
mkdir -p %{buildroot}/usr/bin %{buildroot}/usr/lib/systemd/system \
         %{buildroot}/etc/llmpager %{buildroot}/usr/share/doc/llmpager \
         %{buildroot}/var/lib/llmpager
install -m 0755 %{stage}/usr/bin/* %{buildroot}/usr/bin/
install -m 0644 %{stage}/lib/systemd/system/llmpager.service %{buildroot}/usr/lib/systemd/system/
install -m 0644 %{stage}/etc/llmpager/serve.json %{buildroot}/etc/llmpager/
install -m 0644 %{stage}/usr/share/doc/llmpager/* %{buildroot}/usr/share/doc/llmpager/

%post
systemctl daemon-reload >/dev/null 2>&1 || :

%postun
systemctl daemon-reload >/dev/null 2>&1 || :

%files
/usr/bin/llmpager-serve
/usr/bin/llmpager-run
/usr/bin/llmpager-convert
/usr/lib/systemd/system/llmpager.service
%dir /var/lib/llmpager
%config(noreplace) /etc/llmpager/serve.json
%doc /usr/share/doc/llmpager/README.md
%doc /usr/share/doc/llmpager/CHANGELOG.md
%doc /usr/share/doc/llmpager/DESIGN.md
%doc /usr/share/doc/llmpager/PERFORMANCE.md
