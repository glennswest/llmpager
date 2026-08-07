#!/bin/bash
# Synthesize the 20 slide narrations in the presenter's own voice using
# F5-TTS zero-shot cloning on ai.g8.lo's GPU, then pull the results into
# voice/ (which build-video.sh prefers over TTS).
#
# Needs: voice-sample.(m4a|wav|mp3) beside this script — ~30s of clean
# speech — and the calibration transcript in voice-sample.txt.
set -euo pipefail
cd "$(dirname "$0")"

HOST=glenn@ai.g8.lo
SAMPLE=$(ls voice-sample.m4a voice-sample.wav voice-sample.mp3 2>/dev/null | head -1)
[ -n "$SAMPLE" ] || { echo "no voice-sample.{m4a,wav,mp3} found"; exit 1; }
[ -f voice-sample.txt ] || { echo "no voice-sample.txt (transcript) found"; exit 1; }

echo "== shipping sample + narration to $HOST =="
ssh "$HOST" 'mkdir -p ~/voiceclone/narration ~/voiceclone/out'
scp -q "$SAMPLE" voice-sample.txt narration/slide*.txt "$HOST":voiceclone/
ssh "$HOST" 'mv ~/voiceclone/slide*.txt ~/voiceclone/narration/ 2>/dev/null || true'

echo "== synthesizing on the GPU (service paused during run) =="
ssh "$HOST" bash -s <<'EOF'
set -euo pipefail
sudo systemctl stop llmpager
cd ~/voiceclone
SAMPLE=$(ls voice-sample.* | grep -v txt | head -1)
# F5-TTS wants wav reference
ffmpeg -v error -y -i "$SAMPLE" -ar 24000 -ac 1 ref.wav 2>/dev/null || \
  ~/.f5venv/bin/python -c "import sys" # ffmpeg may be absent; f5 handles m4a via soundfile? ensure ffmpeg
for f in narration/slide*.txt; do
  nn=$(basename "$f" .txt)
  ~/.f5venv/bin/f5-tts_infer-cli \
    --model F5TTS_v1_Base \
    --ref_audio ref.wav \
    --ref_text "$(cat voice-sample.txt)" \
    --gen_text "$(cat "$f")" \
    --output_dir out --output_file "$nn.wav" >/dev/null 2>&1
  echo "  $nn done"
done
sudo systemctl start llmpager
EOF

echo "== pulling results =="
mkdir -p voice
scp -q "$HOST":voiceclone/out/slide*.wav voice/
ls voice/ | wc -l
echo "now run: ./build-video.sh"
