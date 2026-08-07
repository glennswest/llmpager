#!/bin/bash
# Build the narrated presentation video (llmpager-video.mp4).
#
# Audio per slide, in order of preference:
#   1. voice/slideNN.(m4a|wav|aiff|mp3)  — your own recordings
#   2. macOS `say` TTS from narration/slideNN.txt  (VOICE env overrides voice)
#
# Requires: ffmpeg, and either Google Chrome (slide rendering) or existing
# frames/slideNN.png.
set -euo pipefail
cd "$(dirname "$0")"

SLIDES=20
VOICE="${VOICE:-Samantha}"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PAD=0.9   # seconds of silence appended after each slide's narration

mkdir -p frames tts segs
rm -f segs/*.mp4 concat.txt

for i in $(seq 1 $SLIDES); do
  nn=$(printf "%02d" "$i")

  # 1. Frame
  if [ ! -f "frames/slide$nn.png" ]; then
    "$CHROME" --headless=new --disable-gpu --hide-scrollbars \
      --window-size=1920,1080 --screenshot="frames/slide$nn.png" \
      "file://$PWD/llmpager.html#$i" >/dev/null 2>&1
  fi

  # 2. Audio
  aud=""
  for ext in m4a wav aiff mp3; do
    [ -f "voice/slide$nn.$ext" ] && aud="voice/slide$nn.$ext" && break
  done
  if [ -z "$aud" ]; then
    say -v "$VOICE" -o "tts/slide$nn.aiff" -f "narration/slide$nn.txt"
    aud="tts/slide$nn.aiff"
  fi

  # 3. Segment: still frame held for narration + padding
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$aud")
  total=$(python3 -c "print(f'{float('$dur') + $PAD:.2f}')")
  ffmpeg -v error -y -loop 1 -framerate 30 -i "frames/slide$nn.png" -i "$aud" \
    -af "apad=pad_dur=$PAD" -t "$total" \
    -c:v libx264 -tune stillimage -pix_fmt yuv420p -r 30 \
    -c:a aac -b:a 160k -ar 48000 "segs/slide$nn.mp4"
  echo "file 'segs/slide$nn.mp4'" >> concat.txt
  echo "slide $nn: $total s ($(basename "$aud"))"
done

ffmpeg -v error -y -f concat -safe 0 -i concat.txt -c copy llmpager-video.mp4
echo "== done =="
ffprobe -v error -show_entries format=duration,size -of default=nw=1 llmpager-video.mp4
