#!/usr/bin/env bash
# Cut the three demo videos (sources and licences in DATA.md) to their 3-minute segments, as 1280x720 H.264 at
# 30 fps with no audio, so every demo run decodes the same kind of file. The model sees each frame at 512x512,
# so 720p loses nothing it uses. Segments chosen 2026-11-27 from contact sheets, before any demo run; they avoid
# the burned-in text in the flood video (its first seconds and its last minute) and the OR58 end card.
#
#   bash demo/prepare_videos.sh        # reads data/clips/*.webm, writes data/clips/demo_*.mp4
set -euo pipefail
cd "$(dirname "$0")/../data/clips"

cut() {  # source start_s output
    ffmpeg -loglevel error -y -ss "$2" -i "$1" -t 180 -vf "scale=1280:720,fps=30" -an \
        -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p "$3"
}
cut or58_truck_recovery.webm            0   demo_or58_truck_recovery.mp4      # original 0:00-3:00
cut minnesota_flood_hwy41_1080p.webm   40   demo_minnesota_flood.mp4          # original 0:40-3:40
cut surfing_dana_point_1080p.webm     120   demo_surfing_dana_point.mp4       # original 2:00-5:00
