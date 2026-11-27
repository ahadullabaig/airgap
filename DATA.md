# Data

Everything under `data/` and `models/` stays local and is git-ignored. This file records what goes there, where it came from, and its terms.

## Layout

```
models/SmolVLM2-500M-Video-Instruct/   model weights and configs (no ONNX files)
data/capera/                            CapERA caption files (train 1,473 + test 1,391 clips)
data/era/                               ERA videos and key frames (manual download, see below)
data/clips/                             other aerial clips (test clips, demo candidates)
```

## Sources

| Item | Source | Terms |
|---|---|---|
| SmolVLM2-500M-Video-Instruct | https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct | See the model card |
| CapERA captions | https://github.com/yakoubbazi/CapEra (`dataset/*.json`) | No licence stated: used for research only, and the reference captions are not redistributed |
| ERA videos | https://lcmou.github.io/ERA_Dataset (Google Drive, `ERA_Dataset.zip`) | Academic use only; do not redistribute (`data/era/README.txt`). No ERA frames in the repo, artifacts or outside services; numbers and file names only |
| `clips/sheboygan_falls.webm` | https://commons.wikimedia.org/wiki/File:Sheboygan_Falls,_Wisconsin;_aerial_drone_video.webm | CC BY 3.0 (attribution required if shown publicly). Used to develop `demo/run_mission.py`, so not a demo video for E3; it also shows a TV station's logo and a place-name banner, which the model reads |
| `clips/or58_truck_recovery.webm` (demo video) | https://commons.wikimedia.org/wiki/File:OR58_Truck_Recovery_drone_video.webm (original file, 1080p) | CC BY 2.0, Oregon Department of Transportation (via Flickr). Credit when shown |
| `clips/minnesota_flood_hwy41_1080p.webm` (demo video) | https://commons.wikimedia.org/wiki/File:Minnesota_River_Flooding_6-30-2024_Drone_Footage_Hwy_41.webm (Commons' 1080p transcode of the 1440p original) | CC BY 3.0, Skye Kiefer. Credit when shown. Burned-in text in the first seconds and from 6:08, outside the segment used |
| `clips/surfing_dana_point_1080p.webm` (demo video) | https://commons.wikimedia.org/wiki/File:Surfing_competition_at_Salt_Creek_Beach,_Dana_Point,_California,_USA_as_viewed_from_a_FPV_drone.webm (Commons' 1080p transcode of the 4K original) | CC BY 3.0, Daniel Peckham. Credit when shown |
| `clips/demo_*.mp4` | Cut from the three demo videos by `demo/prepare_videos.sh`: 3-minute segments (OR58 0:00-3:00, flood 0:40-3:40, surfing 2:00-5:00), 1280x720 H.264, 30 fps, no audio | The source's licence and credit; an edited version of each, so say "cut and re-encoded" when crediting |

## Downloading ERA (manual)

The ERA videos are one Google Drive file: 2,864 clips of about 5.75 s, 24 fps, 640x640.

1. Open https://drive.google.com/file/d/1yxXjDNAq5RAufSgSOE4QmdsNfAc4dOfM/view
2. Click **Download**. Google warns that it can't scan large files for viruses; click **Download anyway**.
3. Move the zip into the repo and unpack it:
   ```
   mv ~/Downloads/ERA_Dataset.zip ~/airgap/data/era/
   cd ~/airgap/data/era && unzip ERA_Dataset.zip
   ```
4. Check: `find ~/airgap/data/era -name '*.mp4' | wc -l` should print 2864, the count on the dataset page.

Command-line alternative to steps 1-3 (may hit Google's download quota on busy days):
```
cd ~/airgap && uvx gdown 1yxXjDNAq5RAufSgSOE4QmdsNfAc4dOfM -O data/era/ERA_Dataset.zip
```

### ERA as downloaded (checked 2026-11-21)

```
data/era/README.txt                                   authors, citation, academic-use notice
data/era/Videos/{Tra,Test}/<Class>/<Class>_NNN .mp4   1,473 train + 1,391 test = 2,864 clips
data/era/SingleFrames/{Tra,Test}/<Class>/<Class>_NNN .png   one key frame per clip
```

- 25 classes (NonEvent is the largest: 192 train, 155 test).
- File names end in `" .mp4"` (a space before the extension). Code maps them to CapERA ids by removing the space; the files are not renamed.
- CapERA matches one to one per split: every caption id has a clip and every clip has 5 captions. Ids repeat across splits (`Baseball_001.mp4` is in both), so key on (split, id).
- All clips: H.264, yuv420p, 640x640, nominal 24 fps, 98-147 frames (median 138, about 5.75 s).
- Per-clip scan (timestamps, repeated frames, static blocks, camera motion, shot changes): `bench/era_scan.py` writes `results/era/scan.json`; summary in `bench/era_report.py`.
