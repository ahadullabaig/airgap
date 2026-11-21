# Data

Everything under `data/` and `models/` stays local and is git-ignored. This file records what goes there, where it came from, and its terms.

## Layout

```
models/SmolVLM2-500M-Video-Instruct/   model weights and configs (no ONNX files)
data/capera/                            CapERA caption files (train 1,473 + test 1,391 clips)
data/era/                               ERA videos (manual download, see below)
data/clips/                             other aerial clips (test clips, demo candidates)
```

## Sources

| Item | Source | Terms |
|---|---|---|
| SmolVLM2-500M-Video-Instruct | https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct | See the model card |
| CapERA captions | https://github.com/yakoubbazi/CapEra (`dataset/*.json`) | No licence stated; coursework only until confirmed |
| ERA videos | https://lcmou.github.io/ERA_Dataset (Google Drive, `ERA_Dataset.zip`) | See the dataset page; coursework only until confirmed |
| `clips/sheboygan_falls.webm` | https://commons.wikimedia.org/wiki/File:Sheboygan_Falls,_Wisconsin;_aerial_drone_video.webm | CC BY 3.0 (attribution required if shown publicly) |

## Downloading ERA (manual)

The ERA videos are one Google Drive file: 2,864 clips, 5 s each, 24 fps, 640x640.

1. Open https://drive.google.com/file/d/1yxXjDNAq5RAufSgSOE4QmdsNfAc4dOfM/view
2. Click **Download**. Google warns that it can't scan large files for viruses; click **Download anyway**.
3. Move the zip into the repo and unpack it:
   ```
   mv ~/Downloads/ERA_Dataset.zip ~/airgap/data/era/
   cd ~/airgap/data/era && unzip ERA_Dataset.zip
   ```
4. Check: `find ~/airgap/data/era -name '*.mp4' | wc -l` should print 2864, the count on the dataset page. The zip's folder layout and file type are unchecked until it arrives.

Command-line alternative to steps 1-3 (may hit Google's download quota on busy days):
```
cd ~/airgap && uvx gdown 1yxXjDNAq5RAufSgSOE4QmdsNfAc4dOfM -O data/era/ERA_Dataset.zip
```
