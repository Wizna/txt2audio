# txt2audio

将中文小说转换成有声书，支持 `.txt`、`.epub`、`.mobi` 输入，基于 [Fun-CosyVoice3-0.5B](https://github.com/FunAudioLLM/CosyVoice) 做零样本 TTS，也可以进一步导出带封面和烧录字幕的 MP4。

## 功能

- 自动识别卷/章结构，支持序、楔子、后记等特殊章节
- 自动检测文本编码
- `epub` / `mobi` 自动转换为可复用的 `*.txt2audio.txt`
- 使用参考音频做声音克隆
- 支持断点续生成；音频、字幕、视频均采用临时文件后原子替换
- 可生成 SRT 字幕，或在视频模式下直接烧录字幕
- 提供 JSON 输出、JSONL 事件流和 MCP server，便于 agent 集成

## 环境要求

| 依赖 | 说明 |
|------|------|
| Python 3.11 / 3.12 | `pyproject.toml` 当前约束为 `>=3.11,<3.13` |
| [uv](https://github.com/astral-sh/uv) | Python 包管理器 |
| `ffmpeg` | 输出 MP3 或 MP4 时必需；仅输出 WAV 时不需要 |
| `ebook-convert`（Calibre，可选） | 处理 `epub` / `mobi` 时需要 |

当前明确验证的平台是 macOS Apple Silicon。Windows x64 的路径兼容和部分测试已补齐，但完整媒体生成链路仍应视为实验性支持。

如果需要字幕烧录，请确认 ffmpeg 带有 `subtitles` 滤镜；可用下面的命令检查：

```bash
ffmpeg -filters | grep subtitles
```

部分 Homebrew 构建默认不带 `libass`，这种情况下视频仍可生成，但会跳过字幕烧录。

## 快速开始

```bash
git clone --recursive https://github.com/Wizna/txt2audio.git
cd txt2audio
uv sync
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')"
```

如果仓库已经克隆但没带子模块，补一次：

```bash
git submodule update --init --recursive
```

## 常用命令

```bash
uv run txt2audio book.txt --range all
uv run txt2audio book.txt --range all --srt
uv run txt2audio book.txt --video --range all
uv run txt2audio book.txt --video --keep-srt --range 0~8
uv run txt2audio book.epub --range all
uv run txt2audio book.txt --range all --speed 0.95 --output-format wav
uv run txt2audio book.txt --range all --set audio.mp3_bitrate=192k
uv run txt2audio book.txt --range all --chapter-manifest
```

说明：

- 不传 `--range` 时，交互模式会先显示目录并提示选择；非交互模式会默认处理全部章节
- 自动化脚本、agent 或 CI 调用时，建议显式传 `--range`
- `epub` / `mobi` 会先转换为同目录下的 `*.txt2audio.txt`，后续运行优先复用这个文本文件

## Agent / 结构化输出

```bash
uv run txt2audio book.txt --range all --json
uv run txt2audio --dump-config --json
uv run txt2audio book.txt --validate-paths --json
uv run txt2audio book.txt --plan-json
uv run txt2audio book.txt --range all --json --events-jsonl events.jsonl
uv run txt2audio book.txt --range all --quiet
```

- `--json`：最终结果写到 stdout
- 默认仅显示书名、章节级进度和最终摘要；`--verbose` 可恢复模型/依赖层诊断输出
- `--events-jsonl PATH`：进度事件写到 JSONL；传 `-` 时写到 stderr
- `schemas/`：包含运行结果、计划结果、路径校验、事件流、错误 envelope、章节清单的 schema
- 退出码：`0` 成功，`1` 失败

## MCP

项目内置 `txt2audio-mcp`，复用同一套 CLI 和 `schemas/` 契约。

可用工具：

- `txt2audio_validate_book`
- `txt2audio_plan_conversion`
- `txt2audio_convert_book`
- `txt2audio_get_manifest`

示例配置：

```json
{
  "mcpServers": {
    "txt2audio": {
      "command": "uv",
      "args": ["run", "txt2audio-mcp"]
    }
  }
}
```

## 输出结构

```text
output/{书名}/
├── 目录.txt
├── chapter_manifest.json          # 仅在 --chapter-manifest 时生成
├── {章名}-{片段}.mp3              # 纯音频默认输出
├── {章名}-{片段}.wav              # --output-format wav
├── {章名}-{片段}.srt              # --srt 或 --keep-srt
└── {章名}-{片段}.mp4              # --video
```

- 如果存在卷结构，章节文件会落在对应卷目录下
- `目录.txt` 会在解析章节时自动生成
- 视频模式会在对应目录生成并复用 `cover.jpg`
- 默认纯音频输出是 `mp3`；视频模式最终输出是 `mp4`

## 配置

编辑根目录的 `config.yaml` 即可。配置文件里的相对路径会以项目根目录为基准解析；运行时通过 `--dump-config` 看到的是已展开后的绝对路径。

最常用的配置项：

```yaml
tts:
  model_dir: pretrained_models/Fun-CosyVoice3-0.5B
  speaker_wav: resources/my_speaker.wav
  prompt_text: "You are a helpful assistant.<|endofprompt|>参考音频中的原文"
  speed: 1.05

audio:
  max_chars_per_clip: 6300
  output_format: mp3
  mp3_bitrate: 128k

video:
  orientation: portrait
  subtitles: true
  subtitle_style: "FontSize=20,PrimaryColour=&H00ffff,Outline=1,OutlineColour=&H000000,Alignment=2,MarginL=40,MarginR=40,MarginV=90"

paths:
  output_dir: output
```

注意：

- `prompt_text` 中 `<|endofprompt|>` 后的文本必须与 `speaker_wav` 里的实际朗读内容逐字一致
- `max_chars_per_clip = -1` 表示整章不分片
- `--speed`、`--output-format`、`--output-dir`、`--set KEY=VALUE` 都会覆盖 `config.yaml`，并刷新运行时常量

### 字幕位置

字幕样式由 `video.subtitle_style` 控制。当前默认值里：

- `FontSize=20`
- `Alignment=2`：底部居中
- `MarginV=90`：底部边距

这里的 `MarginV` 不是最终视频上的固定像素，而是 ASS 坐标空间里的值。实际像素约等于：

`MarginV * 视频高度 / 288`

例如：

- 竖屏 `1280` 高时约为 `400px`
- 横屏 `720` 高时约为 `225px`

## 开发校验

仓库已经有 `tests/`。修改 `src/` 后，至少建议跑这些检查：

```bash
python -m compileall src
uv run pytest
uv run txt2audio --dump-config --json
uv run txt2audio sample.txt --range 0~0
```

如果改了视频或字幕逻辑，再额外跑一次：

```bash
uv run txt2audio sample.txt --video --range 0~0 --set video.subtitles=true
```

## 代码结构

```text
src/
├── transform_to_audio.py   # CLI 入口
├── utility.py              # 章节解析、TTS、导出、CLI 主流程
├── video.py                # 封面生成、MP4 导出
├── subtitle.py             # SRT 生成
├── config.py               # 配置加载与路径规范化
├── pathing.py              # 输出路径与跨平台清理
└── txt2audio_mcp.py        # MCP server
```

`third_party/CosyVoice/` 是 vendored 上游依赖，除非必要不要随意改动。
