# txt2audio

> 将网络小说转换成有声书（主要是为了保护我的眼睛）

将中文小说转换为有声书，支持 `.txt`、`.epub`、`.mobi` 输入。使用 [Fun-CosyVoice3-0.5B](https://github.com/FunAudioLLM/CosyVoice) 进行零样本语音合成（zero-shot TTS），支持生成带封面的 MP4 视频。

## 功能特性

- **智能文本处理**
  - 支持 `.txt`、`.epub`、`.mobi` 输入
  - 自动识别卷/章结构（支持序、楔子、后记等特殊章节）
  - 自动检测文件编码
  - `epub` / `mobi` 自动转换为可复用的 `.txt2audio.txt`
  - 三层文本预处理，各层各司其职：
    - 小说专用标点清洗 —— 破折号、省略号、引号、URL 等
    - `wetext` 文本归一化 —— 数字/符号自动转口语形式
    - CosyVoice 内置清理与分段
  - 多音字智能标注 —— 使用 `pypinyin` 根据上下文为多音字（如 的/得/地、为/wéi/wèi）添加拼音标注
- **音频生成**
  - 声音克隆 —— 提供一段参考音频即可用该声音朗读全书
  - 每个音频片段约 30 分钟（6300 汉字）
  - 断点续生成 —— 原子写入（先写临时文件再重命名），中断不会产生损坏文件
- **视频输出**（可选）
  - 生成 MP4 视频，带自动生成的封面图
  - 支持横屏/竖屏（`--landscape` 或 `config.yaml` 中设置）
  - 可选内嵌字幕 —— 自动生成句级 SRT 字幕并烧入视频

## 环境要求

| 依赖 | 说明 |
|------|------|
| Python 3.11+ | 见 `.python-version` |
| [uv](https://github.com/astral-sh/uv) | Python 包管理器 |
| `ffmpeg`（可选） | 输出 MP3 或生成视频时需要；仅输出 WAV 时不需要 |
| `ebook-convert`（Calibre，可选） | 输入为 `epub` / `mobi` 时需要；输入为 `.txt` 时不需要 |

## 快速开始

```bash
# 1. 克隆仓库（含子模块）
git clone --recursive https://github.com/user/txt2audio.git
cd txt2audio

# 2. 安装依赖
uv sync

# 3. 下载预训练模型（约 1GB）
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')"
```

按需安装系统依赖：

- `ffmpeg`：仅当你需要输出 MP3，或使用 `--video` 生成 MP4 时安装
- `ebook-convert`（Calibre）：仅当你要处理 `epub` / `mobi` 输入时安装

不同系统的安装方式不同，请使用各自平台对应的包管理器或安装器。

<details>
<summary>已克隆但没有拉取子模块？</summary>

```bash
git submodule update --init --recursive
```

</details>

## 使用方法

**基本用法：**

```bash
uv run txt2audio your_book.txt --range all              # 纯音频（默认 MP3）
uv run txt2audio your_book.epub --range all             # 首次会先生成 your_book.txt2audio.txt
uv run txt2audio your_book.mobi --range all             # 首次会先生成 your_book.txt2audio.txt
uv run txt2audio your_book.txt --video --range all      # 音频 + 视频（MP4）
```

`epub` / `mobi` 输入会先调用 Calibre 的 `ebook-convert`，在原书文件同目录生成 `*.txt2audio.txt`。后续再次运行时会优先复用这个文本文件，因此你可以先手动修正文稿，再继续使用原始电子书路径执行转换。若输入本身就是 `.txt`，则不需要安装 Calibre。

**指定章节范围：**

```bash
uv run txt2audio your_book.txt --range 0~8              # 第 0 到第 8 章
uv run txt2audio your_book.txt --range 5                # 仅第 5 章
```

**运行时覆盖配置：**

```bash
uv run txt2audio your_book.txt --range all --speed 0.95 --output-format wav
uv run txt2audio your_book.txt --range all --set tts.speed=0.9 --set audio.mp3_bitrate=192k
```

**组合示例：**

```bash
uv run txt2audio your_book.txt --video --landscape --range all
```

> 不带 `--range` 时，交互模式会提示选择范围；非交互模式（如管道/脚本）自动处理全部章节。

### AI / Agent 集成

```bash
# JSON 结构化输出（stdout 为 JSON，进度信息走 stderr）
uv run txt2audio your_book.txt --range all --json

# 查看当前生效配置
uv run txt2audio --dump-config --json

# 静默模式（仅输出错误）
uv run txt2audio your_book.txt --range all --quiet
```

退出码：`0` 成功，`1` 失败。`--json` 模式输出示例：

```json
{
  "status": "success",
  "book_name": "三体",
  "chapters_generated": 8,
  "chapters_skipped": 0,
  "total_clips": 24,
  "output_format": "mp3",
  "output_directory": "output/三体",
  "source_text_file": "/path/to/三体.txt2audio.txt",
  "elapsed_seconds": 120.5
}
```

## 输出结构

```
output/{书名}/
├── 目录.txt                              # 自动生成的目录
└── {卷名}/
    ├── {章名}-{片段}.mp3                  # 默认（纯音频，MP3 128kbps）
    ├── {章名}-{片段}.mp4                  # --video 模式
    └── {章名}-{片段}.srt                  # 中间字幕文件（烧入视频后自动删除）
```

## 项目结构

```
txt2audio/
├── src/
│   ├── transform_to_audio.py              # 入口
│   ├── utility.py                         # 文本解析、音频生成、CLI 逻辑
│   ├── video.py                           # 封面图生成、ffmpeg 转视频
│   ├── subtitle.py                        # SRT 字幕生成
│   └── config.py                          # 配置加载
├── config.yaml                            # 运行时配置（TTS、音频、视频、路径）
├── resources/                             # 字体文件、参考音频
├── third_party/CosyVoice/                 # CosyVoice 子模块
└── pretrained_models/                     # 模型权重（gitignored）
```

对于 `epub` / `mobi` 输入，程序还会在原书文件同目录生成一个可编辑的中间文本文件：

```text
your_book.epub
your_book.txt2audio.txt
```

## 配置

编辑项目根目录的 `config.yaml` 即可自定义参数，无需改代码。所有路径均相对于项目根目录，完整配置项及说明见 `config.yaml`。

<details>
<summary><b>TTS 配置</b></summary>

```yaml
tts:
  model_dir: pretrained_models/Fun-CosyVoice3-0.5B
  speaker_wav: resources/my_speaker.wav    # 说话人参考音频（声音克隆来源）
  prompt_text: "..."                       # 零样本 TTS 提示文本
  speed: 1.05                              # 语速（有声书建议 0.9~1.0）
  inter_sentence_silence_ms: 150           # 句间静音（毫秒，0=禁用）
```

#### 自定义声音（声音克隆）

1. 准备一段清晰、无噪音的参考音频（建议 5~15 秒）
2. 放入 `resources/` 目录
3. 修改 `config.yaml` 中的 `speaker_wav` 和 `prompt_text`：

```yaml
tts:
  speaker_wav: resources/my_speaker.wav
  prompt_text: "You are a helpful assistant.<|endofprompt|>参考音频中说的原文。"
```

> **注意：** `<|endofprompt|>` 后面的文本必须与参考音频内容**逐字一致**，否则音色提取会失败。

</details>

<details>
<summary><b>音频配置</b></summary>

```yaml
audio:
  max_chars_per_clip: 6300                 # 每个音频片段的汉字上限（-1 = 不分片）
  model_sentence_limit: 2000              # TTS 模型单次输入的字符上限
  output_format: mp3                       # 纯音频输出格式：mp3 或 wav
  mp3_bitrate: 128k                        # MP3 码率
```

</details>

<details>
<summary><b>视频配置</b></summary>

```yaml
video:
  orientation: portrait                    # portrait (竖屏) | landscape (横屏)
  width: 720                               # 封面图尺寸
  height: 1280
  subtitles: false                         # 是否烧入字幕
  subtitle_style: "FontSize=22,..."        # ASS 字幕样式
  ffmpeg_audio_bitrate: 96k                # 语音 96k 足够
  ffmpeg_video_crf: 28                     # 静态封面可用高 CRF
  ffmpeg_video_framerate: 1                # 静态封面 1fps（字幕模式自动提升到 10fps）
```

</details>

<details>
<summary><b>路径配置</b></summary>

```yaml
paths:
  output_dir: output                       # 输出目录
```

</details>

## 注意事项

> [!CAUTION]
> **PyTorch 版本必须 < 2.4**（已锁定 2.3.x）。PyTorch 2.4+ 改变了 Qwen2 注意力计算的默认行为，会导致语音合成输出乱码。`torch`、`torchaudio`、`transformers`、`onnxruntime` 均已锁定兼容版本，升级前请务必测试音频质量。

- Apple Silicon (M1/M2) 上 0.5B 模型 CPU 推理约需 16GB 内存
- macOS ARM 上 WeTextProcessing 可能存在编译问题，项目使用 `wetext` 替代
- 输出 MP3 或生成视频时依赖 `ffmpeg`；若只输出 WAV，则不需要安装
- `epub` / `mobi` 输入依赖 Calibre 的 `ebook-convert`；若未安装，程序会直接报错提示
