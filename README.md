# txt2audio

> 将网络小说转换成有声书（主要是为了保护我的眼睛）

将 `.txt` 格式的中文小说转换为有声书。使用 [Fun-CosyVoice3-0.5B](https://github.com/FunAudioLLM/CosyVoice) 进行零样本语音合成（zero-shot TTS），支持生成带封面的 MP4 视频。

## 功能特性

- 自动识别卷/章结构（支持序、楔子、后记等特殊章节）
- 自动检测文件编码
- 多音字智能标注 —— 使用 `pypinyin` 根据上下文为多音字（如 的/得/地、为/wéi/wèi）添加拼音标注，提升语音合成准确度
- 断点续生成 —— 原子写入（先写临时文件再重命名），中断不会产生损坏文件，自动清理残留
- 每个音频片段约 30 分钟（6300 汉字）
- 可选生成 MP4 视频（带自动生成的封面图）
- 支持横屏/竖屏视频（`--landscape` 或 `config.yaml` 中设置）
- 可选内嵌字幕 —— 自动生成句级 SRT 字幕并烧入视频
- 声音克隆 —— 提供一段参考音频即可用该声音朗读全书

## 环境要求

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) — Python 包管理器
- `ffmpeg` — 音频转 MP3 及生成视频均需要

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

如果已经克隆但没有拉取子模块：

```bash
git submodule update --init --recursive
```

## 使用方法

```bash
# 纯音频（默认）
uv run python src/transform_to_audio.py your_book.txt

# 音频 + 视频（生成 MP4，完成后自动删除中间 .wav 文件）
uv run python src/transform_to_audio.py your_book.txt --video

# 指定章节范围（跳过交互式提示）
uv run python src/transform_to_audio.py your_book.txt --range 0~8    # 第 0 到第 8 章
uv run python src/transform_to_audio.py your_book.txt --range 5      # 仅第 5 章
uv run python src/transform_to_audio.py your_book.txt --range all    # 全部章节

# 横屏视频
uv run python src/transform_to_audio.py your_book.txt --video --landscape

# 组合使用
uv run python src/transform_to_audio.py your_book.txt --video --range all
```

运行后会交互式提示选择章节范围（使用 `--range` 可跳过）。

## 输出结构

```
output/{书名}/{卷名}/{章名}-{片段}.mp3    # 默认（纯音频，MP3 128kbps）
output/{书名}/{卷名}/{章名}-{片段}.mp4    # 使用 --video 时
output/{书名}/{卷名}/{章名}-{片段}.srt    # 中间字幕文件（烧入视频后自动删除）
output/{书名}/目录.txt                    # 自动生成的目录
```

## 项目结构

```
src/
  transform_to_audio.py   # 入口
  utility.py              # 文本解析、音频生成、CLI 逻辑
  video.py                # 封面图生成、ffmpeg 转视频
  subtitle.py             # SRT 字幕生成
  config.py               # 配置加载
config.yaml               # 配置文件（TTS、音频、视频、路径等参数）
resources/                # 字体文件、参考音频
third_party/CosyVoice/   # CosyVoice 子模块
pretrained_models/        # 模型权重（gitignored）
```

## 配置

编辑项目根目录的 `config.yaml` 即可自定义参数，无需改代码：

```yaml
tts:
  speaker_wav: resources/my_speaker.wav    # 换声音：替换参考音频文件
  prompt_text: "..."                       # 零样本 TTS 提示文本
  model_dir: pretrained_models/Fun-CosyVoice3-0.5B

audio:
  max_chars_per_clip: 6300                 # 每个音频片段的汉字上限（-1 = 不分片）
  model_sentence_limit: 2000              # TTS 模型单次输入的字符上限
  output_format: mp3                       # 纯音频输出格式：mp3 或 wav
  mp3_bitrate: 128k                        # MP3 码率

video:
  orientation: portrait                    # portrait (竖屏) | landscape (横屏)
  width: 720                               # 封面图尺寸
  height: 1280
  subtitles: false                         # 是否烧入字幕
  subtitle_style: "FontSize=22,..."        # ASS 字幕样式
  ffmpeg_audio_bitrate: 96k                # 语音 96k 足够
  ffmpeg_video_crf: 28                     # 静态封面可用高 CRF
  ffmpeg_video_framerate: 1                # 静态封面 1fps（字幕模式自动提升到 10fps）

paths:
  output_dir: output                       # 输出目录
```

所有路径均相对于项目根目录。完整配置项及说明见 `config.yaml`。

### 自定义声音（声音克隆）

准备一段清晰、无噪音的参考音频（建议 5~15 秒），放入 `resources/` 目录，然后修改 `config.yaml`：

```yaml
tts:
  speaker_wav: resources/my_speaker.wav          # 替换为你的参考音频
  prompt_text: "You are a helpful assistant.<|endofprompt|>参考音频中说的原文。"
```

> **注意：** `<|endofprompt|>` 后面的文本必须与参考音频内容**逐字一致**，否则音色提取会失败。

## 注意事项

- **PyTorch 版本必须 < 2.4**（已锁定 2.3.x）。PyTorch 2.4+ 改变了 Qwen2 注意力计算的默认行为，会导致语音合成输出乱码。`torch`、`torchaudio`、`transformers`、`onnxruntime` 均已锁定兼容版本，升级前请务必测试音频质量。
- Apple Silicon (M1/M2) 上 0.5B 模型 CPU 推理约需 16GB 内存。
- macOS ARM 上 WeTextProcessing 可能存在编译问题，项目使用 `wetext` 替代。
