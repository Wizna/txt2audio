# txt2audio

> 将网络小说转换成有声书（主要是为了保护我的眼睛）

将 `.txt` 格式的中文小说转换为有声书。使用 [Fun-CosyVoice3-0.5B](https://github.com/FunAudioLLM/CosyVoice) 进行零样本语音合成（zero-shot TTS），支持生成带封面的 MP4 视频。

## 功能特性

- 自动识别卷/章结构（支持序、楔子、后记等特殊章节）
- 自动检测文件编码
- 多音字智能标注 —— 使用 `pypinyin` 根据上下文为多音字（如 的/得/地、为/wéi/wèi）添加拼音标注，提升语音合成准确度
- 断点续生成 —— 跳过已有的音频/视频文件
- 每个音频片段约 30 分钟（6300 汉字）
- 可选生成 MP4 视频（带自动生成的封面图）

## 环境要求

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) — Python 包管理器
- `ffmpeg` — 生成视频时需要

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
uv run python src/transform_to_audio.py demo/《英雄志》（校对第1-22卷）作者：孙晓.txt

# 音频 + 视频（生成 MP4，完成后自动删除中间 .wav 文件）
uv run python src/transform_to_audio.py demo/《英雄志》（校对第1-22卷）作者：孙晓.txt --video

# 指定章节范围（跳过交互式提示）
uv run python src/transform_to_audio.py your_book.txt --range 0~8    # 第 0 到第 8 章
uv run python src/transform_to_audio.py your_book.txt --range 5      # 仅第 5 章
uv run python src/transform_to_audio.py your_book.txt --range all    # 全部章节

# 组合使用
uv run python src/transform_to_audio.py your_book.txt --video --range all
```

运行后会交互式提示选择章节范围（使用 `--range` 可跳过）。

## 输出结构

```
output/{书名}/{卷名}/{章名}-{片段}.wav    # 默认（纯音频）
output/{书名}/{卷名}/{章名}-{片段}.mp4    # 使用 --video 时
output/{书名}/目录.txt                    # 自动生成的目录
```

## 项目结构

```
src/
  transform_to_audio.py   # 入口
  utility.py              # 文本解析、音频生成、CLI 逻辑
  video.py                # 封面图生成、ffmpeg 转视频
  config.py               # 配置加载
config.yaml               # 配置文件（TTS、音频、视频、路径等参数）
resources/                # 字体文件、参考音频
third_party/CosyVoice/   # CosyVoice 子模块
pretrained_models/        # 模型权重（gitignored）
demo/                     # 示例文本文件
```

## 配置

编辑项目根目录的 `config.yaml` 即可自定义参数，无需改代码：

```yaml
tts:
  speaker_wav: resources/my_speaker.wav    # 换声音：替换参考音频文件
  prompt_text: "..."                       # 零样本 TTS 提示文本
  model_dir: pretrained_models/Fun-CosyVoice3-0.5B

audio:
  chinese_word_limit_half_hour: 6300       # 每个音频片段的汉字上限（约 30 分钟）
  model_sentence_limit: 200               # TTS 模型单次输入的字符上限

video:
  width: 720                               # 封面图尺寸
  height: 1280
  ffmpeg_audio_bitrate: 192k               # FFmpeg 编码参数

paths:
  output_dir: output                       # 输出目录
```

所有路径均相对于项目根目录。完整配置项及说明见 `config.yaml`。

## 注意事项

- **PyTorch 版本必须 < 2.4**（已锁定 2.3.x）。PyTorch 2.4+ 改变了 Qwen2 注意力计算的默认行为，会导致语音合成输出乱码。`torch`、`torchaudio`、`transformers`、`onnxruntime` 均已锁定兼容版本，升级前请务必测试音频质量。
- Apple Silicon (M1/M2) 上 0.5B 模型 CPU 推理约需 16GB 内存。
- macOS ARM 上 WeTextProcessing 可能存在编译问题，项目使用 `wetext` 替代。
