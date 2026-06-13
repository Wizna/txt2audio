# 文本处理职责边界

更新时间：2026-05-31

本文记录 txt2audio 和 CosyVoice frontend 在文本处理链路中的职责边界，避免后续重复实现归一化、分句或过度拼音标注。

## 总原则

txt2audio 只做小说输入和有声书输出所需的轻量预处理。语言级文本归一化和细粒度 TTS 分段继续交给 CosyVoice frontend。

这能减少两类风险：

- 在 txt2audio 里重复做数字、英文、符号归一化，和 CosyVoice 的 `ttsfrd` / `wetext` 行为冲突。
- 在进入 TTS 前过度拆句或过度标注拼音，破坏 CosyVoice 内部 `split_paragraph` 的韵律处理。

## 当前处理层次

### 1. 输入文件层

负责人：txt2audio

实现位置：

- `load_book_file()`
- `load_txt_file()`
- `convert_book_to_txt()`

职责：

- 支持 `.txt`、`.epub`、`.mobi` 输入。
- 自动检测文本编码。
- 对 EPUB/MOBI 调用 Calibre `ebook-convert`，生成可复用的 `.txt2audio.txt`。

不负责：

- 修改书籍正文语义。
- 做语言级数字/单位/英文读法归一化。

### 2. 章节解析层

负责人：txt2audio

实现位置：

- `construct_text_and_name()`
- `generate_chapter_parts()`
- `check_special_delimiter()`

职责：

- 按当前 `book_delimiter` 解析卷/章结构。
- 处理当前支持的特殊段落：序、序章、序言、前言、楔子、引言、后记、终章。
- 生成显示目录和文件系统安全输出路径。

约束：

- 不要在没有回归样本的情况下扩大章节识别正则。
- 显示路径和落盘路径必须继续分离。

### 3. 小说标点清洗层

负责人：txt2audio

实现位置：

- `mask_punctuations()`

职责：

- 清理网络小说中常见的装饰标点、URL、书名号和重复标点。
- 将破折号、省略号等转换为更适合朗读的标点。
- 确保中文句子进入 TTS 前有句末标点。

不负责：

- 数字、金额、日期、英文缩写等语言级归一化。
- 根据 TTS token 长度做最终分段。

### 4. 多音字保守标注层

负责人：txt2audio

实现位置：

- `annotate_polyphones()`

职责：

- 只对当前确认高价值且 CosyVoice token 支持稳定的多音字做拼音标注。
- 当前启用集合聚焦于 `校`。
- 在 `mask_punctuations()` 之前运行，避免清洗后字符索引漂移。

不负责：

- 为所有常见多音字做泛化标注。
- 覆盖没有试听验证的读音规则。

### 5. CosyVoice frontend 层

负责人：CosyVoice

实现位置：

- `third_party/CosyVoice/cosyvoice/cli/frontend.py`
- `third_party/CosyVoice/cosyvoice/cli/cosyvoice.py`

职责：

- 优先使用 `ttsfrd`，失败后回落到 `wetext`。
- 对数字、符号、英文等做语言级文本归一化。
- 通过内部 `split_paragraph` 做 TTS 友好的细粒度分段。

txt2audio 调用约束：

- 默认保留 `inference_zero_shot(..., text_frontend=True)`。
- 不在 txt2audio 中增加和 CosyVoice frontend 重复的归一化层，除非有明确的音频质量回归样本。

## `model_sentence_limit` 说明

`audio.model_sentence_limit` 当前是说明项，不直接参与运行时逻辑。实际细粒度拆分由 CosyVoice frontend 的 `split_paragraph` 负责。

如果未来要让该配置生效，需要先完成：

- 证明 CosyVoice 内部分段不能覆盖目标场景。
- 增加不会破坏标点、字幕时间戳和断点续跑的测试。
- 抽样试听长句和普通句的输出质量。

## 后续改动准则

新增文本处理逻辑时，先回答：

1. 这是小说输入清洗，还是语言级归一化？
2. CosyVoice frontend 是否已经处理？
3. 是否会改变 `annotate_polyphones()` 和 `mask_punctuations()` 的顺序？
4. 是否会影响字幕时间戳和句子边界？
5. 是否有模型无关测试和必要的音频抽样验证？

只有这些问题都有明确答案后，再改代码。
