# txt2audio 项目优化方案

更新时间：2026-05-31

## 背景

本方案基于当前代码、`AGENTS.md`、`CLAUDE.md`、`README.md`、`config.yaml`、`docs/platform-support-plan.md` 和现有测试整理。目标不是大规模重构，而是优先修复会影响结果可信度、断点续跑和 agent 集成体验的问题。

需要特别保留的现有约束：

- 继续优先使用 `uv run txt2audio` 作为 CLI 验证入口。
- `torch` / `torchaudio` / `transformers` / `onnxruntime` 版本栈不能随意升级，避免 CosyVoice3 中文输出质量退化。
- `annotate_polyphones()` 必须在 `mask_punctuations()` 前运行。
- WAV、MP3、MP4、SRT 等输出继续使用 `.tmp` 临时文件再原子替换。
- `check_export_file_exists()` 的断点续跑语义不能破坏。
- 章节识别逻辑先补回归样本，再考虑小步扩展；不要直接用宽泛正则替换现有行为。

## 当前结论

当前项目主流程已经具备可用的 CLI、配置覆盖、路径安全化、字幕烧录和基础单元测试。最值得优先优化的不是新增大功能，而是让失败状态更准确、让副产物更可控、让文档和代码保持一致。

已经确认：

- CosyVoice frontend 默认会先尝试 `ttsfrd`，失败后回落到 `wetext`。项目 README 中提到 `wetext` 有依据，但可以补充“由 CosyVoice frontend 负责”的表述。
- 章节识别当前按中文数字和 `卷/章` 分隔符工作，属于项目既有设计边界。扩展识别能力需要测试先行。
- 当前已有 `tests/`，不再是“没有正式测试框架”的状态。

## P0：结果可信度和错误收口

### 1. 媒体转换失败必须影响退出码

现状：

- MP3 转换失败时 `convert_wav_to_mp3()` 只记录 warning 并返回原 WAV。
- MP4 转换失败时 `transform_wav_to_video()` 只记录 error，主流程仍可能返回成功。
- JSON 模式可能输出 `"status": "success"`，但目标 MP3/MP4 实际没有生成。

方案：

- 让 MP3/MP4 转换函数返回明确结果对象，或在失败时抛出可捕获异常。
- CLI 汇总转换失败数量；如果有失败，JSON 输出 `status: "error"` 或 `status: "partial_error"`，退出码返回 `1`。
- 人类模式输出失败文件路径和保留的中间文件路径。

验收：

- 模拟 ffmpeg 失败时，CLI 退出码为 `1`。
- JSON 输出包含失败文件和错误摘要。
- 成功路径仍保持 `.tmp` 原子替换和中间文件清理。

### 2. `--range` 输入错误结构化返回

现状：

- 非法 range 会抛出未捕获异常。
- `8~0` 这类反向范围会变成空任务并成功退出。

方案：

- 校验 range 只能是 `all`、单个非负整数、或 `start~end` / `start-end`。
- 校验 `start <= end`，并给出清晰错误。
- 超出章节上限时明确提示，而不是静默生成空列表。

验收：

- `--json --range abc` 返回结构化错误，退出码 `1`。
- `--range 8~0` 返回错误。
- 现有 `all`、`5`、`0~8` 行为不变。

### 3. SRT 副产物生成策略收口

现状：

- `generate_audio_clip()` 在纯音频模式也会生成 SRT。
- 视频成功后会删除 SRT，但纯 MP3/WAV 模式会留下字幕文件。

方案：

- 增加内部参数控制是否生成 SRT。
- 默认只在 `--video` 且 `video.subtitles=true` 时生成 SRT。
- 如果希望保留独立字幕，另设显式开关，例如未来的 `--srt` 或 `--keep-srt`。

验收：

- 默认 MP3/WAV 运行不生成 `.srt`。
- 视频字幕模式仍能生成并在 MP4 成功后清理 `.srt`。
- 视频失败时保留 `.srt` 便于复用或排查。

## P1：测试基线和文档一致性

### 4. 为章节识别建立回归样本

现状：

- 章节识别规则较敏感，直接扩展正则容易破坏现有正常输入。
- 当前没有针对 `construct_text_and_name()` 的测试。

方案：

- 先新增测试，锁住现有支持格式：
  - `第一卷` / `第一章`
  - 序、序章、楔子、后记、终章
  - 引言内容落到书目录内的 `引言`
  - 正文中重复出现当前章节名时不误开新章
  - 输出显示路径和落盘路径分离
- 再把新增需求做成单独待办，不和基线测试混在一起。

验收：

- 不改解析规则时，新增基线测试先通过。
- 后续任何章节识别改动都必须先跑这些测试。

### 5. 清理 `CLAUDE.md` / README 过期描述

现状：

- `CLAUDE.md` 写了 `split_long_sentences()`，当前代码没有该函数。
- `CLAUDE.md` 写多音字覆盖“的/得/地、为”等，当前 `_POLYPHONE_CHARS` 只有 `校`。
- `CLAUDE.md` 写“没有正式测试框架”，但现在已有 unittest。

方案：

- 文档改成当前真实状态。
- 对 `wetext` 的描述改为“由 CosyVoice frontend 处理，优先 `ttsfrd`，否则回落到 `wetext`，都不可用时退化为基础清理”。
- 多音字描述只写当前已启用字符，并把扩展多音字作为后续优化。

验收：

- README 和 CLAUDE 不再承诺代码里不存在的函数或覆盖范围。
- agent 使用文档和真实命令一致。

### 6. 补核心纯函数测试

建议覆盖：

- `parse_range_string()`
- `construct_text_and_name()`
- `split_subtitle_entries()`
- `format_srt_time()`
- `mask_punctuations()`
- `annotate_polyphones()` 的最小行为和顺序约束

验收：

- `uv run python -m unittest discover -s tests -p 'test_*.py'` 覆盖这些纯函数。
- 不需要加载 TTS 模型即可运行。

## P2：受控增强

### 7. 文本预处理策略显式化

方向：

- 保留项目自己的小说清洗层。
- 明确哪些事情交给 CosyVoice frontend，例如 `wetext` / `ttsfrd` 文本归一化和内部分段。
- 如果以后引入自己的长句拆分，先确认不会破坏 CosyVoice 内部 `split_paragraph` 的韵律效果。

验收：

- 配置里的 `model_sentence_limit` 要么被真实读取，要么继续明确标注为说明项。
- 不新增和 CosyVoice frontend 重复且难验证的清洗逻辑。

### 8. 多音字标注扩展

方向：

- 只加入真实听感中已经听到读错、且 CosyVoice token 支持稳定的字。
- 不为了扩大覆盖面而添加多音字；语言模型本身能读对一部分多音字，过度标注反而可能影响自然度。
- 每个字都配小样本文本和预期 token 行为。
- 保持“少标注”原则，避免影响自然度。

验收：

- 扩展前后抽样试听中文输出。
- 单元测试只验证 token 合法性和目标字是否标注，不把 pypinyin 全部行为写死。

### 9. 可选运行计划输出

方向：

- 增加 dry-run / plan JSON，输出章节数量、目标路径、预计输出格式、是否会跳过已存在文件。
- 与 `--validate-paths` 区分：`--validate-paths` 只关心解析和路径；plan 模式还关心恢复状态和输出动作。

验收：

- 不加载 TTS 模型。
- 适合 agent 在生成前确认任务范围。

## P3：产品体验增强

候选能力：

- 独立 SRT 导出开关。
- M4B 或章节清单导出。
- 封面长标题自动换行、缩放或截断。
- 更完整的 Windows 真机 smoke test 记录。
- 运行摘要包含成功、跳过、失败文件列表。

这些不应抢在 P0 之前做。

## 建议执行顺序

1. 修媒体转换失败返回和 CLI 汇总。
2. 修 `--range` 错误收口。
3. 修 SRT 生成策略。
4. 补章节识别基线测试，不扩展规则。
5. 清理 CLAUDE/README 过期描述。
6. 补 subtitle、range、mask/polyphone 的纯函数测试。
7. 再评估章节识别、小范围多音字、dry-run plan 等增强项。

## 当前验证

已在当前工作区执行：

- `uv run python -m compileall src tests`
- `uv run python -m unittest discover -s tests -p 'test_*.py'`

结果：语法检查通过，9 个 unittest 通过。
