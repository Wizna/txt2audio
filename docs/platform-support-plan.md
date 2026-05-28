# macOS / Windows 兼容性排查与实施方案

## 当前状态

当前代码层面的跨平台改造已经完成主要收口，项目现处于“macOS 媒体链路已验证，等待 Windows 真机验证”的阶段。

已经完成的能力包括：

- 收紧正式支持边界到 Python 3.11 / 3.12
- 将显示标题和文件系统输出路径解耦
- 对 Windows 非法文件名、保留设备名、尾随空格/句点进行路径组件清洗
- 将临时文件、封面图、字幕和媒体派生路径统一到 `Path` 驱动
- 为 `ffmpeg subtitles` 过滤器增加专门的路径转义 helper
- 在 CLI 中加入轻量 `--validate-paths` 校验模式，避免为了做路径验证而加载 TTS 模型
- 为路径清洗、字幕过滤器转义、轻量校验模式补充基础测试

当前还没有完成的是正式支持声明前必须要做的 Windows 真机冒烟验证，尤其是带盘符路径的媒体输出与字幕烧录。

## 支持边界

当前建议维持下面这组支持矩阵：

- macOS：Python 3.11 / 3.12
- Windows x64：Python 3.11 / 3.12
- `ffmpeg`：输出 MP3 或 MP4 必需；字幕烧录要求 `subtitles` 滤镜可用
- `ebook-convert`：仅 EPUB / MOBI 输入需要

在 Windows 的真实媒体 smoke test 完成前，不建议把 README 或对外说明改成“Windows 已正式支持”。

## 已完成改造

### 1. 输出路径安全化

已新增 `src/pathing.py`，把章节显示文本和落盘路径拆开处理。

结果：

- 原始目录文本仍可用于展示和日志
- 实际输出文件名会做 Windows 兼容清洗
- 清洗后发生冲突时，会追加稳定后缀避免静默覆盖

这解决了 Windows 对 `< > : " / \ | ? *`、保留设备名、尾随空格/句点等文件名限制。

### 2. 派生文件统一改为 `Path` 处理

`wav`、`mp3`、`mp4`、`srt`、封面图以及 `.tmp` 临时文件的路径派生逻辑已统一改成 `Path` 语义，不再依赖字符串替换或手工拼接目录分隔符。

这一点直接降低了跨平台路径边界问题，并保留了原有 `.tmp` -> 正式文件 的原子写入约束。

### 3. 配置路径规范化与运行时刷新

配置路径现在支持在 CLI 覆盖后重新规范化，并刷新运行时常量。

已经覆盖的场景包括：

- `--output-dir`
- `--set KEY=VALUE`
- 资源目录、模型目录、speaker wav、prompt 文件等路径型配置

这使 macOS 和 Windows 用户传入绝对路径、相对路径、带空格路径时，运行时行为与 CLI 输入保持一致。

### 4. 字幕烧录路径转义收口

已新增 `src/ffmpeg_filters.py` 并在 `src/video.py` 中统一使用，用来构造 `ffmpeg subtitles` 滤镜字符串。

当前已覆盖的转义关注点包括：

- Windows 盘符路径
- 空格路径
- 引号
- 方括号
- 逗号
- 分号
- 前后斜杠规范化

这里的单元测试已经补上，但仍缺少 Windows 真机 `ffmpeg/libass` 运行验证。

### 5. 轻量验证模式

CLI 新增 `--validate-paths` 模式，用于在不加载 TTS 模型的情况下验证：

- 输入解析
- 章节输出路径规划
- CLI 覆盖后的配置路径
- 目标输出目录

这一步的实现同时把部分重依赖改成了延迟导入，使路径校验、文档排查、CI 级别的基础验证更快、更稳定。

## 已完成验证

当前工作区已经完成以下验证：

- `python3 -m compileall src tests`
- `python3 -m unittest discover -s tests -p 'test_*.py'`
- `uv run txt2audio --dump-config --json`
- `uv run txt2audio <sample>.txt --validate-paths --json`
- 使用《亵渎》仅第 `0` 章（引言）完成 macOS Apple Silicon 真实媒体验证
- 复用同一章已生成的 `wav + srt`，单独验证 `wav -> mp3`、`wav -> mp4`、`wav + srt -> mp4`

这些验证已经证明：

- 新增 helper 和 CLI 入口可通过语法与单元测试
- 运行时配置解析和路径规划可以独立工作
- 不必加载模型也能做跨平台路径规则验证
- macOS 当前环境下的 `wav`、`mp3`、`mp4`、`mp4 + subtitles` 链路可用
- 字幕视频在 macOS 当前 `ffmpeg + libass` 环境下可完成 `tmp.mp4 -> mp4` 提交并清理 `wav/srt`

这些验证还不能证明：

- Windows 当前环境下的真实媒体生成稳定可用
- Windows 上字幕烧录在真实盘符路径和本地 `ffmpeg` 环境中没有差异

## 下一步

下一步不是继续扩写路径逻辑，而是做 Windows 真机媒体链路的 smoke test。

macOS 已经完成验证，当前最小剩余验证矩阵如下：

1. `wav`
2. `mp3`
3. `mp4`
4. `mp4 + subtitles`

推荐命令：

```bash
uv run txt2audio <sample>.txt --range all --output-format wav --json
uv run txt2audio <sample>.txt --range all --output-format mp3 --json
uv run txt2audio <sample>.txt --video --range all --json
uv run txt2audio <sample>.txt --video --range all --set video.subtitles=true --json
```

Windows 额外需要覆盖下面这些边界：

- 输出目录包含空格
- 输出目录包含中文
- 绝对路径为盘符路径，如 `C:\work\demo output`
- 章节标题包含 `: ? * | [ ] '`

如果上述 smoke test 全部通过，再把 README 中的 Windows 支持表述提升为正式支持。

## 验收标准

只有下面条件都满足时，才建议正式声明“支持 macOS / Windows”：

- `uv sync` 能在目标平台安装成功
- `txt2audio --dump-config --json` 正常输出有效配置
- `--output-dir` 和 `--set` 传入绝对路径、相对路径、带空格路径都能正确生效
- 章节名包含 Windows 非法字符时不会崩溃，且能稳定落盘
- `wav`、`mp3`、`mp4`、`mp4 + subtitles` 都能成功生成
- Windows 上字幕烧录可在真实盘符路径下工作
- 原子写入、断点续跑、字幕时间戳、视频成功后清理中间文件等现有行为保持不变

## 结论

当前阶段的主要风险已经不再是代码设计，而是 Windows 目标平台上的真实运行结果。

因此下一步应聚焦：

1. Windows 媒体输出 smoke test
2. Windows 字幕烧录 smoke test
3. 根据 Windows 结果补最后一轮 README 与支持矩阵说明
