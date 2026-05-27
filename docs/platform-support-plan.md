# macOS / Windows 兼容性排查与实施方案

## 目标

本方案的目标不是“看起来能跑”，而是把项目收敛到一个可以正式声明的支持范围，并给出实现顺序。

建议先承诺下面这组支持矩阵：

- macOS：Python 3.11 / 3.12
- Windows x64：Python 3.11 / 3.12
- `ffmpeg`：输出 MP3 或 MP4 必需；字幕烧录要求 `subtitles` 滤镜可用
- `ebook-convert`：仅 EPUB / MOBI 输入需要

不建议在改造完成前继续使用“支持 Python 3.13+”或“Windows 已支持”的表述。

## 当前结论

当前代码在“基础路径处理”和“子进程调用方式”上已经有一部分跨平台基础：

- 主流程大量使用 `pathlib.Path`
- `ffmpeg`、`ebook-convert` 调用都是参数列表，不依赖 shell
- WAV / MP3 / MP4 / SRT 都保留了 `.tmp` -> 正式文件的原子写入思路

但静态排查后，项目还不能正式宣称“已支持 Windows”，也不适合把 macOS 支持范围泛化到所有架构。主要原因不是 TTS 主流程，而是文件系统约束、`ffmpeg subtitles` 过滤器转义、以及配置路径刷新不完整。

## 主要问题清单

### 1. 章节标题直接当作输出路径，Windows 会直接失败

当前输出路径来自书名 / 卷名 / 章名的原始文本：

- `src/utility.py` 的 `generate_chapter()`
- `src/utility.py` 的 `construct_text_and_name()`
- `src/utility.py` 的 `cli_main_process()` 中 `OUTPUT_DIR / toc[idx]`

现状问题：

- Windows 非法文件名字符 `< > : " / \ | ? *` 没有处理
- 保留设备名 `CON`、`PRN`、`AUX`、`NUL`、`COM1`... 没有处理
- 结尾空格、结尾句点在 Windows 上会失败
- 过长章节名可能触发路径长度问题
- 不同标题清洗后可能发生路径冲突
- 当前 `toc[idx]` 同时承担“显示文本”和“文件系统路径”两种职责，耦合过深

这是 Windows 支持的首要阻塞项。

### 2. `ffmpeg subtitles` 过滤器的路径转义还不够

当前字幕烧录逻辑位于 `src/video.py` 的 `transform_wav_to_video()`，使用的是手工拼接：

- `subtitles='...':force_style='...':wrap_unicode=1`

当前只处理了：

- 反斜杠
- 冒号
- 单引号

但 Windows 真正棘手的是下面这些组合：

- 盘符路径，如 `C:\...`
- 反斜杠和驱动器冒号同时出现
- 含空格路径
- 含中文路径
- 含 `'`、`,`、`;`、`[`、`]` 等 filtergraph 敏感字符的路径

这部分现在没有专门的辅助函数，也没有测试样例，因此不适合直接宣称“Windows 字幕烧录可用”。

### 3. 仍有多处路径派生依赖字符串拼接或后缀替换

当前存在这类实现：

- `src/video.py`：`f'{os.path.dirname(audio)}/cover.jpg'`
- `src/video.py`：`video_path.replace('.mp4', '.tmp.mp4')`
- `src/utility.py`：`audio_file_path.replace('.wav', '.tmp.wav')`
- `src/utility.py`：`mp3_path.replace('.mp3', '.tmp.mp3')`
- `src/subtitle.py`：`srt_path.replace('.srt', '.tmp.srt')`

这些写法在大多数场景下能跑，但不稳：

- 混用 `/` 和本机路径分隔符
- 依赖字符串内容而不是 `Path` 语义
- 后缀替换容易把目录名或文件名中间的同名片段也替换掉

这不一定只在 Windows 出问题，但 Windows 会把这些边界情况放大。

### 4. 配置路径解析和 CLI 覆盖刷新不完整

当前有两层问题：

1. `src/config.py` 在加载配置时只做了初始路径解析。
2. `src/utility.py` 的 `_apply_config_overrides()` 只刷新了部分模块级常量。

具体缺口：

- `RESOURCES_DIR`
- `MODEL_DIR`
- `SPEAKER_WAV`
- `PROMPT_TEXT`

这些值在 `--set KEY=VALUE` 后不会同步刷新到模块级常量里。也就是说，下面这类用法当前不可靠：

- `--set tts.speaker_wav=...`
- `--set tts.model_dir=...`
- `--set paths.resources_dir=...`

对跨平台来说，这个问题很实际，因为 Windows / macOS 用户更容易使用绝对路径、自定义资源目录和带空格路径。

### 5. Python 版本声明过宽，和依赖现实不一致

`pyproject.toml` 当前是：

- `requires-python = ">=3.11"`

但项目依赖的 `torch==2.3.x` / `torchaudio==2.3.x` 这一组并不适合把支持范围写成 3.13+。如果不先收窄元数据，Windows 和 macOS 用户都会先在安装阶段遇到问题，再误以为是项目本身不可用。

建议先把正式支持范围收敛到：

- `>=3.11,<3.13`

### 6. `ffmpeg` 输出解码方式在 Windows 报错路径上有风险

当前 `src/video.py` 和 `src/utility.py` 里还有这类代码：

- `ret.stdout.decode()`
- `ret.stderr.decode()`

这默认按 UTF-8 解码。Windows 下如果 `ffmpeg` 使用本地代码页输出，错误处理路径本身可能再次触发解码异常或日志乱码。这个问题不影响“成功路径”，但会显著影响排障体验。

### 7. 现有代码已经暴露出一处 Python 版本兼容性告警

执行 `python3 -m compileall src` 时，`src/utility.py` 的 `get_delimiter_pattern()` 会产生 `SyntaxWarning`：

- `"\s"` 是无效转义序列，未来 Python 版本会更严格

这不是 Windows 专属问题，但既然目标支持矩阵包含 Python 3.12，就应该顺手修掉，避免把“平台兼容性”问题和“解释器版本兼容性”问题混在一起。

## 建议的实现方案

### 第一阶段：先收紧支持边界和元数据

先把“声明”修正到和实际能力一致：

1. `pyproject.toml`
   - 把 `requires-python` 改成 `>=3.11,<3.13`
2. `README.md`
   - 把“已验证平台”和“计划支持平台”拆开写
   - 明确 `ffmpeg` / `libass` / `ebook-convert` 的角色
3. 支持口径
   - 先以“macOS + Windows x64，Python 3.11/3.12”为目标
   - macOS Intel 不写入正式支持声明，除非做过真实安装和生成验证

这一阶段成本低，但能立刻避免错误承诺。

### 第二阶段：把“显示标题”和“文件系统路径”拆开

这是核心改造。

建议新增一个专门的路径辅助模块，例如 `src/pathing.py`，至少包含：

- `resolve_runtime_path(value, project_root)`
- `sanitize_path_component(name)`
- `truncate_component(name, max_len)`
- `ensure_unique_component(name, existing_names)`
- `build_output_stem(output_dir, display_parts)`
- `tmp_output_path(path)`

清洗规则建议如下：

- 替换 Windows 非法字符 `< > : " / \ | ? *`
- 去掉控制字符
- 去掉结尾空格和结尾句点
- 规避保留设备名
- 空名称回退到固定占位符，如 `_`
- 对超长组件截断，并追加稳定短哈希避免冲突

结构上不要继续让 `toc[idx]` 同时承担显示和落盘职责，建议改成二选一：

1. `toc[idx]` 保持“显示路径”，另建 `output_targets[idx]`
2. 引入 `ChapterEntry` 数据结构，至少保存：
   - `display_parts`
   - `display_path`
   - `output_parts`
   - `output_stem`

推荐做法是第 2 种。这样：

- 封面图继续使用原始中文标题
- 目录打印继续使用原始中文标题
- 真实落盘只使用清洗后的安全路径

### 第三阶段：统一改成 `Path` 驱动的派生路径

把所有“根据已有文件派生新文件”的逻辑统一收口，不再手写字符串拼接。

建议替换的点：

- `cover.jpg` 路径：改成 `Path(audio).with_name('cover.jpg')`
- 临时文件：统一用 `tmp_output_path(path)`
- `check_export_file_exists()`：输入改成 `Path` stem，再派生 `.wav/.mp3/.mp4/.srt`
- `save_table_of_contents()`：改成 `Path(file_path).parent.mkdir(...)`

这样做的好处：

- 同一套逻辑自动覆盖 macOS / Windows
- 避免目录名中带 `.mp4` / `.wav` 时被误替换
- 后续排查时更容易写单元测试

### 第四阶段：重做字幕烧录的路径转义

`src/video.py` 需要引入专门的 filtergraph 转义函数，例如：

- `build_subtitles_filter(srt_path: Path, subtitle_style: str) -> str`

这个函数的职责必须单一：

- 接收 `Path`
- 先转为绝对路径
- 针对 `ffmpeg subtitles` filter 的语法做转义
- 只负责生成 `-vf` 参数字符串

这一阶段要重点覆盖的路径样例：

- Windows 盘符路径
- 含空格路径
- 含中文路径
- 含单引号路径
- 含方括号 / 逗号 / 分号路径

如果这一层仍然不稳定，不要在命令拼接处继续堆 `replace()`；宁可集中在一个 helper 中收口并测试。

### 第五阶段：统一配置路径解析与覆盖刷新

建议把配置路径解析逻辑做成可重复调用，而不是只在启动时运行一次。

具体做法：

1. `src/config.py`
   - 提供 `normalize_config_paths(cfg)` 一类的 helper
   - 支持相对路径、绝对路径、`~` 展开
2. `src/utility.py`
   - `_apply_config_overrides()` 在处理完 `--output-dir` 和 `--set` 后，重新调用路径规范化
   - 同步刷新所有依赖配置的模块级常量

至少要刷新这些全局值：

- `RESOURCES_DIR`
- `OUTPUT_DIR`
- `MODEL_DIR`
- `SPEAKER_WAV`
- `PROMPT_TEXT`
- `SPEED`
- `INTER_SENTENCE_SILENCE_MS`
- `MAX_CHARS_PER_CLIP`

否则用户在 macOS / Windows 下传入自定义路径时，CLI 看起来接受了配置，实际运行仍然用旧值。

### 第六阶段：补最小可执行测试面

当前没有自动化测试，正式支持 Windows 之前至少要补下面三类验证。

#### A. 纯逻辑测试

建议新增 `tests/` 并优先覆盖：

- 路径清洗
- 路径冲突去重
- `tmp` 文件派生
- 配置路径规范化
- `ffmpeg subtitles` filter 字符串生成

这类测试不依赖模型，成本最低。

#### B. 轻量 CLI 测试

至少保留以下命令：

```bash
uv run txt2audio --dump-config --json
python -m compileall src
```

如果补一个不加载模型的 `--dry-run` / `--validate-paths` 模式，Windows 和 macOS CI 会容易很多，因为可以验证：

- 输入解析
- 路径落盘
- 配置覆盖
- 目录生成

而不需要真的跑完整 TTS。

#### C. 真机冒烟测试

支持声明前至少做下面两组人工验证：

1. macOS
   - `wav`
   - `mp3`
   - `mp4`
   - `mp4 + subtitles`
2. Windows
   - `wav`
   - `mp3`
   - `mp4`
   - `mp4 + subtitles`
   - 输出目录为带空格、带中文、较长路径
   - 章节标题包含 `: ? * | [ ] '`

## 推荐的实施顺序

建议按下面顺序推进，避免一开始就大改主流程：

1. 收紧 `requires-python`，修正文档支持边界
2. 引入 `pathing` 辅助模块
3. 拆分“显示标题”和“文件系统路径”
4. 把临时文件和封面路径全部改成 `Path`
5. 重做 `ffmpeg subtitles` filter 转义
6. 统一配置路径规范化和覆盖刷新
7. 补路径 / 配置 / filter helper 测试
8. 做 macOS / Windows 真机冒烟验证

这样改动面是从外围到核心，风险最低。

## 验收标准

当下面条件全部满足时，才建议在 README 和项目描述里正式写“支持 macOS / Windows”：

- `uv sync` 能在目标平台安装成功
- `txt2audio --dump-config --json` 正常输出有效配置
- `--output-dir` 和 `--set` 传入绝对路径、相对路径、带空格路径都能生效
- 章节名包含 Windows 非法字符时不会崩溃，且能稳定落盘
- 章节名清洗后不会产生静默覆盖
- `wav`、`mp3`、`mp4`、`mp4 + subtitles` 都能成功生成
- `ffmpeg` 失败时日志不会因为编码问题再次报错
- 原子写入、断点续跑、字幕时间戳、视频成功后清理中间文件等现有约束全部保持不变

## 结论

这次兼容性改造的真正重点不是 TTS 模型，而是“路径系统”和“外部工具接口”。

如果只改 README 或只补几个 `Path(...)`，Windows 支持依然不可靠。要把这件事做扎实，至少需要完成三件事：

1. 输出路径清洗与去耦
2. `ffmpeg subtitles` 路径转义收口
3. 配置路径规范化与覆盖刷新统一

这三项完成后，项目才具备进入 macOS / Windows 联合验证阶段的条件。
