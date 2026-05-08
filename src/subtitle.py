import os
import re

from config import config


_STRIP_EDGE_PUNCT = '，。、；：—'  # 装饰/节奏性标点，字幕帧首尾去掉；？！…… 保留


def _strip_edge_punct(text: str) -> str:
    return text.strip(_STRIP_EDGE_PUNCT)


def split_subtitle_entries(entries):
    """将过长的字幕条目在分句标点处拆分，按字符数比例分配时间。"""
    max_chars_per_line = config['video'].get('subtitle_max_chars_per_line', 18)
    max_lines = config['video'].get('subtitle_max_lines', 2)
    max_chars = max_chars_per_line * max_lines

    result = []
    for start, end, text in entries:
        if len(text) <= max_chars:
            cleaned = _strip_edge_punct(text)
            if cleaned:
                result.append((start, end, cleaned))
            continue
        parts = re.split(r'([，、；：])', text)
        segments = []
        for part in parts:
            if part in '，、；：':
                if segments and len(segments[-1]) + len(part) <= max_chars:
                    segments[-1] += part
                elif segments:
                    # 标点放不进上一段（已满）→ 暂存为新段，与下段文字合并
                    segments.append(part)
                # segments 为空（文本开头的标点）→ 丢弃
                continue
            if segments and len(segments[-1]) + len(part) <= max_chars:
                segments[-1] += part
            else:
                segments.append(part)
        total_chars = sum(len(s) for s in segments)
        duration = end - start
        t = start
        for seg in segments:
            seg_dur = duration * len(seg) / total_chars if total_chars > 0 else 0
            cleaned = _strip_edge_punct(seg)
            if cleaned:
                result.append((t, t + seg_dur, cleaned))
            t += seg_dur
    return result


def _wrap_text(text, max_chars_per_line):
    if len(text) <= max_chars_per_line:
        return text
    lines = []
    for i in range(0, len(text), max_chars_per_line):
        lines.append(text[i:i + max_chars_per_line])
    return '\n'.join(lines)


def format_srt_time(seconds):
    """秒数 → SRT 时间格式 HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f'{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}'


def save_subtitle_file(entries, output_path, clip_index):
    """将 [(start_sec, end_sec, text), ...] 写为 SRT 字幕文件。
    使用临时文件确保原子写入。"""
    if not entries:
        return
    entries = split_subtitle_entries(entries)
    max_chars_per_line = config['video'].get('subtitle_max_chars_per_line', 18)
    max_lines = config['video'].get('subtitle_max_lines', 2)
    orientation = config['video'].get('orientation', 'portrait')
    wrap_width = max_chars_per_line if orientation == 'portrait' else max_chars_per_line * max_lines
    srt_path = f'{output_path}-{clip_index}.srt'
    tmp_path = srt_path.replace('.srt', '.tmp.srt')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        for i, (start, end, text) in enumerate(entries, 1):
            f.write(f'{i}\n')
            f.write(f'{format_srt_time(start)} --> {format_srt_time(end)}\n')
            f.write(f'{_wrap_text(text, wrap_width)}\n\n')
    os.replace(tmp_path, srt_path)
