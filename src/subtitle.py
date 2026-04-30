import os
import re

from config import config


def split_subtitle_entries(entries):
    """将过长的字幕条目在分句标点处拆分，按字符数比例分配时间。"""
    max_chars_per_line = config['video'].get('subtitle_max_chars_per_line', 18)
    max_lines = config['video'].get('subtitle_max_lines', 2)
    max_chars = max_chars_per_line * max_lines

    result = []
    for start, end, text in entries:
        if len(text) <= max_chars:
            result.append((start, end, text))
            continue
        parts = re.split(r'([，、；：])', text)
        segments = []
        buf = []
        for part in parts:
            buf.append(part)
            if part in '，、；：':
                continue
            joined = ''.join(buf)
            if segments and len(segments[-1]) + len(joined) <= max_chars:
                segments[-1] += joined
            else:
                segments.append(joined)
            buf = []
        if buf:
            tail = ''.join(buf)
            if segments and len(segments[-1]) + len(tail) <= max_chars:
                segments[-1] += tail
            else:
                segments.append(tail)
        if not segments:
            result.append((start, end, text))
            continue
        total_chars = sum(len(s) for s in segments)
        duration = end - start
        t = start
        for seg in segments:
            seg_dur = duration * len(seg) / total_chars if total_chars > 0 else 0
            result.append((t, t + seg_dur, seg))
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
    os.rename(tmp_path, srt_path)
