import os


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
    srt_path = f'{output_path}-{clip_index}.srt'
    tmp_path = srt_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        for i, (start, end, text) in enumerate(entries, 1):
            f.write(f'{i}\n')
            f.write(f'{format_srt_time(start)} --> {format_srt_time(end)}\n')
            f.write(f'{text}\n\n')
    os.rename(tmp_path, srt_path)
