from pathlib import Path, PurePath


_FILTER_OPTION_ESCAPE_CHARS = frozenset("\\':")
_FILTERGRAPH_ESCAPE_CHARS = frozenset("\\'[],;")


def _escape_ffmpeg_value(text: str, escape_chars) -> str:
    escaped = []
    for char in text:
        if char in escape_chars:
            escaped.append('\\')
        escaped.append(char)
    return ''.join(escaped)


def normalize_subtitles_path(srt_path: Path | PurePath) -> str:
    if isinstance(srt_path, Path):
        srt_path = srt_path.resolve()
    return srt_path.as_posix()


def escape_subtitles_filter_value(value: str) -> str:
    option_escaped = _escape_ffmpeg_value(value, _FILTER_OPTION_ESCAPE_CHARS)
    return _escape_ffmpeg_value(option_escaped, _FILTERGRAPH_ESCAPE_CHARS)


def build_subtitles_filter(srt_path: Path | PurePath, subtitle_style: str) -> str:
    escaped_srt_path = escape_subtitles_filter_value(normalize_subtitles_path(srt_path))
    escaped_style = escape_subtitles_filter_value(subtitle_style)
    return f"subtitles=filename={escaped_srt_path}:force_style={escaped_style}:wrap_unicode=1"
