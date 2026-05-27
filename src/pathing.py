from __future__ import annotations

import re
from hashlib import sha1
from pathlib import Path


_WINDOWS_RESERVED_NAMES = {
    'CON', 'PRN', 'AUX', 'NUL',
    'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
    'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9',
}
_WINDOWS_INVALID_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_TRAILING_CHARS_RE = re.compile(r'[ .]+$')


def resolve_runtime_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def truncate_component(name: str, max_length: int = 120) -> str:
    if len(name) <= max_length:
        return name
    digest = sha1(name.encode('utf-8')).hexdigest()[:8]
    keep = max(1, max_length - len(digest) - 1)
    return f'{name[:keep]}-{digest}'


def sanitize_path_component(name: str, max_length: int = 120) -> str:
    cleaned = _WINDOWS_INVALID_CHARS_RE.sub('_', name).strip()
    cleaned = _WINDOWS_TRAILING_CHARS_RE.sub('', cleaned)
    if not cleaned:
        cleaned = '_'

    reserved_key = cleaned.split('.')[0].upper()
    if reserved_key in _WINDOWS_RESERVED_NAMES:
        cleaned = f'_{cleaned}'

    return truncate_component(cleaned, max_length=max_length)


def ensure_unique_path(path: Path, used_paths: set[Path], max_length: int = 120) -> Path:
    candidate = path
    counter = 2
    while candidate in used_paths:
        candidate = path.with_name(truncate_component(f'{path.name}-{counter}', max_length=max_length))
        counter += 1
    used_paths.add(candidate)
    return candidate


def build_display_path(parts: list[str]) -> str:
    return '/'.join(parts)


def build_output_relpath(parts: list[str], used_paths: set[Path], max_length: int = 120) -> Path:
    sanitized_parts = [sanitize_path_component(part, max_length=max_length) for part in parts]
    return ensure_unique_path(Path(*sanitized_parts), used_paths, max_length=max_length)


def tmp_output_path(path: Path) -> Path:
    return path.with_name(f'{path.stem}.tmp{path.suffix}')


def build_clip_output_path(output_stem: Path, clip_index: int, suffix: str) -> Path:
    return output_stem.parent / f'{output_stem.name}-{clip_index}{suffix}'
