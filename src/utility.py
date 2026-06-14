import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from typing import List, Dict
from dataclasses import dataclass
import sys
import subprocess as _subprocess
import argparse
import time
import logging
import json
import shutil
import tempfile

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.panel import Panel

from charset_normalizer import from_path
import re
from pathlib import Path
import math
from config import config, PROJECT_ROOT, normalize_config_paths
from pathing import build_clip_output_path, build_display_path, build_output_relpath, tmp_output_path
from validation import build_path_validation_entries

logger = logging.getLogger('txt2audio')
console = Console(stderr=True)

RESOURCES_DIR = config['paths']['resources_dir']
OUTPUT_DIR = config['paths']['output_dir']

# CosyVoice submodule needs to be on sys.path for its internal imports
sys.path.insert(0, str(PROJECT_ROOT / 'third_party' / 'CosyVoice'))
sys.path.insert(0, str(PROJECT_ROOT / 'third_party' / 'CosyVoice' / 'third_party' / 'Matcha-TTS'))

MODEL_DIR = config['tts']['model_dir']
SPEAKER_WAV = config['tts']['speaker_wav']
PROMPT_TEXT = config['tts']['prompt_text']
PUBLIC_OUTPUT_SCHEMA_VERSION = 1

MAX_CHARS_PER_CLIP = config['audio']['max_chars_per_clip']
SPEED = config['tts'].get('speed', 1.0)
INTER_SENTENCE_SILENCE_MS = config['tts'].get('inter_sentence_silence_ms', 0)
_tts = None
_torch = None
_torchaudio = None
_polyphone_support_loaded = False
_pinyin = None
_style_tone = None
_to_initials = None
_to_finals_tone = None
_stable_aligner = None


@dataclass
class SentenceSpec:
    raw_sentence: str
    tts_sentence: str
    sub_sentence: str
    word_count: int


@dataclass
class BatchSpec:
    sentences: List[SentenceSpec]
    tts_text: str


@dataclass
class ClipSpec:
    index: int
    sentences: List[SentenceSpec]


class AlignmentError(RuntimeError):
    pass


def _decode_subprocess_output(data):
    return data.decode(errors='replace')


def _book_output_dir(book_name: str) -> Path:
    return OUTPUT_DIR / build_output_relpath([book_name], set())


def _load_torch_modules():
    global _torch, _torchaudio
    if _torch is None or _torchaudio is None:
        import torch as torch_module
        import torchaudio as torchaudio_module
        _torch = torch_module
        _torchaudio = torchaudio_module
    return _torch, _torchaudio


def _ensure_polyphone_support_loaded():
    global _polyphone_support_loaded, _pinyin, _style_tone, _to_initials, _to_finals_tone
    if _polyphone_support_loaded:
        return

    from pypinyin import pinyin as pinyin_func, Style, load_phrases_dict
    from pypinyin.contrib.tone_convert import to_initials as to_initials_func, to_finals_tone as to_finals_tone_func

    load_phrases_dict({
        '精校': [['jīng'], ['jiào']],
        '校对': [['jiào'], ['duì']],
        '校勘': [['jiào'], ['kān']],
    })

    _pinyin = pinyin_func
    _style_tone = Style.TONE
    _to_initials = to_initials_func
    _to_finals_tone = to_finals_tone_func
    _polyphone_support_loaded = True


def get_tts():
    global _tts
    if _tts is None:
        try:
            from cosyvoice.cli.cosyvoice import AutoModel
            _tts = AutoModel(model_dir=MODEL_DIR)
            _tts.add_zero_shot_spk(PROMPT_TEXT, SPEAKER_WAV, 'narrator')
        except Exception as e:
            raise RuntimeError(
                f"无法加载 CosyVoice 模型: {MODEL_DIR}。请确认模型已下载；可运行："
                "uv run python -c \"from huggingface_hub import snapshot_download; "
                "snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', "
                "local_dir='pretrained_models/Fun-CosyVoice3-0.5B')\"。"
                f"原始错误: {e}"
            ) from e
    return _tts


def get_stable_aligner():
    global _stable_aligner
    if _stable_aligner is None:
        try:
            import stable_whisper
            _stable_aligner = stable_whisper.load_model('base')
        except Exception as e:
            raise RuntimeError(
                "无法加载 stable-ts 对齐模型。请先安装 stable-ts，并确认 whisper 相关依赖可用。"
                f" 原始错误: {e}"
            ) from e
    return _stable_aligner


def _normalize_alignment_text(text: str) -> str:
    normalized = re.sub(r'\s+', '', mask_punctuations(text))
    return re.sub(r'[^一-鿿0-9a-zA-Z]+', '', normalized)

book_delimiter = config['text_processing']['book_delimiter']
SUPPORTED_INPUT_SUFFIXES = {'.txt', '.epub', '.mobi'}
CONVERTED_TEXT_SUFFIX = '.txt2audio.txt'
CHAPTER_MANIFEST_FILE_NAME = 'chapter_manifest.json'


def load_txt_file(file_path):
    results = from_path(file_path)  # 自动检测文件编码
    best = results.best()
    if best is None:
        raise ValueError(f'无法检测文件编码或文件为空: {file_path}')
    return str(best)


def get_converted_txt_path(file_path: str) -> Path:
    source = Path(file_path)
    return source.with_name(f'{source.stem}{CONVERTED_TEXT_SUFFIX}')


def convert_book_to_txt(file_path: str, output_txt_path: Path) -> Path:
    tmp_path = output_txt_path.with_name(output_txt_path.stem + '.tmp' + output_txt_path.suffix)
    if tmp_path.exists():
        tmp_path.unlink()
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which('ebook-convert') is None:
        raise RuntimeError(
            "检测到 EPUB/MOBI 输入，但系统未安装 Calibre 的 `ebook-convert`。"
            " 请先安装 Calibre，并确保 `ebook-convert` 已加入 PATH。"
        )

    ret = _subprocess.run(
        ['ebook-convert', file_path, str(tmp_path)],
        capture_output=True,
        text=True
    )
    if ret.returncode != 0:
        if tmp_path.exists():
            tmp_path.unlink()
        error_message = ret.stderr.strip() or ret.stdout.strip() or 'unknown error'
        raise RuntimeError(f'电子书转换失败: {error_message}')

    shutil.move(tmp_path, output_txt_path)
    return output_txt_path


def load_book_file(file_path: str):
    source_path = Path(file_path)
    suffix = source_path.suffix.lower()

    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        supported = ', '.join(sorted(SUPPORTED_INPUT_SUFFIXES))
        raise ValueError(f"不支持的输入格式: {suffix or '无后缀'}，仅支持 {supported}")

    if suffix == '.txt':
        return load_txt_file(file_path), source_path, False

    converted_txt_path = get_converted_txt_path(file_path)
    if converted_txt_path.is_file():
        if converted_txt_path.stat().st_size > 0:
            return load_txt_file(str(converted_txt_path)), converted_txt_path, False
        converted_txt_path.unlink()

    converted_txt_path = convert_book_to_txt(file_path, converted_txt_path)
    return load_txt_file(str(converted_txt_path)), converted_txt_path, True


def get_word_num(text):
    return len(re.findall(u'[\u4e00-\u9fff]', text))


def save_audio_file(wav_tensor, sample_rate, output_path: str, video_clip_index: int, export_indices: List) -> None:
    if wav_tensor is None or wav_tensor.numel() == 0:
        return
    _, torchaudio_module = _load_torch_modules()
    export_indices.append(video_clip_index)
    output_stem = Path(output_path)
    audio_file_path = build_clip_output_path(output_stem, video_clip_index, '.wav')
    tmp_path = tmp_output_path(audio_file_path)
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0)
    torchaudio_module.save(str(tmp_path), wav_tensor, sample_rate)
    shutil.move(tmp_path, audio_file_path)


def convert_wav_to_mp3(wav_path, bitrate='128k'):
    """WAV → MP3，成功后删除原 WAV。使用临时文件确保原子写入。"""
    wav_path = Path(wav_path)
    mp3_path = wav_path.with_suffix('.mp3')
    tmp_path = tmp_output_path(mp3_path)
    ret = _subprocess.run(
        ['ffmpeg', '-y', '-i', str(wav_path), '-codec:a', 'libmp3lame', '-b:a', bitrate, str(tmp_path)],
        capture_output=True
    )
    if ret.returncode == 0:
        shutil.move(tmp_path, mp3_path)
        os.remove(wav_path)
        return str(mp3_path)
    else:
        if tmp_path.is_file():
            os.remove(tmp_path)
        stderr = _decode_subprocess_output(ret.stderr).strip()
        if stderr:
            logger.debug(stderr)
        raise RuntimeError(_subprocess_failure_message('MP3 conversion failed', ret.returncode, stderr))


def check_export_file_exists(output_path, video_clip_index, require_subtitles=False):
    """返回 True 表示需要导出（文件不存在），用于断点续生成。
    自动清理中断留下的临时文件和空文件。"""
    output_stem = Path(output_path)
    # 清理中断留下的临时文件
    for ext in ('.wav', '.mp3', '.mp4', '.srt'):
        tmp = tmp_output_path(build_clip_output_path(output_stem, video_clip_index, ext))
        if tmp.is_file():
            os.remove(tmp)
            logger.debug(f"Removed incomplete temp file: {tmp}")
    # 检查已完成的文件
    subtitle_path = build_clip_output_path(output_stem, video_clip_index, '.srt')
    if require_subtitles and subtitle_path.is_file() and subtitle_path.stat().st_size == 0:
        os.remove(subtitle_path)
        logger.debug(f"Removed empty file: {subtitle_path}")

    for ext in ('.mp4', '.mp3', '.wav'):
        path = build_clip_output_path(output_stem, video_clip_index, ext)
        if path.is_file():
            if path.stat().st_size == 0:
                os.remove(path)
                logger.debug(f"Removed empty file: {path}")
                continue
            if require_subtitles and ext in ('.mp3', '.wav') and not subtitle_path.is_file():
                logger.debug(f"{path} exists but subtitle is missing; clip will be regenerated.")
                return True
            logger.debug(f"{path} is already generated !")
            return False
    return True


def generate_audio_clip(text: str, output_path: str, sample_rate=None, generate_subtitles: bool = True):
    """将一章文本转为音频，按 MAX_CHARS_PER_CLIP 切分为多个片段（-1 则不分片）。
    字幕模式下使用 clip 内小批量 TTS + 对齐器回贴句级时间。"""
    torch_module, _ = _load_torch_modules()
    from subtitle import save_subtitle_file

    cosyvoice = get_tts()
    if sample_rate is None:
        sample_rate = cosyvoice.sample_rate

    exported_clip_indices = []
    sentence_specs = _build_sentence_specs(text)
    clip_specs = _build_clip_specs(sentence_specs)

    for clip in clip_specs:
        export = check_export_file_exists(
            output_path=output_path,
            video_clip_index=clip.index,
            require_subtitles=generate_subtitles,
        )
        if not export:
            continue

        batch_specs = _build_batch_specs(clip.sentences)
        batch_tensors = []
        subtitle_entries = []
        clip_offset_seconds = 0.0
        for batch in batch_specs:
            batch_tensor, batch_entries = _synthesize_sentence_group(
                cosyvoice=cosyvoice,
                sentences=batch.sentences,
                sample_rate=sample_rate,
                generate_subtitles=generate_subtitles,
                torch_module=torch_module,
            )
            if batch_tensor is None:
                continue
            batch_tensors.append(batch_tensor)

            if generate_subtitles:
                subtitle_entries.extend(_shift_subtitle_entries(batch_entries, clip_offset_seconds))

            clip_offset_seconds += batch_tensor.shape[-1] / sample_rate

        combined = torch_module.cat(batch_tensors, dim=-1) if batch_tensors else None
        save_audio_file(combined, sample_rate, output_path, clip.index, exported_clip_indices)
        if generate_subtitles and subtitle_entries:
            save_subtitle_file(subtitle_entries, output_path, clip.index)
    return exported_clip_indices


def _build_sentence_specs(text: str) -> List[SentenceSpec]:
    raw_sentences = [s.strip() for s in re.split(r'(?<=[。！？])', text) if s.strip()]
    sentence_specs = []
    for raw_sentence in raw_sentences:
        tts_sentence = mask_punctuations(text=annotate_polyphones(raw_sentence))
        sub_sentence = mask_punctuations(text=raw_sentence).rstrip('。')
        if not tts_sentence or not sub_sentence:
            continue
        sentence_specs.append(SentenceSpec(
            raw_sentence=raw_sentence,
            tts_sentence=tts_sentence,
            sub_sentence=sub_sentence,
            word_count=get_word_num(text=raw_sentence),
        ))
    return sentence_specs


def _build_clip_specs(sentence_specs: List[SentenceSpec]) -> List[ClipSpec]:
    if MAX_CHARS_PER_CLIP <= 0:
        return [ClipSpec(index=1, sentences=list(sentence_specs))] if sentence_specs else []

    clips = []
    current_sentences = []
    current_words = 0
    clip_index = 1
    for sentence in sentence_specs:
        if current_sentences and current_words + sentence.word_count > MAX_CHARS_PER_CLIP:
            clips.append(ClipSpec(index=clip_index, sentences=current_sentences))
            clip_index += 1
            current_sentences = []
            current_words = 0
        current_sentences.append(sentence)
        current_words += sentence.word_count
    if current_sentences:
        clips.append(ClipSpec(index=clip_index, sentences=current_sentences))
    return clips


def _build_batch_specs(sentences: List[SentenceSpec]) -> List[BatchSpec]:
    batches = []
    idx = 0
    while idx < len(sentences):
        current = [sentences[idx]]
        current_len = len(sentences[idx].tts_sentence)
        idx += 1
        if current_len >= 40:
            batches.append(BatchSpec(sentences=current, tts_text=''.join(s.tts_sentence for s in current)))
            continue
        while idx < len(sentences):
            next_sentence = sentences[idx]
            next_len = len(next_sentence.tts_sentence)
            if len(current) >= 4:
                break
            if current_len >= 60 and current_len + next_len > 90:
                break
            current.append(next_sentence)
            current_len += next_len
            idx += 1
        batches.append(BatchSpec(sentences=current, tts_text=''.join(s.tts_sentence for s in current)))
    return batches


def _synthesize_sentence_group(cosyvoice, sentences: List[SentenceSpec], sample_rate: int, generate_subtitles: bool, torch_module):
    tts_text = ''.join(sentence.tts_sentence for sentence in sentences)
    batch_chunks = []
    for chunk in cosyvoice.inference_zero_shot(
        tts_text, PROMPT_TEXT, SPEAKER_WAV,
        zero_shot_spk_id='narrator', stream=False, speed=SPEED
    ):
        batch_chunks.append(chunk['tts_speech'])

    batch_tensor = torch_module.cat(batch_chunks, dim=-1) if batch_chunks else None
    if batch_tensor is None or not generate_subtitles:
        return batch_tensor, []

    try:
        entries = _align_tensor_subtitles(sentences, batch_tensor, sample_rate)
        return batch_tensor, entries
    except AlignmentError as exc:
        if len(sentences) == 1:
            raise RuntimeError(
                f"字幕对齐失败，且无法继续切分。句子: {sentences[0].raw_sentence[:50]}"
            ) from exc
        split_index = len(sentences) // 2
        logger.warning(
            "Batch alignment failed for %s sentences; retrying smaller groups.",
            len(sentences),
        )
        left_tensor, left_entries = _synthesize_sentence_group(
            cosyvoice=cosyvoice,
            sentences=sentences[:split_index],
            sample_rate=sample_rate,
            generate_subtitles=generate_subtitles,
            torch_module=torch_module,
        )
        right_tensor, right_entries = _synthesize_sentence_group(
            cosyvoice=cosyvoice,
            sentences=sentences[split_index:],
            sample_rate=sample_rate,
            generate_subtitles=generate_subtitles,
            torch_module=torch_module,
        )

        tensors = [tensor for tensor in (left_tensor, right_tensor) if tensor is not None]
        combined_tensor = torch_module.cat(tensors, dim=-1) if tensors else None
        right_offset = left_tensor.shape[-1] / sample_rate if left_tensor is not None else 0.0
        return combined_tensor, left_entries + _shift_subtitle_entries(right_entries, right_offset)


def _align_tensor_subtitles(sentences: List[SentenceSpec], wav_tensor, sample_rate: int):
    wav_path = None
    try:
        wav_path = _write_alignment_temp_wav(wav_tensor, sample_rate)
        return _align_sentences_with_audio(sentences, wav_path)
    finally:
        if wav_path and wav_path.exists():
            wav_path.unlink()


def _write_alignment_temp_wav(wav_tensor, sample_rate: int) -> Path:
    _, torchaudio_module = _load_torch_modules()
    with tempfile.NamedTemporaryFile(prefix='txt2audio-align-', suffix='.wav', delete=False) as tmp_file:
        temp_path = Path(tmp_file.name)
    save_tensor = wav_tensor.unsqueeze(0) if wav_tensor.dim() == 1 else wav_tensor
    torchaudio_module.save(str(temp_path), save_tensor, sample_rate)
    return temp_path


def _align_sentences_with_audio(sentences: List[SentenceSpec], wav_path: Path):
    aligner = get_stable_aligner()
    align_text = ''.join(_normalize_alignment_text(sentence.raw_sentence) for sentence in sentences)
    if not align_text:
        return []
    result = aligner.align(str(wav_path), align_text, language='zh', regroup=False)
    word_segments = _extract_alignment_units(result)
    return _map_alignment_to_sentences(sentences, word_segments)


def _shift_subtitle_entries(entries, offset_seconds: float):
    if offset_seconds == 0:
        return entries
    return [
        (start + offset_seconds, end + offset_seconds, text)
        for start, end, text in entries
    ]


def _extract_alignment_units(result):
    units = []
    segments = getattr(result, 'segments', None) or result.get('segments', [])
    for segment in segments:
        words = getattr(segment, 'words', None) or segment.get('words', [])
        for word in words:
            text = getattr(word, 'word', None) or word.get('word', '')
            start = getattr(word, 'start', None)
            end = getattr(word, 'end', None)
            if start is None or end is None or not text:
                continue
            units.append({
                'text': re.sub(r'\s+', '', text),
                'start': float(start),
                'end': float(end),
            })
    return units


def _map_alignment_to_sentences(sentences: List[SentenceSpec], units):
    if not units:
        raise AlignmentError('No alignment units returned')

    entries = []
    unit_index = 0
    for sentence in sentences:
        target = _normalize_alignment_text(sentence.raw_sentence)
        if not target:
            continue
        start = None
        end = None
        matched = ''
        while unit_index < len(units):
            unit = units[unit_index]
            text = unit['text']
            if not text:
                unit_index += 1
                continue
            remaining = target[len(matched):]
            if remaining and not remaining.startswith(text):
                raise AlignmentError(
                    f"Alignment mismatch for sentence: expected {remaining[:20]!r}, got {text!r}"
                )
            if start is None:
                start = unit['start']
            end = unit['end']
            matched += text
            unit_index += 1
            if len(matched) >= len(target):
                break
        if matched != target or start is None or end is None:
            raise AlignmentError(
                f"Incomplete alignment for sentence: matched {len(matched)}/{len(target)} chars"
            )
        entries.append((start, end, sentence.sub_sentence))
    return entries


def mask_punctuations(text):
    text = text.replace('——', '，')
    text = re.sub(r'[\u201c\u201d\u2018\u2019]', '', text)  # remove Chinese quotes “”''
    text = re.sub(r"([！？=@。])+", r"\1", text)  # replace ?! -> !
    text = re.sub(r"([！@=…？])\1+", r"\1", text)  # replace !! -> !
    text = re.sub(r'[…]+', '。', text)
    text = text.replace('·', '').replace('※', '')
    text = re.sub(r'[=]+', '', text)
    text = text.replace('《', '').replace('》', '').replace("\n", " ").strip()

    # 移除 URL
    text = re.sub(
        r"(?:https?://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)(?:/(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;:@&=])*)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;:@&=])*))*)(?:\?(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;:@&=])*))?)?)|(?:s?ftp://(?:(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?&=])*)(?::(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?&=])*))?@)?(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?))(?:/(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&=])*)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&=])*))*)(?:;type=[AIDaid])?)?)|(?:news:(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;/?:&=])+@(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3})))|(?:[a-zA-Z](?:[a-zA-Z\d]|[_.+-])*)|\*))|(?:nntp://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)/(?:[a-zA-Z](?:[a-zA-Z\d]|[_.+-])*)(?:/(?:\d+))?)|(?:telnet://(?:(?:(?:(?:(?:[a-zA-Z\d$-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?&=])*)(?::(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?&=])*))?@)?(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?))/?)|(?:gopher://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)(?:/(?:[a-zA-Z\d$\-_.+!*'(),;/?:@&=]|(?:%[a-fA-F\d]{2}))(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),;/?:@&=]|(?:%[a-fA-F\d]{2}))*)(?:%09(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;:@&=])*)(?:%09(?:(?:[a-zA-Z\d$\-_.+!*'(),;/?:@&=]|(?:%[a-fA-F\d]{2}))*))?)?)?)?)|(?:wais://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)/(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*)(?:(?:/(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*)/(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*))|\?(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;:@&=])*))?)|(?:mailto:(?:(?:[a-zA-Z\d$\-_.+!*'(),;/?:@&=]|(?:%[a-fA-F\d]{2}))+))|(?:file://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))|localhost)?/(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&=])*)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&=])*))*))|(?:prospero://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)/(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&=])*)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&=])*))*)(?:(?:;(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&])*)=(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&])*)))*)|(?:ldap://(?:(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?))?/(?:(?:(?:(?:(?:(?:(?:[a-zA-Z\d]|%(?:3\d|[46][a-fA-F\d]|[57][Aa\d]))|(?:%20))+|(?:OID|oid)\.(?:(?:\d+)(?:\.(?:\d+))*))(?:(?:%0[Aa])?(?:%20)*)=(?:(?:%0[Aa])?(?:%20)*))?(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*))(?:(?:(?:%0[Aa])?(?:%20)*)\+(?:(?:%0[Aa])?(?:%20)*)(?:(?:(?:(?:(?:[a-zA-Z\d]|%(?:3\d|[46][a-fA-F\d]|[57][Aa\d]))|(?:%20))+|(?:OID|oid)\.(?:(?:\d+)(?:\.(?:\d+))*))(?:(?:%0[Aa])?(?:%20)*)=(?:(?:%0[Aa])?(?:%20)*))?(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*)))*)(?:(?:(?:(?:%0[Aa])?(?:%20)*)(?:[;,])(?:(?:%0[Aa])?(?:%20)*))(?:(?:(?:(?:(?:(?:[a-zA-Z\d]|%(?:3\d|[46][a-fA-F\d]|[57][Aa\d]))|(?:%20))+|(?:OID|oid)\.(?:(?:\d+)(?:\.(?:\d+))*))(?:(?:%0[Aa])?(?:%20)*)=(?:(?:%0[Aa])?(?:%20)*))?(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*))(?:(?:(?:%0[Aa])?(?:%20)*)\+(?:(?:%0[Aa])?(?:%20)*)(?:(?:(?:(?:(?:[a-zA-Z\d]|%(?:3\d|[46][a-fA-F\d]|[57][Aa\d]))|(?:%20))+|(?:OID|oid)\.(?:(?:\d+)(?:\.(?:\d+))*))(?:(?:%0[Aa])?(?:%20)*)=(?:(?:%0[Aa])?(?:%20)*))?(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*)))*))*(?:(?:(?:%0[Aa])?(?:%20)*)(?:[;,])(?:(?:%0[Aa])?(?:%20)*))?)(?:\?(?:(?:(?:(?:[a-zA-Z\d$-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+)(?:,(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+))*)?)(?:\?(?:base|one|sub)(?:\?(?:((?:[a-zA-Z\d$\-_.+!*'(),;/?:@&=]|(?:%[a-fA-F\d]{2}))+)))?)?)?)|(?:(?:z39\.50[rs])://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:d+))?)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+)(?:\+(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+))*(?:\?(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+))?)?(?:;esn=(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+))?(?:;rs=(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+)(?:\+(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+))*)?))|(?:cid:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?:@&=])*))|(?:mid:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?:@&=])*)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?:@&=])*))?)|(?:vemmi://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[/?:@&=])*)(?:(?:;(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[/?:@&])*)=(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[/?:@&])*))*))?)|(?:imap://(?:(?:(?:(?:(?:(?:(?:[a-zA-Z\d$-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~])+)(?:(?:;[Aa][Uu][Tt][Hh]=(?:\*|(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~])+))))?)|(?:(?:;[Aa][Uu][Tt][Hh]=(?:\*|(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-Fd]{2}))|[&=~])+)))(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~])+))?))@)?(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:.(?:\d+)){3}))(?::(?:\d+))?))/(?:(?:(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~:@/])+)?;[Tt][Yy][Pp][Ee]=(?:[Ll](?:[Ii][Ss][Tt]|[Ss][Uu][Bb])))|(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~:@/])+)(?:\?(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~:@/])+))?(?:(?:;[Uu][Ii][Dd][Vv][Aa][Ll][Ii][Dd][Ii][Tt][Yy]=(?:[1-9]\d*)))?)|(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~:@/])+)(?:(?:;[Uu][Ii][Dd][Vv][Aa][Ll][Ii][Dd][Ii][Tt][Yy]=(?:[1-9]\d*)))?(?:/;[Uu][Ii][Dd]=(?:[1-9]\d*))(?:(?:/;[Ss][Ee][Cc][Tt][Ii][Oo][Nn]=(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~:@/])+)))?)))?)|(?:nfs:(?:(?://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)(?:(?:/(?:(?:(?:(?:(?:[a-zA-Z\d\$\-_.!~*'(),])|(?:%[a-fA-F\d]{2})|[:@&=+])*)(?:/(?:(?:(?:[a-zA-Z\d\$\-_.!~*'(),])|(?:%[a-fA-F\d]{2})|[:@&=+])*))*)?)))?)|(?:/(?:(?:(?:(?:(?:[a-zA-Z\d\$\-_.!~*'(),])|(?:%[a-fA-F\d]{2})|[:@&=+])*)(?:/(?:(?:(?:[a-zA-Z\d\$\-_.!~*'(),])|(?:%[a-fA-F\d]{2})|[:@&=+])*))*)?))|(?:(?:(?:(?:(?:[a-zA-Z\d\$\-_.!~*'(),])|(?:%[a-fA-F\d]{2})|[:@&=+])*)(?:/(?:(?:(?:[a-zA-Z\d\$\-_.!~*'(),])|(?:%[a-fA-F\d]{2})|[:@&=+])*))*)?)))",
        '', text)
    text = text.strip()

    if not text or not re.search(u'[\u4e00-\u9fff0-9a-zA-Z]+', text):
        return ''
    if re.search(u'[\u4e00-\u9fff]', text[-1]):  # 确保以句号结尾，TTS 需要
        text += '。'
    return text


# 从 CosyVoice3Tokenizer 的 additional_special_tokens 提取的合法拼音 token 集合
# (来源: third_party/CosyVoice/cosyvoice/tokenizer/tokenizer.py:295-307)
_VALID_PINYIN_TOKENS = {
    # 声母
    'b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n',
    'p', 'q', 'r', 's', 'sh', 't', 'w', 'x', 'y', 'z', 'zh',
    # 韵母 (无声调)
    'a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'i',
    'ian', 'in', 'ing', 'iu', 'o', 'ong', 'ou', 'u', 'uang', 'ue',
    'un', 'uo',
    # 带声调韵母 — i 系列
    'ià', 'iàn', 'iàng', 'iào', 'iá', 'ián', 'iáng', 'iáo',
    'iè', 'ié', 'iòng', 'ióng', 'iù', 'iú',
    'iā', 'iān', 'iāng', 'iāo', 'iē', 'iě', 'iōng', 'iū',
    'iǎ', 'iǎn', 'iǎng', 'iǎo', 'iǒng', 'iǔ',
    # 带声调韵母 — u 系列
    'uà', 'uài', 'uàn', 'uàng', 'uá', 'uái', 'uán', 'uáng',
    'uè', 'ué', 'uì', 'uí', 'uò', 'uó',
    'uā', 'uāi', 'uān', 'uāng', 'uē', 'uě', 'uī', 'uō',
    'uǎ', 'uǎi', 'uǎn', 'uǎng', 'uǐ', 'uǒ',
    # 带声调韵母 — v (ü) 系列
    'vè',
    # 带声调韵母 — 独立元音
    'à', 'ài', 'àn', 'àng', 'ào', 'á', 'ái', 'án', 'áng', 'áo',
    'è', 'èi', 'èn', 'èng', 'èr', 'é', 'éi', 'én', 'éng', 'ér',
    'ì', 'ìn', 'ìng', 'í', 'ín', 'íng',
    'ò', 'òng', 'òu', 'ó', 'óng', 'óu',
    'ù', 'ùn', 'ú', 'ún',
    'ā', 'āi', 'ān', 'āng', 'āo', 'ē', 'ēi', 'ēn', 'ēng',
    'ě', 'ěi', 'ěn', 'ěng', 'ěr',
    'ī', 'īn', 'īng', 'ō', 'ōng', 'ōu', 'ū', 'ūn',
    'ǎ', 'ǎi', 'ǎn', 'ǎng', 'ǎo', 'ǐ', 'ǐn', 'ǐng',
    'ǒ', 'ǒng', 'ǒu', 'ǔ', 'ǔn',
    # 独立 ü 韵母
    'ǘ', 'ǚ', 'ǜ',
}


# 需要拼音标注的高频多音字（仅标注 TTS 经常读错的字，避免过度标注影响自然度）
_POLYPHONE_CHARS = set('校')

def annotate_polyphones(text: str) -> str:
    """对多音字注入 CosyVoice3 拼音 token，如 '给予' → '[j][ǐ]予'。"""
    if not text:
        return text

    _ensure_polyphone_support_loaded()

    # 整句上下文消歧
    tone_readings = _pinyin(text, style=_style_tone, heteronym=False, strict=True)

    result = []
    text_pos = 0
    for reading_entry in tone_readings:
        val = reading_entry[0]
        char = text[text_pos]

        # 非汉字：pypinyin 把连续非汉字合并为一个条目
        if not ('\u4e00' <= char <= '\u9fff'):
            result.append(val)
            text_pos += len(val)
            continue

        text_pos += 1

        # 非多音字
        if char not in _POLYPHONE_CHARS:
            result.append(char)
            continue

        # 轻声不标注（无声调符号，如 'de', 'le'）
        if not any(c in val for c in 'āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ'):
            result.append(char)
            continue

        initial = _to_initials(val, strict=False)
        final = _to_finals_tone(val, strict=False)

        # ü → v fallback (CosyVoice3 用 v 表示复合韵母中的 ü)
        if final not in _VALID_PINYIN_TOKENS:
            final_v = final.replace('ü', 'v')
            if final_v in _VALID_PINYIN_TOKENS:
                final = final_v

        # 验证 token 合法性
        if initial and initial not in _VALID_PINYIN_TOKENS:
            result.append(char)
            continue
        if final and final not in _VALID_PINYIN_TOKENS:
            result.append(char)
            continue

        tokens = []
        if initial:
            tokens.append(f'[{initial}]')
        if final:
            tokens.append(f'[{final}]')

        result.append(''.join(tokens) if tokens else char)

    return ''.join(result)


def generate_chapter_parts(chapter_name, last_special_delimiter):
    """从 chapter_structure 生成章节路径片段，如 [书名, 第一卷, 第一章]。"""
    if last_special_delimiter:  # 序/楔子/后记等，包含末位 special slot
        parts = [i for i in chapter_name if i]
    else:
        parts = [i for i in chapter_name[:-1] if i]

    # 引言内容（出现在第一个卷/章/序之前的文本）只有书名没有子路径，
    # 需要加上"引言"保证输出到书目录内部，而不是和书目录同级
    if len(parts) == 1:
        parts.append('引言')

    return parts


def check_special_delimiter(text):
    for sub_text in text.split(' '):
        for p in ['序', '序章', '序言', '前言', '楔子', '引言', '后记', '终章']:
            if p == sub_text:
                return p

    return ''


def empty_structure(chapter_structure, start):
    # 高层级变化时清空低层级，如新卷时清空章名（末位 special slot 保留）
    for i in range(start, len(chapter_structure) - 1):
        chapter_structure[i] = ''


def get_delimiter_pattern(delimiter):
    return rf"(^|\s)(第[零一二三四五六七八九十]+{delimiter}|{delimiter}[零一二三四五六七八九十]+)($|\s)"


def construct_text_and_name(raw_data, book_name: str):
    table_of_contents = {}
    output_targets = {}
    contents_of_chapter = {}
    toc_index = 0
    # [书名, 卷, 章, special_delimiter]，用于拼接输出路径
    chapter_structure = [book_name] + ['' for _ in book_delimiter] + ['']
    contents = []
    input_text_lines = re.split('\r\n|\n', raw_data)
    last_special_delimiter = False
    used_output_paths = set()

    for line in input_text_lines:
        line = line.strip()

        if not line:
            continue

        # 过滤纯装饰分隔线（===、---、***、~~~ 等），避免被 mask_punctuations 转为多余标点
        if re.match(r'^[=\-*~#]{3,}$', line):
            continue

        new_chapter = False
        special_delimiter = check_special_delimiter(line)
        if special_delimiter:
            new_chapter = True
            chapter_structure[-1] = special_delimiter

        for idx, delimiter in enumerate(book_delimiter):
            pattern = get_delimiter_pattern(delimiter)
            x = re.search(pattern, line)
            if x:
                matched_chapter_name = x.group()
                if chapter_structure[idx + 1] != matched_chapter_name.strip():
                    # NOTE: 有时候文章中会插入卷/章节，如果和之前没有变化，那么就继续
                    new_chapter = True
                    break

        if new_chapter:
            if contents:
                chapter_parts = generate_chapter_parts(chapter_name=chapter_structure,
                                                       last_special_delimiter=last_special_delimiter)
                if chapter_parts:
                    table_of_contents[toc_index] = build_display_path(chapter_parts)
                    output_targets[toc_index] = build_output_relpath(chapter_parts, used_output_paths)
                    contents_of_chapter[toc_index] = contents
                    toc_index += 1
                last_special_delimiter = False
                contents = []

            for idx, delimiter in enumerate(book_delimiter):
                pattern = get_delimiter_pattern(delimiter)
                x = re.search(pattern, line)
                if x:
                    matched_chapter_name = x.group()
                    chapter_structure[idx + 1] = matched_chapter_name.strip()
                    # NOTE: 前提是先卷，后章
                    empty_structure(chapter_structure, start=idx + 2)
        else:
            contents.append(line)

        if special_delimiter:
            last_special_delimiter = True

    if contents:
        chapter_parts = generate_chapter_parts(chapter_name=chapter_structure,
                                               last_special_delimiter=last_special_delimiter)
        if chapter_parts:
            table_of_contents[toc_index] = build_display_path(chapter_parts)
            output_targets[toc_index] = build_output_relpath(chapter_parts, used_output_paths)
            contents_of_chapter[toc_index] = contents
            toc_index += 1

    toc_file_path = _book_output_dir(book_name) / '目录.txt'
    save_table_of_contents(file_path=toc_file_path, table_of_contents=table_of_contents)

    return table_of_contents, output_targets, contents_of_chapter


def save_table_of_contents(file_path, table_of_contents: Dict):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\ufeff')  # UTF-8 BOM，帮助 macOS Finder 正确识别编码
        for k, v in table_of_contents.items():
            w = f'{k:>5}:{v} \n'
            logger.debug(w.rstrip())
            f.write(w)


def save_chapter_manifest(file_path: Path, manifest: dict) -> str:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_output_path(file_path)
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write('\n')
    shutil.move(tmp_path, file_path)
    return str(file_path)


def _check_ffmpeg():
    """Check if ffmpeg is available on the system."""
    try:
        _subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (FileNotFoundError, _subprocess.CalledProcessError):
        raise RuntimeError(
            "未检测到 ffmpeg。请先安装 ffmpeg："
            "macOS: brew install ffmpeg；"
            "Windows: winget install ffmpeg，或从 https://ffmpeg.org/download.html 下载。"
        )


def _format_duration(seconds):
    """秒数 -> 可读时长，如 '2h 15m 30s' 或 '45.2s'。"""
    if seconds < 60:
        return f'{seconds:.1f}s'
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f'{h}h {m}m {s}s' if h > 0 else f'{m}m {s}s'


def _apply_config_overrides(args):
    """将 CLI 传入的配置覆盖应用到全局 config，并刷新模块级常量。"""
    global RESOURCES_DIR, OUTPUT_DIR, MODEL_DIR, SPEAKER_WAV, PROMPT_TEXT
    global SPEED, INTER_SENTENCE_SILENCE_MS, MAX_CHARS_PER_CLIP

    if args.speed is not None:
        config['tts']['speed'] = args.speed
    if args.output_format is not None:
        config['audio']['output_format'] = args.output_format
    if args.output_dir is not None:
        config['paths']['output_dir'] = args.output_dir

    for override in args.set:
        key, _, value = override.partition('=')
        if not _:
            logger.warning(f"Ignoring malformed --set: {override} (expected KEY=VALUE)")
            continue
        parts = key.split('.')
        d = config
        try:
            for p in parts[:-1]:
                d = d[p]
        except KeyError:
            logger.warning(f"Ignoring --set {override}: '{key}' is not a valid config key")
            continue
        existing = d.get(parts[-1])
        if isinstance(existing, bool):
            value = value.lower() in ('true', '1', 'yes')
        elif isinstance(existing, int):
            value = int(value)
        elif isinstance(existing, float):
            value = float(value)
        d[parts[-1]] = value

    normalize_config_paths(config)

    # 刷新模块级常量
    RESOURCES_DIR = config['paths']['resources_dir']
    OUTPUT_DIR = config['paths']['output_dir']
    MODEL_DIR = config['tts']['model_dir']
    SPEAKER_WAV = config['tts']['speaker_wav']
    PROMPT_TEXT = config['tts']['prompt_text']
    SPEED = config['tts'].get('speed', 1.0)
    INTER_SENTENCE_SILENCE_MS = config['tts'].get('inter_sentence_silence_ms', 0)
    MAX_CHARS_PER_CLIP = config['audio']['max_chars_per_clip']


def _json_error(error_code, message, *, retryable=False, details=None):
    """输出 JSON 格式的错误信息到 stdout。"""
    print(json.dumps({
        "schema_version": PUBLIC_OUTPUT_SCHEMA_VERSION,
        "status": "error",
        "error_code": error_code,
        "error": error_code,
        "message": message,
        "retryable": retryable,
        "details": details or {},
    }, ensure_ascii=False))


def _report_error(args, error_code, message, *, retryable=False, details=None):
    """错误输出：JSON 模式写 stdout，人类模式写 stderr rich 格式。"""
    if args.json or getattr(args, 'plan_json', False):
        _json_error(error_code, message, retryable=retryable, details=details)
    else:
        console.print(f"[bold red]Error:[/bold red] {message}")


def _last_stderr_line(stderr):
    for line in reversed(stderr.splitlines()):
        line = line.strip()
        if line:
            return line
    return ''


def _subprocess_failure_message(action, returncode, stderr):
    detail = _last_stderr_line(stderr)
    if detail:
        return f'{action} (code {returncode}): {detail}'
    return f'{action} (code {returncode})'


def _should_show_catalog(args):
    return not args.json and not args.plan_json and not args.quiet and args.range is None


def _should_show_resume_hint(args):
    return _should_show_catalog(args)


def _should_show_chapter_progress(args, chapter_indices):
    return not args.json and not args.quiet and len(chapter_indices) > 1


def _should_print_summary(args, conversion_failures):
    return not args.json and (not args.quiet or bool(conversion_failures))


def _existing_outputs(output_path: Path) -> list[dict]:
    outputs = []
    clip_index = 1
    while True:
        clip_outputs = []
        for ext in ('.mp4', '.mp3', '.wav', '.srt'):
            path = build_clip_output_path(output_path, clip_index, ext)
            if path.is_file() and path.stat().st_size > 0:
                clip_outputs.append({
                    "format": ext.lstrip('.'),
                    "path": str(path),
                    "bytes": path.stat().st_size,
                })
        if not clip_outputs:
            break
        outputs.extend(clip_outputs)
        clip_index += 1
    return outputs


def _artifact_record(path, *, chapter_index, clip_index, role):
    artifact_path = Path(path)
    return {
        "path": str(artifact_path),
        "format": artifact_path.suffix.lstrip('.'),
        "bytes": artifact_path.stat().st_size if artifact_path.is_file() else 0,
        "chapter_index": chapter_index,
        "clip_index": clip_index,
        "role": role,
    }


class _JsonlEventWriter:
    def __init__(self, stream, *, close_stream):
        self._stream = stream
        self._close_stream = close_stream

    def emit(self, event, **payload):
        record = {
            "schema_version": PUBLIC_OUTPUT_SCHEMA_VERSION,
            "event": event,
            "time": round(time.time(), 3),
            "payload": payload,
        }
        print(json.dumps(record, ensure_ascii=False), file=self._stream, flush=True)

    def close(self):
        if self._close_stream:
            self._stream.close()


def _open_event_writer(events_jsonl):
    if not events_jsonl:
        return None
    if events_jsonl == '-':
        return _JsonlEventWriter(sys.stderr, close_stream=False)

    event_path = Path(events_jsonl)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    return _JsonlEventWriter(open(event_path, 'a', encoding='utf-8'), close_stream=True)


def _build_run_plan(args, toc, output_targets, chapter_indices, book_name, book_output_dir, source_text_path):
    audio_format = config['audio'].get('output_format', 'mp3')
    output_format = 'mp4' if args.video else audio_format
    keep_subtitles = getattr(args, 'srt', False) or getattr(args, 'keep_srt', False)
    generate_subtitles = keep_subtitles or (args.video and config['video'].get('subtitles', False))
    chapters = []
    for idx in chapter_indices:
        output_path = OUTPUT_DIR / output_targets[idx]
        existing_outputs = _existing_outputs(output_path)
        final_exists = any(item["format"] == output_format for item in existing_outputs)
        chapters.append({
            "index": idx,
            "display_path": toc[idx],
            "output_stem": str(output_path),
            "target_format": output_format,
            "will_skip_existing": final_exists,
            "existing_outputs": existing_outputs,
        })
    return {
        "schema_version": PUBLIC_OUTPUT_SCHEMA_VERSION,
        "status": "success",
        "mode": "plan",
        "book_name": book_name,
        "chapter_count": len(chapters),
        "output_format": output_format,
        "output_directory": str(book_output_dir),
        "source_text_file": str(source_text_path),
        "video": args.video,
        "generate_subtitles": generate_subtitles,
        "keep_subtitles": keep_subtitles,
        "write_chapter_manifest": getattr(args, 'chapter_manifest', False),
        "chapter_manifest_path": str(book_output_dir / CHAPTER_MANIFEST_FILE_NAME),
        "chapters": chapters,
    }


def _build_chapter_manifest(
    *,
    book_name,
    source_text_path,
    book_output_dir,
    output_format,
    elapsed,
    chapter_results,
):
    return {
        "schema_version": PUBLIC_OUTPUT_SCHEMA_VERSION,
        "book_name": book_name,
        "source_text_file": str(source_text_path),
        "output_directory": str(book_output_dir),
        "output_format": output_format,
        "elapsed_seconds": round(elapsed, 1),
        "chapter_count": len(chapter_results),
        "chapters": chapter_results,
    }


def cli_main_process():
    args = parse_arguments()

    # logging 仅用于库级代码（video.py 等）的 warning/error
    from rich.logging import RichHandler
    if args.json:
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format='%(message)s')
    else:
        logging.basicConfig(level=logging.WARNING,
                            handlers=[RichHandler(console=console, show_time=False, show_path=False)])

    # 应用 CLI 配置覆盖
    _apply_config_overrides(args)

    # --dump-config: 输出当前生效配置后退出
    if args.dump_config:
        import yaml
        serializable = json.loads(json.dumps(config, default=str))
        if args.json:
            print(json.dumps(serializable, ensure_ascii=False, indent=2))
        else:
            yaml.dump(serializable, sys.stdout, allow_unicode=True, default_flow_style=False)
        return 0

    audio_format = config['audio'].get('output_format', 'mp3')
    mp3_bitrate = config['audio'].get('mp3_bitrate', '128k')
    if args.landscape:
        config['video']['orientation'] = 'landscape'
    book_file_path = args.input_file_path
    if len(book_file_path) != 1 or '.' not in book_file_path[0]:
        _report_error(args, "invalid_args", "输入一个文件路径，且必须包含文件后缀")
        return 1
    input_path = Path(book_file_path[0])
    book_name = input_path.stem
    if not input_path.is_file():
        _report_error(args, "file_not_found", f"文件不存在: {book_file_path[0]}")
        return 1
    if input_path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
        supported = ', '.join(sorted(SUPPORTED_INPUT_SUFFIXES))
        _report_error(args, "unsupported_input_format", f"不支持的输入格式: {input_path.suffix or '无后缀'}，仅支持 {supported}")
        return 1

    if not args.json and not args.plan_json and not args.quiet:
        console.print(f"\n[bold]{book_name}[/bold]")

    try:
        raw_data, source_text_path, generated_txt = load_book_file(book_file_path[0])
    except (ValueError, RuntimeError) as e:
        _report_error(args, "input_conversion_failed", str(e))
        return 1

    if not args.json and not args.plan_json and not args.quiet and input_path.suffix.lower() != '.txt':
        status = '已生成' if generated_txt else '复用'
        console.print(f"  [dim]{status} 文本: {source_text_path}[/dim]")

    toc, output_targets, contents = construct_text_and_name(raw_data=raw_data, book_name=book_name)
    book_output_dir = _book_output_dir(book_name)

    if not toc:
        _report_error(args, "no_chapters", "未解析到任何章节，请检查文件内容")
        return 1

    if args.validate_paths:
        validation_entries = build_path_validation_entries(OUTPUT_DIR, toc, output_targets)
        if args.json:
            print(json.dumps({
                "schema_version": PUBLIC_OUTPUT_SCHEMA_VERSION,
                "status": "success",
                "mode": "validate_paths",
                "book_name": book_name,
                "chapter_count": len(validation_entries),
                "output_directory": str(book_output_dir),
                "source_text_file": str(source_text_path),
                "chapters": validation_entries,
            }, ensure_ascii=False))
        else:
            console.print(f"  [green]输出[/green]  {book_output_dir}")
            for item in validation_entries:
                console.print(
                    f"  [dim]{item['index']:>5}[/dim]: "
                    f"{item['display_path']} -> {item['output_stem']}"
                )
        return 0

    # 显示目录供用户选择章节
    if _should_show_catalog(args) and toc:
        for k, v in toc.items():
            console.print(f"  [dim]{k:>5}[/dim]: {v}")

    # 找到最后已生成的章节，提示用户断点位置
    for idx in sorted(toc.keys(), reverse=True):
        output_path = OUTPUT_DIR / output_targets[idx]
        if any(build_clip_output_path(output_path, 1, ext).is_file() for ext in ('.wav', '.mp4', '.mp3')):
            if _should_show_resume_hint(args):
                console.print(f"  [dim]上次生成到第 {idx} 章[/dim]")
            break

    try:
        if args.range is not None:
            span = parse_range_string(args.range, total=max(toc.keys()))
        elif args.plan_json:
            span = range(max(toc.keys()) + 1)
        else:
            span = ask_for_output_range(total=max(toc.keys()))
    except ValueError as e:
        _report_error(args, "invalid_range", str(e))
        return 1

    # 预计算章节列表（保留遇到空隙即停止的行为）
    chapter_indices = []
    for idx in span:
        if idx not in toc:
            break
        chapter_indices.append(idx)

    if args.plan_json:
        plan = _build_run_plan(args, toc, output_targets, chapter_indices, book_name, book_output_dir, source_text_path)
        print(json.dumps(plan, ensure_ascii=False))
        return 0

    if args.video or audio_format == 'mp3':
        try:
            _check_ffmpeg()
        except RuntimeError as e:
            _report_error(args, "ffmpeg_not_found", str(e))
            return 1

    event_writer = _open_event_writer(args.events_jsonl)
    start_time = time.time()
    chapters_generated = 0
    chapters_skipped = 0
    total_clips = 0
    generated_outputs = []
    artifacts = []
    skipped_chapters = []
    conversion_failures = []
    chapter_results = []
    show_progress = _should_show_chapter_progress(args, chapter_indices)
    generate_subtitles = args.srt or args.keep_srt or (args.video and config['video'].get('subtitles', False))

    try:
        if event_writer:
            event_writer.emit(
                "run_started",
                book_name=book_name,
                chapter_count=len(chapter_indices),
                output_format='mp4' if args.video else audio_format,
                output_directory=str(book_output_dir),
                source_text_file=str(source_text_path),
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
            disable=not show_progress,
        ) as progress:
            task = progress.add_task("准备中...", total=len(chapter_indices))

            for idx in chapter_indices:
                chapter_title = toc[idx].split('/')[-1]
                progress.update(task, description=f"[cyan]{chapter_title}[/cyan]")

                output_path = OUTPUT_DIR / output_targets[idx]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                chapter_generated_outputs = []
                chapter_artifacts = []
                chapter_failures = []
                chapter_existing_outputs = []
                if event_writer:
                    event_writer.emit(
                        "chapter_started",
                        chapter_index=idx,
                        display_path=toc[idx],
                        output_stem=str(output_path),
                    )

                try:
                    clip_num = generate_audio_clip(
                        text=''.join(contents[idx]),
                        output_path=str(output_path),
                        generate_subtitles=generate_subtitles,
                    )
                except RuntimeError as e:
                    if event_writer:
                        event_writer.emit(
                            "error",
                            error_code="tts_generation_failed",
                            message=str(e),
                            chapter_index=idx,
                        )
                    _report_error(args, "tts_generation_failed", str(e))
                    return 1

                if clip_num:
                    chapters_generated += 1
                    total_clips += len(clip_num)
                else:
                    chapters_skipped += 1
                    skipped_chapters.append({
                        "chapter_index": idx,
                        "display_path": toc[idx],
                        "output_stem": str(output_path),
                        "reason": "existing_output",
                    })

                if args.video:
                    from video import transform_wav_to_video
                    i = 1
                    while True:
                        mp4_path = build_clip_output_path(output_path, i, '.mp4')
                        wav_path = build_clip_output_path(output_path, i, '.wav')
                        mp3_path = build_clip_output_path(output_path, i, '.mp3')
                        if mp4_path.is_file():
                            i += 1
                            continue
                        if wav_path.is_file():
                            video_warnings = []
                            try:
                                output_file = transform_wav_to_video(
                                    number=idx,
                                    audio=str(wav_path),
                                    toc=toc[idx],
                                    resources_dir=RESOURCES_DIR,
                                    keep_subtitles=args.keep_srt,
                                    warnings=video_warnings,
                                )
                                if event_writer:
                                    for warning in video_warnings:
                                        event_writer.emit(
                                            "warning",
                                            chapter_index=idx,
                                            clip_index=i,
                                            **warning,
                                        )
                                generated_outputs.append(output_file)
                                chapter_generated_outputs.append(output_file)
                                artifact = _artifact_record(output_file, chapter_index=idx, clip_index=i, role="video")
                                artifacts.append(artifact)
                                chapter_artifacts.append(artifact)
                                if event_writer:
                                    event_writer.emit("artifact_created", artifact=artifact)
                            except RuntimeError as e:
                                failure = {
                                    "chapter_index": idx,
                                    "clip_index": i,
                                    "input_file": str(wav_path),
                                    "target_file": str(mp4_path),
                                    "message": str(e),
                                }
                                conversion_failures.append(failure)
                                chapter_failures.append(failure)
                                if event_writer:
                                    event_writer.emit("error", error_code="media_conversion_failed", **failure)
                        elif mp3_path.is_file():
                            video_warnings = []
                            try:
                                output_file = transform_wav_to_video(
                                    number=idx,
                                    audio=str(mp3_path),
                                    toc=toc[idx],
                                    resources_dir=RESOURCES_DIR,
                                    keep_subtitles=args.keep_srt,
                                    warnings=video_warnings,
                                )
                                if event_writer:
                                    for warning in video_warnings:
                                        event_writer.emit(
                                            "warning",
                                            chapter_index=idx,
                                            clip_index=i,
                                            **warning,
                                        )
                                generated_outputs.append(output_file)
                                chapter_generated_outputs.append(output_file)
                                artifact = _artifact_record(output_file, chapter_index=idx, clip_index=i, role="video")
                                artifacts.append(artifact)
                                chapter_artifacts.append(artifact)
                                if event_writer:
                                    event_writer.emit("artifact_created", artifact=artifact)
                            except RuntimeError as e:
                                failure = {
                                    "chapter_index": idx,
                                    "clip_index": i,
                                    "input_file": str(mp3_path),
                                    "target_file": str(mp4_path),
                                    "message": str(e),
                                }
                                conversion_failures.append(failure)
                                chapter_failures.append(failure)
                                if event_writer:
                                    event_writer.emit("error", error_code="media_conversion_failed", **failure)
                        else:
                            break
                        i += 1
                elif audio_format == 'mp3':
                    for i in clip_num:
                        wav_path = build_clip_output_path(output_path, i, '.wav')
                        mp3_path = build_clip_output_path(output_path, i, '.mp3')
                        try:
                            output_file = convert_wav_to_mp3(wav_path, bitrate=mp3_bitrate)
                            generated_outputs.append(output_file)
                            chapter_generated_outputs.append(output_file)
                            artifact = _artifact_record(output_file, chapter_index=idx, clip_index=i, role="audio")
                            artifacts.append(artifact)
                            chapter_artifacts.append(artifact)
                            if event_writer:
                                event_writer.emit("artifact_created", artifact=artifact)
                        except RuntimeError as e:
                            failure = {
                                "chapter_index": idx,
                                "clip_index": i,
                                "input_file": str(wav_path),
                                "target_file": str(mp3_path),
                                "message": str(e),
                            }
                            conversion_failures.append(failure)
                            chapter_failures.append(failure)
                            if event_writer:
                                event_writer.emit("error", error_code="media_conversion_failed", **failure)
                elif audio_format == 'wav':
                    for i in clip_num:
                        output_file = str(build_clip_output_path(output_path, i, '.wav'))
                        generated_outputs.append(output_file)
                        chapter_generated_outputs.append(output_file)
                        artifact = _artifact_record(output_file, chapter_index=idx, clip_index=i, role="audio")
                        artifacts.append(artifact)
                        chapter_artifacts.append(artifact)
                        if event_writer:
                            event_writer.emit("artifact_created", artifact=artifact)
                if args.srt:
                    for i in clip_num:
                        srt_path = build_clip_output_path(output_path, i, '.srt')
                        if srt_path.is_file():
                            output_file = str(srt_path)
                            generated_outputs.append(output_file)
                            chapter_generated_outputs.append(output_file)
                            artifact = _artifact_record(output_file, chapter_index=idx, clip_index=i, role="subtitle")
                            artifacts.append(artifact)
                            chapter_artifacts.append(artifact)
                            if event_writer:
                                event_writer.emit("artifact_created", artifact=artifact)

                if not clip_num:
                    chapter_status = "skipped"
                    chapter_existing_outputs = _existing_outputs(output_path)
                elif chapter_failures:
                    chapter_status = "error"
                else:
                    chapter_status = "generated"
                chapter_result = {
                    "index": idx,
                    "display_path": toc[idx],
                    "output_stem": str(output_path),
                    "status": chapter_status,
                    "clip_count": len(clip_num),
                    "existing_outputs": chapter_existing_outputs,
                    "generated_outputs": chapter_generated_outputs,
                    "artifacts": chapter_artifacts,
                    "failures": chapter_failures,
                }
                chapter_results.append(chapter_result)
                if event_writer:
                    event_writer.emit("chapter_completed", **chapter_result)

                progress.advance(task)

        elapsed = time.time() - start_time
        fmt = 'mp4' if args.video else audio_format
        out_dir = book_output_dir
        chapter_manifest_path = None
        if args.chapter_manifest:
            chapter_manifest_path = save_chapter_manifest(
                out_dir / CHAPTER_MANIFEST_FILE_NAME,
                _build_chapter_manifest(
                    book_name=book_name,
                    source_text_path=source_text_path,
                    book_output_dir=out_dir,
                    output_format=fmt,
                    elapsed=elapsed,
                    chapter_results=chapter_results,
                ),
            )
            artifact = _artifact_record(chapter_manifest_path, chapter_index=None, clip_index=None, role="manifest")
            artifacts.append(artifact)
            if event_writer:
                event_writer.emit("artifact_created", artifact=artifact)

        status = "error" if conversion_failures else "success"
        if event_writer:
            event_writer.emit(
                "run_completed",
                status=status,
                chapters_generated=chapters_generated,
                chapters_skipped=chapters_skipped,
                total_clips=total_clips,
                artifact_count=len(artifacts),
                elapsed_seconds=round(elapsed, 1),
            )

        if args.json:
            result = {
                "schema_version": PUBLIC_OUTPUT_SCHEMA_VERSION,
                "status": status,
                "book_name": book_name,
                "chapters_generated": chapters_generated,
                "chapters_skipped": chapters_skipped,
                "total_clips": total_clips,
                "output_format": fmt,
                "output_directory": str(out_dir),
                "source_text_file": str(source_text_path),
                "elapsed_seconds": round(elapsed, 1),
                "generated_outputs": generated_outputs,
                "artifacts": artifacts,
                "skipped_chapters": skipped_chapters,
            }
            if chapter_manifest_path:
                result["chapter_manifest"] = chapter_manifest_path
            if conversion_failures:
                result["error"] = "media_conversion_failed"
                result["failed_outputs"] = conversion_failures
            print(json.dumps(result, ensure_ascii=False))
        elif _should_print_summary(args, conversion_failures):
            summary = (
                f"[green]章节[/green]  {chapters_generated} 已生成, {chapters_skipped} 跳过\n"
                f"[green]片段[/green]  {total_clips}\n"
                f"[green]格式[/green]  {fmt}\n"
                f"[green]文件[/green]  {len(generated_outputs)} 新生成\n"
                f"[green]输出[/green]  {out_dir}\n"
                f"[green]耗时[/green]  {_format_duration(elapsed)}"
            )
            if chapter_manifest_path:
                summary += f"\n[green]清单[/green]  {chapter_manifest_path}"
            if conversion_failures:
                failed_paths = '\n'.join(f"  {item['target_file']}: {item['message']}" for item in conversion_failures)
                summary += f"\n[red]转换失败[/red]  {len(conversion_failures)}\n{failed_paths}"
                console.print(Panel(summary, title="[bold red]部分失败[/bold red]", border_style="red"))
            else:
                console.print(Panel(summary, title="[bold green]完成[/bold green]", border_style="green"))
        return 1 if conversion_failures else 0
    finally:
        if event_writer:
            event_writer.close()


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='txt2audio',
        description='Convert Chinese text novels to audiobooks using CosyVoice TTS.',
        epilog='Examples:\n'
               '  txt2audio novel.txt --range all\n'
               '  txt2audio novel.epub --range all\n'
               '  txt2audio novel.txt --validate-paths --json\n'
               '  txt2audio novel.txt --plan-json\n'
               '  txt2audio novel.txt --range all --chapter-manifest\n'
               '  txt2audio novel.txt --range all --json --events-jsonl events.jsonl\n'
               '  txt2audio novel.txt --video --range 0~8 --json\n'
               '  txt2audio novel.txt --range all --srt\n'
               '  txt2audio novel.txt --range all --speed 0.95 --set audio.mp3_bitrate=192k\n'
               '  txt2audio --dump-config --json\n',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('input_file_path', metavar='input_file_path', type=str, nargs='?',
                        help='path to the book file (.txt/.epub/.mobi), required unless --dump-config')
    parser.add_argument('--video', action='store_true',
                        help='generate MP4 video with cover image (default: audio only)')
    parser.add_argument('--landscape', action='store_true',
                        help='use landscape (horizontal) video orientation (overrides config)')
    parser.add_argument('--srt', action='store_true',
                        help='export SRT subtitle sidecar files with audio output')
    parser.add_argument('--keep-srt', action='store_true',
                        help='keep generated SRT files after video subtitle burn-in')
    parser.add_argument('--range', type=str, default=None,
                        help='chapter range, e.g. "all", "8", "0~8" (skips interactive prompt)')
    parser.add_argument('--json', action='store_true',
                        help='output result as JSON to stdout (machine-readable)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='suppress progress output, only show errors')
    parser.add_argument('--speed', type=float, default=None,
                        help='TTS speed override (e.g. 0.95)')
    parser.add_argument('--output-format', type=str, choices=['mp3', 'wav'], default=None,
                        help='audio output format override')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='output directory override')
    parser.add_argument('--set', action='append', metavar='KEY=VALUE', default=None,
                        help='override any config.yaml value, e.g. --set tts.speed=0.9')
    parser.add_argument('--dump-config', action='store_true',
                        help='print effective config as YAML (or JSON with --json) and exit')
    parser.add_argument('--validate-paths', action='store_true',
                        help='parse chapters and output paths without loading the TTS model or generating media')
    parser.add_argument('--plan-json', action='store_true',
                        help='output planned chapters and existing outputs as JSON without loading the TTS model')
    parser.add_argument('--chapter-manifest', action='store_true',
                        help='write output chapter_manifest.json after a generation run')
    parser.add_argument('--events-jsonl', type=str, default=None,
                        help='write machine-readable progress events to a JSONL file, or "-" for stderr')

    args = parser.parse_args()

    if args.set is None:
        args.set = []

    # nargs='?' 返回标量或 None，统一包装为列表以兼容下游代码
    if args.input_file_path is not None:
        args.input_file_path = [args.input_file_path]
    elif not args.dump_config:
        parser.error('input_file_path is required unless --dump-config is used')
    else:
        args.input_file_path = []

    return args


def parse_range_string(var, total):
    var = var.strip()
    if len(var) == 0 or var == 'all':
        return range(total + 1)

    if not re.fullmatch(r'\d+(?:[~-]\d+)?', var):
        raise ValueError("请输入 all、单个非负数字，或者范围，例如 8 或 0~8")

    indices = re.split('[~-]', var)
    if len(indices) == 1:
        s = int(indices[0])
        if s > total:
            raise ValueError(f"章节编号超出范围: {s}，最大编号为 {total}")
        return range(s, s + 1)

    s = int(indices[0])
    e = int(indices[1])
    if s > e:
        raise ValueError(f"章节范围起点不能大于终点: {s}~{e}")
    if e > total:
        raise ValueError(f"章节编号超出范围: {e}，最大编号为 {total}")
    return range(s, e + 1)


def ask_for_output_range(total):
    if not sys.stdin.isatty():
        return range(total + 1)
    var = input("请输入转换范围, (all 表示全部): \n")
    return parse_range_string(var, total)
