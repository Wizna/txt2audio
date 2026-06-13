import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from typing import List, Dict
import sys
import subprocess as _subprocess
import argparse
import time
import logging
import json
import shutil

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
            logger.error(f"Failed to load CosyVoice model from: {MODEL_DIR}")
            logger.error(f"Error: {e}")
            logger.error("Please ensure the model is downloaded:")
            logger.error("  uv run python -c \"from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')\"")
            raise SystemExit(1)
    return _tts

book_delimiter = config['text_processing']['book_delimiter']
SUPPORTED_INPUT_SUFFIXES = {'.txt', '.epub', '.mobi'}
CONVERTED_TEXT_SUFFIX = '.txt2audio.txt'


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
        logger.error(f'MP3 conversion failed (code {ret.returncode}), keeping {wav_path}')
        if stderr:
            logger.error(stderr)
        raise RuntimeError(f'MP3 conversion failed for {wav_path} (code {ret.returncode})')


def check_export_file_exists(output_path, video_clip_index):
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
    for ext in ('.mp4', '.mp3', '.wav'):
        path = build_clip_output_path(output_stem, video_clip_index, ext)
        if path.is_file():
            if path.stat().st_size == 0:
                os.remove(path)
                logger.debug(f"Removed empty file: {path}")
                continue
            logger.debug(f"{path} is already generated !")
            return False
    return True


def generate_audio_clip(text: str, output_path: str, sample_rate=None, generate_subtitles: bool = True):
    """将一章文本转为音频，按 MAX_CHARS_PER_CLIP 切分为多个片段（-1 则不分片）。
    同时收集句级时间戳，生成 SRT 字幕文件。"""
    torch_module, _ = _load_torch_modules()
    from subtitle import save_subtitle_file

    cosyvoice = get_tts()
    if sample_rate is None:
        sample_rate = cosyvoice.sample_rate

    word_count = 0
    video_clip_index = 1
    exported_clip_indices = []
    wav_chunks = []
    subtitle_entries = []
    current_time = 0.0
    export = check_export_file_exists(output_path=output_path, video_clip_index=video_clip_index)

    silence = None
    if INTER_SENTENCE_SILENCE_MS > 0:
        silence = torch_module.zeros(1, int(sample_rate * INTER_SENTENCE_SILENCE_MS / 1000))

    # 在句尾标点处拆分为独立句子，确保每个句子有精确的时间戳
    raw_sentences = [s.strip() for s in re.split(r'(?<=[。！？])', text) if s.strip()]

    for raw_sentence in raw_sentences:
        tts_sentence = mask_punctuations(text=annotate_polyphones(raw_sentence))
        sub_sentence = mask_punctuations(text=raw_sentence).rstrip('。')
        if not tts_sentence or not sub_sentence:
            continue

        if export:
            sentence_start = current_time
            if silence is not None and wav_chunks:
                wav_chunks.append(silence)
                current_time += INTER_SENTENCE_SILENCE_MS / 1000
                sentence_start = current_time
            for chunk in cosyvoice.inference_zero_shot(
                tts_sentence, PROMPT_TEXT, SPEAKER_WAV,
                zero_shot_spk_id='narrator', stream=False, speed=SPEED
            ):
                wav_chunks.append(chunk['tts_speech'])
                current_time += chunk['tts_speech'].shape[-1] / sample_rate
            subtitle_entries.append((sentence_start, current_time, sub_sentence))
        word_count += get_word_num(text=raw_sentence)

        if MAX_CHARS_PER_CLIP > 0 and word_count > MAX_CHARS_PER_CLIP:
            combined = torch_module.cat(wav_chunks, dim=-1) if wav_chunks else None
            save_audio_file(combined, sample_rate, output_path, video_clip_index, exported_clip_indices)
            if generate_subtitles and export and subtitle_entries:
                save_subtitle_file(subtitle_entries, output_path, video_clip_index)
            video_clip_index += 1
            wav_chunks = []
            word_count = 0
            subtitle_entries = []
            current_time = 0.0
            export = check_export_file_exists(output_path=output_path, video_clip_index=video_clip_index)

    combined = torch_module.cat(wav_chunks, dim=-1) if wav_chunks else None
    save_audio_file(combined, sample_rate, output_path, video_clip_index, exported_clip_indices)
    if generate_subtitles and export and subtitle_entries:
        save_subtitle_file(subtitle_entries, output_path, video_clip_index)
    return exported_clip_indices


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


def _check_ffmpeg():
    """Check if ffmpeg is available on the system."""
    try:
        _subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (FileNotFoundError, _subprocess.CalledProcessError):
        logger.error("ffmpeg not found. Please install ffmpeg.")
        logger.error("  macOS:   brew install ffmpeg")
        logger.error("  Windows: winget install ffmpeg  (or download from https://ffmpeg.org/download.html)")
        raise SystemExit(1)


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


def _json_error(error_code, message):
    """输出 JSON 格式的错误信息到 stdout。"""
    print(json.dumps({"status": "error", "error": error_code, "message": message}))


def _report_error(args, error_code, message):
    """错误输出：JSON 模式写 stdout，人类模式写 stderr rich 格式。"""
    if args.json:
        _json_error(error_code, message)
    else:
        console.print(f"[bold red]Error:[/bold red] {message}")


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
    if (args.video or audio_format == 'mp3') and not args.validate_paths:
        _check_ffmpeg()
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

    if not args.json:
        console.print(f"\n[bold]{book_name}[/bold]")

    try:
        raw_data, source_text_path, generated_txt = load_book_file(book_file_path[0])
    except (ValueError, RuntimeError) as e:
        _report_error(args, "input_conversion_failed", str(e))
        return 1

    if not args.json and input_path.suffix.lower() != '.txt':
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
    if not args.json and not args.quiet and toc:
        for k, v in toc.items():
            console.print(f"  [dim]{k:>5}[/dim]: {v}")

    # 找到最后已生成的章节，提示用户断点位置
    for idx in sorted(toc.keys(), reverse=True):
        output_path = OUTPUT_DIR / output_targets[idx]
        if any(build_clip_output_path(output_path, 1, ext).is_file() for ext in ('.wav', '.mp4', '.mp3')):
            if not args.json:
                console.print(f"  [dim]上次生成到第 {idx} 章[/dim]")
            break

    try:
        if args.range is not None:
            span = parse_range_string(args.range, total=max(toc.keys()))
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

    start_time = time.time()
    chapters_generated = 0
    chapters_skipped = 0
    total_clips = 0
    conversion_failures = []
    show_progress = not args.json and not args.quiet
    generate_subtitles = args.video and config['video'].get('subtitles', False)

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

            clip_num = generate_audio_clip(
                text=''.join(contents[idx]),
                output_path=str(output_path),
                generate_subtitles=generate_subtitles,
            )

            if clip_num:
                chapters_generated += 1
                total_clips += len(clip_num)
            else:
                chapters_skipped += 1

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
                        try:
                            transform_wav_to_video(number=idx, audio=str(wav_path), toc=toc[idx],
                                                   resources_dir=RESOURCES_DIR)
                        except RuntimeError as e:
                            conversion_failures.append({
                                "chapter_index": idx,
                                "clip_index": i,
                                "input_file": str(wav_path),
                                "target_file": str(mp4_path),
                                "message": str(e),
                            })
                    elif mp3_path.is_file():
                        try:
                            transform_wav_to_video(number=idx, audio=str(mp3_path), toc=toc[idx],
                                                   resources_dir=RESOURCES_DIR)
                        except RuntimeError as e:
                            conversion_failures.append({
                                "chapter_index": idx,
                                "clip_index": i,
                                "input_file": str(mp3_path),
                                "target_file": str(mp4_path),
                                "message": str(e),
                            })
                    else:
                        break
                    i += 1
            elif audio_format == 'mp3':
                for i in clip_num:
                    wav_path = build_clip_output_path(output_path, i, '.wav')
                    mp3_path = build_clip_output_path(output_path, i, '.mp3')
                    try:
                        convert_wav_to_mp3(wav_path, bitrate=mp3_bitrate)
                    except RuntimeError as e:
                        conversion_failures.append({
                            "chapter_index": idx,
                            "clip_index": i,
                            "input_file": str(wav_path),
                            "target_file": str(mp3_path),
                            "message": str(e),
                        })

            progress.advance(task)

    elapsed = time.time() - start_time
    fmt = 'mp4' if args.video else audio_format
    out_dir = book_output_dir

    if args.json:
        result = {
            "status": "error" if conversion_failures else "success",
            "book_name": book_name,
            "chapters_generated": chapters_generated,
            "chapters_skipped": chapters_skipped,
            "total_clips": total_clips,
            "output_format": fmt,
            "output_directory": str(out_dir),
            "source_text_file": str(source_text_path),
            "elapsed_seconds": round(elapsed, 1),
        }
        if conversion_failures:
            result["error"] = "media_conversion_failed"
            result["failures"] = conversion_failures
        print(json.dumps(result, ensure_ascii=False))
    else:
        summary = (
            f"[green]章节[/green]  {chapters_generated} 已生成, {chapters_skipped} 跳过\n"
            f"[green]片段[/green]  {total_clips}\n"
            f"[green]格式[/green]  {fmt}\n"
            f"[green]输出[/green]  {out_dir}\n"
            f"[green]耗时[/green]  {_format_duration(elapsed)}"
        )
        if conversion_failures:
            failed_paths = '\n'.join(f"  {item['target_file']}: {item['message']}" for item in conversion_failures)
            summary += f"\n[red]转换失败[/red]  {len(conversion_failures)}\n{failed_paths}"
            console.print(Panel(summary, title="[bold red]部分失败[/bold red]", border_style="red"))
        else:
            console.print(Panel(summary, title="[bold green]完成[/bold green]", border_style="green"))
    return 1 if conversion_failures else 0


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='txt2audio',
        description='Convert Chinese text novels to audiobooks using CosyVoice TTS.',
        epilog='Examples:\n'
               '  txt2audio novel.txt --range all\n'
               '  txt2audio novel.epub --range all\n'
               '  txt2audio novel.txt --validate-paths --json\n'
               '  txt2audio novel.txt --video --range 0~8 --json\n'
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
